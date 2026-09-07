"""Fit an articulated SUPR foot to one prepared normal-shoe support."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from foot_prior.alignment import build_support_foot_fit
from foot_prior.mesh import (
    TriangleMesh,
    combine_colored_meshes,
    load_triangle_mesh,
    save_triangle_mesh,
    transform_mesh,
)
from foot_prior.normalization import (
    EXPECTED_SHOE_COORDINATE_SYSTEM,
    NORMAL_SHOE_PROFILE,
    SHOE_SIDE,
    validate_shoe_frame_metadata,
)
from foot_prior.supr_foot import load_neutral_supr_foot, load_posable_supr_foot


ARTIFACT_NAMES = (
    "support_fit.json",
    "foot_support_fitted.ply",
    "footbed_normalized.ply",
    "support_fit_overlay.ply",
)
SHOE_COLOR = np.asarray([150, 150, 150, 255], dtype=np.uint8)
FOOT_COLOR = np.asarray([45, 105, 220, 255], dtype=np.uint8)
FOOTBED_COLOR = np.asarray([40, 180, 90, 255], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit SUPR ankle and midfoot pitch to a prepared normalized "
            "normal-shoe support."
        )
    )
    parser.add_argument("--preparation-dir", required=True, type=Path)
    parser.add_argument("--supr-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the four known support-fit artifacts.",
    )
    return parser.parse_args()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _array(payload: Any, shape: tuple[int, ...], label: str) -> np.ndarray:
    values = np.asarray(payload, dtype=np.float64)
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError(f"{label} must be a finite array with shape {shape}")
    return values


def _require_matching_bounds(
    actual: np.ndarray, expected: Any, label: str
) -> None:
    recorded = _array(expected, (2, 3), f"recorded {label} bounds")
    if not np.allclose(actual, recorded, atol=1e-6, rtol=1e-6):
        raise ValueError(f"{label} does not match shoe_preparation.json")


def _load_prepared_inputs(
    preparation_dir: Path,
) -> tuple[
    TriangleMesh,
    TriangleMesh,
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    metadata_path = preparation_dir / "shoe_preparation.json"
    shoe_path = preparation_dir / "shoe_normalized.ply"
    footbed_path = preparation_dir / "footbed_surface.ply"
    for path in (metadata_path, shoe_path, footbed_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    payload = _load_json_object(metadata_path)
    if payload.get("schema_version") != 1:
        raise ValueError("shoe_preparation.json must use schema_version 1")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(
        inputs.get("canonicalization"), str
    ):
        raise ValueError(
            "shoe_preparation.json is missing inputs.canonicalization"
        )
    canonicalization_path = Path(inputs["canonicalization"]).expanduser().resolve(
        strict=True
    )
    canonical_profile = validate_shoe_frame_metadata(canonicalization_path)
    recorded_profile = payload.get("shoe_profile")
    if recorded_profile is not None and recorded_profile != canonical_profile:
        raise ValueError(
            "shoe profile disagrees between preparation and canonicalization metadata"
        )
    if canonical_profile != NORMAL_SHOE_PROFILE:
        raise ValueError(
            "articulated SUPR support fitting currently accepts "
            "shoe_profile='normal' "
            f"only; received {canonical_profile!r}"
        )

    coordinate_contract = payload.get("coordinate_contract")
    if not isinstance(coordinate_contract, dict):
        raise ValueError("shoe_preparation.json is missing coordinate_contract")
    if (
        coordinate_contract.get("coordinate_system")
        != EXPECTED_SHOE_COORDINATE_SYSTEM
        or coordinate_contract.get("side") != SHOE_SIDE
    ):
        raise ValueError("shoe_preparation.json has an incompatible coordinate contract")

    normalization = payload.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("shoe_preparation.json is missing normalization")
    shoe_to_normalized = _array(
        normalization.get("shoe_to_normalized"),
        (4, 4),
        "shoe_to_normalized",
    )
    normalized_to_shoe = _array(
        normalization.get("normalized_to_shoe"),
        (4, 4),
        "normalized_to_shoe",
    )
    centerline = normalization.get("centerline")
    if not isinstance(centerline, dict):
        raise ValueError("normalization is missing its centerline")
    centerline_xz = np.asarray(centerline.get("normalized_xz"), dtype=np.float64)

    shoe = load_triangle_mesh(shoe_path)
    footbed = load_triangle_mesh(footbed_path)
    bounds = normalization.get("bounds")
    if not isinstance(bounds, dict):
        raise ValueError("normalization is missing recorded bounds")
    _require_matching_bounds(
        shoe.bounds, bounds.get("normalized_shoe"), "normalized shoe"
    )
    footbed_selection = payload.get("footbed_selection")
    if not isinstance(footbed_selection, dict):
        raise ValueError("shoe_preparation.json is missing footbed_selection")
    _require_matching_bounds(
        footbed.bounds,
        footbed_selection.get("bounds"),
        "original-frame footbed",
    )
    if int(footbed_selection.get("face_count", -1)) != len(footbed.faces):
        raise ValueError("footbed face count does not match shoe_preparation.json")
    if int(footbed_selection.get("vertex_count", -1)) != len(footbed.vertices):
        raise ValueError("footbed vertex count does not match shoe_preparation.json")
    grid_shape = np.asarray(footbed_selection.get("grid_shape"), dtype=np.int64)
    grid_x_bounds = _array(
        footbed_selection.get("grid_x_bounds"),
        (2,),
        "footbed grid X bounds",
    )
    if grid_shape.shape != (2,) or np.any(grid_shape <= 0):
        raise ValueError("footbed grid_shape must contain two positive dimensions")
    original_cell_spacing = float(np.ptp(grid_x_bounds) / grid_shape[0])
    normalized_x_scale = float(np.linalg.norm(shoe_to_normalized[:3, 0]))
    support_grid_cell_spacing = original_cell_spacing * normalized_x_scale
    return (
        shoe,
        footbed,
        payload,
        shoe_to_normalized,
        normalized_to_shoe,
        centerline_xz,
        support_grid_cell_spacing,
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    preparation_dir = args.preparation_dir.expanduser().resolve(strict=True)
    supr_path = args.supr_model.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    targets = {name: output_dir / name for name in ARTIFACT_NAMES}
    existing = [path for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "support-fit artifacts already exist: "
            f"{formatted}; pass --overwrite to replace them"
        )

    (
        normalized_shoe,
        original_footbed,
        preparation,
        shoe_to_normalized,
        normalized_to_shoe,
        centerline_xz,
        support_grid_cell_spacing,
    ) = _load_prepared_inputs(preparation_dir)
    normalized_footbed = transform_mesh(original_footbed, shoe_to_normalized)
    neutral_foot = load_neutral_supr_foot(supr_path)
    posable_foot = load_posable_supr_foot(supr_path, num_betas=10)
    support_fit = build_support_foot_fit(
        supr_model=posable_foot,
        neutral_foot_mesh=neutral_foot,
        normalized_shoe_mesh=normalized_shoe,
        normalized_support_mesh=normalized_footbed,
        normalized_centerline_xz=centerline_xz,
        shoe_to_normalized=shoe_to_normalized,
        normalized_to_shoe=normalized_to_shoe,
        support_grid_cell_spacing=support_grid_cell_spacing,
    )
    aligned_foot = TriangleMesh(support_fit.aligned_vertices, neutral_foot.faces)
    overlay = combine_colored_meshes(
        (normalized_shoe, SHOE_COLOR), (aligned_foot, FOOT_COLOR)
    )
    payload: dict[str, object] = {
        "schema_version": 2,
        "shoe_profile": NORMAL_SHOE_PROFILE,
        "inputs": {
            "preparation_directory": str(preparation_dir),
            "shoe_preparation": str(preparation_dir / "shoe_preparation.json"),
            "normalized_shoe": str(preparation_dir / "shoe_normalized.ply"),
            "original_footbed": str(preparation_dir / "footbed_surface.ply"),
            "supr_model": str(supr_path),
        },
        "source_preparation_schema_version": preparation["schema_version"],
        **support_fit.to_dict(),
    }
    payload_json = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    output_dir.mkdir(parents=True, exist_ok=True)
    foot_colors = np.tile(FOOT_COLOR, (len(aligned_foot.vertices), 1))
    footbed_colors = np.tile(
        FOOTBED_COLOR, (len(normalized_footbed.vertices), 1)
    )
    save_triangle_mesh(
        targets["foot_support_fitted.ply"], aligned_foot, foot_colors
    )
    save_triangle_mesh(
        targets["footbed_normalized.ply"],
        normalized_footbed,
        footbed_colors,
    )
    save_triangle_mesh(targets["support_fit_overlay.ply"], overlay)
    targets["support_fit.json"].write_text(payload_json, encoding="utf-8")
    return payload


def main() -> None:
    parser_args = parse_args()
    try:
        payload = run(parser_args)
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(f"SUPR support fit failed: {error}") from error
    contact = payload["support_contact"]
    supr = payload["supr"]
    assert isinstance(contact, dict)
    assert isinstance(supr, dict)
    region_contact = contact["face_centroids_by_region"]
    angles = supr["selected_angles_degrees"]
    assert isinstance(region_contact, dict)
    assert isinstance(angles, dict)
    print(
        f"wrote articulated SUPR support-fit artifacts to "
        f"{parser_args.output_dir.expanduser().resolve()}"
    )
    print(
        f"ankle {float(angles['ankle_pitch']):+.2f} degrees; "
        f"midfoot {float(angles['midfoot_pitch']):+.2f} degrees; "
        f"heel RMS {float(region_contact['heel']['rms_gap']):.6f}; "
        f"forefoot RMS {float(region_contact['forefoot']['rms_gap']):.6f}"
    )
    sizing = payload["sizing"]
    assert isinstance(sizing, dict)
    print(
        f"foot ratio {float(sizing['foot_length_ratio']):.6f}; "
        f"toe allowance {float(sizing['toe_allowance_mm']):.2f} mm "
        f"(anchor {float(sizing['anchor_toe_allowance_mm']):.2f} mm)"
    )


if __name__ == "__main__":
    main()
