#!/usr/bin/env python3
"""Measure cavity clearance around one fitted SUPR foot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from foot_prior.cavity import analyze_fitted_foot_cavity
from foot_prior.mesh import TriangleMesh, load_triangle_mesh, save_triangle_mesh
from foot_prior.normalization import (
    EXPECTED_SHOE_COORDINATE_SYSTEM,
    NORMAL_SHOE_PROFILE,
    SHOE_SIDE,
    validate_shoe_frame_metadata,
)


ARTIFACT_NAMES = (
    "cavity_analysis.json",
    "foot_clearance_colored.ply",
    "cavity_overlay.ply",
)
SHOE_COLOR = np.asarray([150, 150, 150, 255], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure footbed contact and non-footbed shoe-surface clearance "
            "without changing the fitted foot."
        )
    )
    parser.add_argument("--preparation-dir", required=True, type=Path)
    parser.add_argument("--support-fit-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the three known cavity-analysis artifacts.",
    )
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return payload


def _finite_array(
    value: Any, shape: tuple[int, ...], label: str
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite array with shape {shape}")
    return array


def _require_bounds(
    actual: np.ndarray, value: Any, label: str
) -> None:
    expected = _finite_array(value, (2, 3), f"recorded {label} bounds")
    if not np.allclose(actual, expected, atol=1e-6, rtol=1e-6):
        raise ValueError(f"{label} does not match its recorded bounds")


def _validate_inverse_pair(
    forward: np.ndarray, inverse: np.ndarray, label: str
) -> None:
    identity = np.eye(4, dtype=np.float64)
    if not np.allclose(inverse @ forward, identity, atol=1e-9, rtol=0.0):
        raise ValueError(f"{label} matrices are not mutual inverses")
    if not np.allclose(forward @ inverse, identity, atol=1e-9, rtol=0.0):
        raise ValueError(f"{label} matrices are not mutual inverses")


def _validate_normal_preparation(
    preparation_dir: Path,
) -> dict[str, Any]:
    metadata_path = preparation_dir / "shoe_preparation.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    payload = _load_json(metadata_path)
    if payload.get("schema_version") != 1:
        raise ValueError("shoe_preparation.json must use schema_version 1")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict) or not isinstance(
        inputs.get("canonicalization"), str
    ):
        raise ValueError("shoe_preparation.json is missing canonicalization input")
    canonicalization = Path(inputs["canonicalization"]).expanduser().resolve(
        strict=True
    )
    authoritative_profile = validate_shoe_frame_metadata(canonicalization)
    if payload.get("shoe_profile") != authoritative_profile:
        raise ValueError(
            "shoe profile disagrees between preparation and canonicalization metadata"
        )
    if authoritative_profile != NORMAL_SHOE_PROFILE:
        raise ValueError(
            "cavity analysis currently accepts shoe_profile='normal' only; "
            f"received {authoritative_profile!r}"
        )
    contract = payload.get("coordinate_contract")
    if not isinstance(contract, dict) or (
        contract.get("coordinate_system") != EXPECTED_SHOE_COORDINATE_SYSTEM
        or contract.get("side") != SHOE_SIDE
    ):
        raise ValueError(
            "shoe_preparation.json has an incompatible coordinate contract"
        )
    return payload


def _load_inputs(
    preparation_dir: Path, support_fit_dir: Path
) -> tuple[
    TriangleMesh,
    TriangleMesh,
    TriangleMesh,
    dict[str, Any],
    dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
]:
    preparation = _validate_normal_preparation(preparation_dir)
    support_metadata_path = support_fit_dir / "support_fit.json"
    shoe_path = preparation_dir / "shoe_normalized.ply"
    footbed_path = support_fit_dir / "footbed_normalized.ply"
    foot_path = support_fit_dir / "foot_support_fitted.ply"
    for path in (support_metadata_path, shoe_path, footbed_path, foot_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    support_fit = _load_json(support_metadata_path)
    if support_fit.get("schema_version") != 2:
        raise ValueError(
            "support_fit.json must use schema_version 2; regenerate "
            "Checkpoint 5 with the current alignment runner"
        )
    if support_fit.get("shoe_profile") != NORMAL_SHOE_PROFILE:
        raise ValueError("support_fit.json must describe shoe_profile='normal'")
    support_inputs = support_fit.get("inputs")
    if not isinstance(support_inputs, dict) or not isinstance(
        support_inputs.get("preparation_directory"), str
    ):
        raise ValueError("support_fit.json is missing its preparation input")
    recorded_preparation = Path(
        support_inputs["preparation_directory"]
    ).expanduser().resolve(strict=True)
    if recorded_preparation != preparation_dir:
        raise ValueError(
            "support fit was not generated from the supplied preparation directory"
        )

    normalized_shoe = load_triangle_mesh(shoe_path)
    normalized_footbed = load_triangle_mesh(footbed_path)
    fitted_foot = load_triangle_mesh(foot_path)
    normalization = preparation.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("shoe_preparation.json is missing normalization")
    preparation_bounds = normalization.get("bounds")
    if not isinstance(preparation_bounds, dict):
        raise ValueError("shoe preparation is missing normalized bounds")
    _require_bounds(
        normalized_shoe.bounds,
        preparation_bounds.get("normalized_shoe"),
        "normalized shoe",
    )
    fit_bounds = support_fit.get("bounds")
    if not isinstance(fit_bounds, dict):
        raise ValueError("support_fit.json is missing bounds")
    _require_bounds(
        normalized_shoe.bounds,
        fit_bounds.get("normalized_shoe"),
        "support-fit normalized shoe",
    )
    _require_bounds(
        normalized_footbed.bounds,
        fit_bounds.get("normalized_support"),
        "normalized footbed",
    )
    _require_bounds(
        fitted_foot.bounds,
        fit_bounds.get("aligned_foot"),
        "fitted foot",
    )

    selection = preparation.get("footbed_selection")
    if not isinstance(selection, dict):
        raise ValueError("shoe_preparation.json is missing footbed_selection")
    footbed_faces = np.asarray(
        selection.get("original_face_indices"), dtype=np.int64
    )
    if len(footbed_faces) != len(normalized_footbed.faces):
        raise ValueError("normalized footbed face count disagrees with preparation")
    regions = support_fit.get("contact_regions")
    if not isinstance(regions, dict):
        raise ValueError("support_fit.json is missing contact_regions")
    plantar_vertices = np.asarray(
        regions.get("plantar_vertex_indices"), dtype=np.int64
    )
    plantar_faces = np.asarray(
        regions.get("plantar_face_indices"), dtype=np.int64
    )
    grid_spacing = float(support_fit.get("support_grid_cell_spacing", np.nan))
    if not np.isfinite(grid_spacing) or grid_spacing <= 0.0:
        raise ValueError("support_fit.json has invalid support_grid_cell_spacing")

    transforms = support_fit.get("transforms")
    if not isinstance(transforms, dict):
        raise ValueError("support_fit.json is missing transforms")
    transform_arrays = {
        name: _finite_array(transforms.get(name), (4, 4), name)
        for name in (
            "shoe_to_normalized",
            "normalized_to_shoe",
            "posed_supr_to_normalized_shoe",
            "normalized_shoe_to_posed_supr",
            "posed_supr_to_original_shoe",
            "original_shoe_to_posed_supr",
        )
    }
    _validate_inverse_pair(
        transform_arrays["shoe_to_normalized"],
        transform_arrays["normalized_to_shoe"],
        "shoe normalization",
    )
    _validate_inverse_pair(
        transform_arrays["posed_supr_to_normalized_shoe"],
        transform_arrays["normalized_shoe_to_posed_supr"],
        "normalized-shoe/posed-SUPR",
    )
    _validate_inverse_pair(
        transform_arrays["posed_supr_to_original_shoe"],
        transform_arrays["original_shoe_to_posed_supr"],
        "original-shoe/posed-SUPR",
    )
    return (
        normalized_shoe,
        normalized_footbed,
        fitted_foot,
        preparation,
        support_fit,
        footbed_faces,
        plantar_vertices,
        plantar_faces,
        grid_spacing,
    )


def _make_overlay(
    shoe: TriangleMesh, foot: TriangleMesh, foot_colors: np.ndarray
) -> TriangleMesh:
    shoe_colors = np.tile(SHOE_COLOR, (len(shoe.vertices), 1))
    return TriangleMesh(
        np.concatenate((shoe.vertices, foot.vertices), axis=0),
        np.concatenate((shoe.faces, foot.faces + len(shoe.vertices)), axis=0),
        np.concatenate((shoe_colors, foot_colors), axis=0),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    preparation_dir = args.preparation_dir.expanduser().resolve(strict=True)
    # Validate the profile before requiring a support-fit directory. This gives
    # high heels the intended explicit rejection instead of a missing-file error.
    _validate_normal_preparation(preparation_dir)
    support_fit_dir = args.support_fit_dir.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    targets = {name: output_dir / name for name in ARTIFACT_NAMES}
    existing = [path for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "cavity-analysis artifacts already exist: "
            f"{formatted}; pass --overwrite to replace them"
        )

    (
        shoe,
        footbed,
        foot,
        preparation,
        support_fit,
        footbed_faces,
        plantar_vertices,
        plantar_faces,
        grid_spacing,
    ) = _load_inputs(preparation_dir, support_fit_dir)
    analysis = analyze_fitted_foot_cavity(
        shoe,
        footbed,
        foot,
        footbed_faces,
        plantar_vertices,
        plantar_faces,
        np.asarray(
            preparation["normalization"]["centerline"]["normalized_xz"],
            dtype=np.float64,
        ),
    )
    foot_colors = analysis.foot_vertex_colors(foot, grid_spacing)
    overlay = _make_overlay(shoe, foot, foot_colors)
    payload: dict[str, Any] = {
        "schema_version": 2,
        "shoe_profile": NORMAL_SHOE_PROFILE,
        "inputs": {
            "preparation_directory": str(preparation_dir),
            "support_fit_directory": str(support_fit_dir),
            "shoe_preparation": str(preparation_dir / "shoe_preparation.json"),
            "support_fit": str(support_fit_dir / "support_fit.json"),
            "normalized_shoe": str(preparation_dir / "shoe_normalized.ply"),
            "normalized_footbed": str(
                support_fit_dir / "footbed_normalized.ply"
            ),
            "fitted_foot": str(support_fit_dir / "foot_support_fitted.ply"),
        },
        "source_schema_versions": {
            "shoe_preparation": preparation["schema_version"],
            "support_fit": support_fit["schema_version"],
        },
        "support_grid_cell_spacing": grid_spacing,
        "transforms": support_fit["transforms"],
        **analysis.to_dict(),
    }
    payload_json = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    output_dir.mkdir(parents=True, exist_ok=True)
    save_triangle_mesh(
        targets["foot_clearance_colored.ply"], foot, foot_colors
    )
    save_triangle_mesh(targets["cavity_overlay.ply"], overlay)
    targets["cavity_analysis.json"].write_text(payload_json, encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    try:
        payload = run(args)
    except (
        FileExistsError,
        FileNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(f"cavity analysis failed: {error}") from error
    collisions = payload["obstacle_collisions"]
    clearance = payload["clearance"]["summaries"]["overall"]
    print(f"wrote cavity-analysis artifacts to {args.output_dir.resolve()}")
    print(
        f"status {payload['status']}; "
        f"intersecting pairs {collisions['intersecting_pair_count']}; "
        f"minimum obstacle clearance {float(clearance['minimum']):.9f}"
    )


if __name__ == "__main__":
    main()
