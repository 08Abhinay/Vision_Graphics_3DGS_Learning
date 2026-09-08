#!/usr/bin/env python3
"""Fit a Checkpoint 5 SUPR foot against normal-shoe cavity obstacles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from foot_prior.alignment import transform_points
from foot_prior.alignment import identify_supr_contact_regions
from foot_prior.containment import build_containment_foot_fit
from foot_prior.mesh import TriangleMesh, save_triangle_mesh
from foot_prior.normalization import NORMAL_SHOE_PROFILE
from foot_prior.supr_foot import (
    build_supr_mesh_subdivision,
    load_neutral_supr_foot,
    load_posable_supr_foot,
)
from run_cavity_analysis import (
    _load_inputs,
    _load_json,
    _make_overlay,
    _validate_normal_preparation,
)


ARTIFACT_NAMES = (
    "containment_fit.json",
    "foot_containment_fitted.ply",
    "foot_clearance_colored.ply",
    "containment_fit_overlay.ply",
)
FOOT_COLOR = np.asarray([45, 105, 220, 255], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Adjust normal-shoe SUPR placement and shape to reduce forbidden "
            "surface collisions while preserving plantar support."
        )
    )
    parser.add_argument("--preparation-dir", required=True, type=Path)
    parser.add_argument("--support-fit-dir", required=True, type=Path)
    parser.add_argument("--cavity-analysis-dir", required=True, type=Path)
    parser.add_argument("--supr-model", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--subdivision-levels",
        type=int,
        choices=(0, 1, 2),
        default=0,
        help=(
            "Deterministically subdivide every SUPR candidate before support "
            "and containment scoring; 0 preserves the native topology."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the four known containment-fit artifacts.",
    )
    return parser.parse_args()


def _finite_vector(value: Any, length: int, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{label} must contain {length} finite values")
    return array


def _finite_matrix(value: Any, label: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    return matrix


def _validate_cavity_record(
    cavity_dir: Path,
    preparation_dir: Path,
    support_fit_dir: Path,
) -> dict[str, Any]:
    path = cavity_dir / "cavity_analysis.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = _load_json(path)
    if payload.get("schema_version") != 2:
        raise ValueError(
            "cavity_analysis.json must use schema_version 2; regenerate "
            "Checkpoint 6 with the current cavity runner"
        )
    if payload.get("shoe_profile") != NORMAL_SHOE_PROFILE:
        raise ValueError("cavity_analysis.json must describe shoe_profile='normal'")
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError("cavity_analysis.json is missing inputs")
    expected = {
        "preparation_directory": preparation_dir,
        "support_fit_directory": support_fit_dir,
    }
    for name, expected_path in expected.items():
        value = inputs.get(name)
        if not isinstance(value, str):
            raise ValueError(f"cavity_analysis.json is missing {name}")
        actual = Path(value).expanduser().resolve(strict=True)
        if actual != expected_path:
            raise ValueError(
                f"cavity analysis {name} does not match the supplied input"
            )
    return payload


def _reproduce_support_fit(
    supr_model: Any,
    support_fit: dict[str, Any],
    fitted_foot: TriangleMesh,
) -> tuple[np.ndarray, np.ndarray]:
    supr = support_fit.get("supr")
    transforms = support_fit.get("transforms")
    if not isinstance(supr, dict) or not isinstance(transforms, dict):
        raise ValueError("support_fit.json is missing SUPR or transform data")
    pose = _finite_vector(
        supr.get("pose_parameters_radians"),
        supr_model.num_pose_parameters,
        "support-fit pose",
    )
    betas = _finite_vector(
        supr.get("betas"), supr_model.num_betas, "support-fit betas"
    )
    posed, _ = supr_model.evaluate(pose, betas)
    transform = _finite_matrix(
        transforms.get("posed_supr_to_normalized_shoe"),
        "posed SUPR transform",
    )
    reproduced = transform_points(posed, transform)
    difference = reproduced - fitted_foot.vertices
    rigid_drift = difference.mean(axis=0)
    nonrigid_error = difference - rigid_drift
    # The reviewed support fits were produced on several GPU models. CUDA
    # evaluation can introduce a few float32 units of rigid drift while the
    # reproduced shape remains identical.
    if (
        float(np.max(np.abs(rigid_drift))) > 1e-5
        or float(np.max(np.abs(nonrigid_error))) > 1e-6
    ):
        raise ValueError(
            "stored Checkpoint 5 mesh cannot be reproduced from its SUPR "
            "parameters and transform"
        )
    return pose, betas


def run(args: argparse.Namespace) -> dict[str, Any]:
    preparation_dir = args.preparation_dir.expanduser().resolve(strict=True)
    _validate_normal_preparation(preparation_dir)
    support_fit_dir = args.support_fit_dir.expanduser().resolve(strict=True)
    cavity_dir = args.cavity_analysis_dir.expanduser().resolve(strict=True)
    supr_path = args.supr_model.expanduser().resolve(strict=True)
    output_dir = args.output_dir.expanduser().resolve()
    targets = {name: output_dir / name for name in ARTIFACT_NAMES}
    existing = [path for path in targets.values() if path.exists()]
    if existing and not args.overwrite:
        formatted = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            "containment-fit artifacts already exist: "
            f"{formatted}; pass --overwrite to replace them"
        )

    (
        shoe,
        footbed,
        fitted_foot,
        preparation,
        support_fit,
        footbed_faces,
        plantar_vertices,
        plantar_faces,
        grid_spacing,
    ) = _load_inputs(preparation_dir, support_fit_dir)
    cavity_record = _validate_cavity_record(
        cavity_dir, preparation_dir, support_fit_dir
    )
    neutral_foot = load_neutral_supr_foot(supr_path)
    supr_model = load_posable_supr_foot(supr_path, num_betas=10)
    pose, betas = _reproduce_support_fit(
        supr_model, support_fit, fitted_foot
    )
    subdivision = build_supr_mesh_subdivision(
        supr_model.faces,
        len(neutral_foot.vertices),
        args.subdivision_levels,
    )

    normalization = preparation.get("normalization")
    if not isinstance(normalization, dict):
        raise ValueError("shoe_preparation.json is missing normalization")
    centerline = normalization.get("centerline")
    if not isinstance(centerline, dict):
        raise ValueError("shoe preparation is missing its centerline")
    normalized_centerline = np.asarray(
        centerline.get("normalized_xz"), dtype=np.float64
    )
    shoe_to_normalized = _finite_matrix(
        normalization.get("shoe_to_normalized"), "shoe_to_normalized"
    )
    normalized_to_shoe = _finite_matrix(
        normalization.get("normalized_to_shoe"), "normalized_to_shoe"
    )
    stored_pairs = np.asarray(
        cavity_record.get("obstacle_collisions", {}).get(
            "intersecting_face_pairs", []
        ),
        dtype=np.int64,
    ).reshape(-1, 2)

    result = build_containment_foot_fit(
        supr_model=supr_model,
        neutral_foot_mesh=neutral_foot,
        normalized_shoe_mesh=shoe,
        normalized_support_mesh=footbed,
        normalized_centerline_xz=normalized_centerline,
        footbed_source_face_indices=footbed_faces,
        shoe_to_normalized=shoe_to_normalized,
        normalized_to_shoe=normalized_to_shoe,
        support_grid_cell_spacing=grid_spacing,
        initial_pose_parameters=pose,
        initial_betas=betas,
        baseline_fitted_foot=fitted_foot,
        expected_baseline_collision_pairs=stored_pairs,
        expected_baseline_status=cavity_record.get("status"),
        cavity_subdivision=(
            subdivision if subdivision.levels > 0 else None
        ),
    )

    final_foot = TriangleMesh(result.aligned_vertices, result.foot_faces)
    colors = result.final_cavity.foot_vertex_colors(
        final_foot, grid_spacing
    )
    overlay = _make_overlay(shoe, final_foot, colors)
    plain_colors = np.tile(FOOT_COLOR, (len(final_foot.vertices), 1))
    payload: dict[str, Any] = {
        "schema_version": 5,
        "shoe_profile": NORMAL_SHOE_PROFILE,
        "inputs": {
            "preparation_directory": str(preparation_dir),
            "support_fit_directory": str(support_fit_dir),
            "cavity_analysis_directory": str(cavity_dir),
            "supr_model": str(supr_path),
            "normalized_shoe": str(preparation_dir / "shoe_normalized.ply"),
            "normalized_footbed": str(
                support_fit_dir / "footbed_normalized.ply"
            ),
        },
        "source_schema_versions": {
            "shoe_preparation": preparation["schema_version"],
            "support_fit": support_fit["schema_version"],
            "cavity_analysis": cavity_record["schema_version"],
        },
        "contact_regions": identify_supr_contact_regions(
            subdivision.apply_mesh(neutral_foot)
        ).to_dict(),
        "supr_topology": subdivision.to_dict(),
        "support_grid_cell_spacing": grid_spacing,
        **result.to_dict(),
    }
    payload_json = json.dumps(payload, indent=2, sort_keys=True) + "\n"

    output_dir.mkdir(parents=True, exist_ok=True)
    save_triangle_mesh(
        targets["foot_containment_fitted.ply"], final_foot, plain_colors
    )
    save_triangle_mesh(
        targets["foot_clearance_colored.ply"], final_foot, colors
    )
    save_triangle_mesh(targets["containment_fit_overlay.ply"], overlay)
    targets["containment_fit.json"].write_text(
        payload_json, encoding="utf-8"
    )
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
        raise SystemExit(f"containment fit failed: {error}") from error
    scores = payload["collision_score"]
    sizing = payload["sizing"]
    betas = payload["supr"]["betas"]
    print(f"wrote containment-fit artifacts to {args.output_dir.resolve()}")
    print(
        f"status {payload['status']}; foot ratio "
        f"{sizing['foot_length_ratio']:.6f} "
        f"({sizing['toe_allowance_mm']:.1f} mm toe allowance); collision area "
        f"{scores['baseline']['collision_area_fraction']:.6f} -> "
        f"{scores['final']['collision_area_fraction']:.6f}"
    )
    print(
        "betas "
        + " ".join(f"{float(value):+.2f}" for value in betas)
        + f"; nonzero {sum(abs(float(value)) > 1e-9 for value in betas)}/{len(betas)}"
    )


if __name__ == "__main__":
    main()
