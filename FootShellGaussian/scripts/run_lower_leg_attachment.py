#!/usr/bin/env python3
"""Attach a neutral SUPR shank and measure its intersections with shoe collars."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np

from foot_prior.anatomy import (
    DENSE_FOOT_FACE_COUNT as DENSE_FACE_COUNT,
    DENSE_FOOT_VERTEX_COUNT as DENSE_VERTEX_COUNT,
    build_extended_canonical_supr_anatomy,
    load_dense_canonical_supr_reference,
)
from foot_prior.lower_leg_fit import build_lower_leg_collar_fit
from foot_prior.mesh import TriangleMesh, load_triangle_mesh, save_triangle_mesh
from foot_prior.supr_foot import build_supr_mesh_subdivision
from foot_prior.supr_lower_leg import load_posable_supr_lower_leg


ARTIFACT_NAMES = (
    "lower_leg_attachment.json",
    "foot_lower_leg.ply",
    "lower_leg_collar_colored.ply",
    "lower_leg_collar_overlay.ply",
)
SHOE_COLOR = np.asarray((150, 150, 150, 255), dtype=np.uint8)
FOOT_COLOR = np.asarray((45, 105, 220, 255), dtype=np.uint8)
LEG_COLOR = np.asarray((55, 175, 205, 255), dtype=np.uint8)
BRIDGE_COLOR = np.asarray((240, 145, 50, 255), dtype=np.uint8)
COLLISION_COLOR = np.asarray((220, 45, 45, 255), dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attach a natural SUPR lower leg to accepted fitted feet, adjust ankle "
            "pitch/roll, and measure exact collar intersections."
        )
    )
    parser.add_argument("--anatomical-surface-root", required=True, type=Path)
    parser.add_argument("--preparation-root", required=True, type=Path)
    parser.add_argument("--support-fit-root", required=True, type=Path)
    parser.add_argument("--full-body-supr-model", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("shoe_names", nargs="*")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected one JSON object: {path}")
    return value


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mesh_with_colors(
    mesh: TriangleMesh,
    lower_leg_vertices: np.ndarray,
    bridge_faces: np.ndarray,
    collision_faces: np.ndarray,
) -> TriangleMesh:
    colors = np.tile(FOOT_COLOR, (len(mesh.vertices), 1))
    colors[lower_leg_vertices] = LEG_COLOR
    bridge_vertices = np.unique(mesh.faces[bridge_faces])
    colors[bridge_vertices] = BRIDGE_COLOR
    if len(collision_faces):
        collision_vertices = np.unique(mesh.faces[collision_faces])
        colors[collision_vertices] = COLLISION_COLOR
    return TriangleMesh(mesh.vertices, mesh.faces, colors)


def _overlay(shoe: TriangleMesh, anatomy: TriangleMesh) -> TriangleMesh:
    shoe_colors = np.tile(SHOE_COLOR, (len(shoe.vertices), 1))
    if anatomy.vertex_colors is None:
        raise ValueError("colored anatomy is required for the overlay")
    return TriangleMesh(
        np.vstack((shoe.vertices, anatomy.vertices)),
        np.vstack((shoe.faces, anatomy.faces + len(shoe.vertices))),
        np.vstack((shoe_colors, anatomy.vertex_colors)),
    )


def _validate_shoe_inputs(
    shoe_name: str,
    anatomical_root: Path,
    preparation_root: Path,
    support_fit_root: Path,
) -> tuple[TriangleMesh, TriangleMesh, TriangleMesh, np.ndarray, np.ndarray, dict[str, Any]]:
    anatomy_dir = anatomical_root / shoe_name
    preparation_dir = preparation_root / shoe_name
    support_dir = support_fit_root / shoe_name
    anatomy_json = anatomy_dir / "anatomical_surface.json"
    preparation_json = preparation_dir / "shoe_preparation.json"
    paths = (
        anatomy_json,
        anatomy_dir / "foot_dense.ply",
        preparation_json,
        preparation_dir / "shoe_normalized.ply",
        support_dir / "footbed_normalized.ply",
    )
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    anatomy_record = _load_json(anatomy_json)
    preparation = _load_json(preparation_json)
    if anatomy_record.get("schema_version") != 1 or anatomy_record.get("shoe_profile") != "normal":
        raise ValueError(f"{shoe_name}: unsupported anatomical-surface record")
    if preparation.get("schema_version") != 1 or preparation.get("shoe_profile") != "normal":
        raise ValueError(f"{shoe_name}: lower-leg attachment accepts normal shoes only")
    dense_foot = load_triangle_mesh(anatomy_dir / "foot_dense.ply")
    shoe = load_triangle_mesh(preparation_dir / "shoe_normalized.ply")
    footbed = load_triangle_mesh(support_dir / "footbed_normalized.ply")
    selection = preparation.get("footbed_selection")
    normalization = preparation.get("normalization")
    if not isinstance(selection, dict) or not isinstance(normalization, dict):
        raise ValueError(f"{shoe_name}: preparation metadata is incomplete")
    centerline = normalization.get("centerline")
    if not isinstance(centerline, dict):
        raise ValueError(f"{shoe_name}: normalized centerline is missing")
    footbed_faces = np.asarray(selection.get("original_face_indices"), dtype=np.int64)
    centerline_xz = np.asarray(centerline.get("normalized_xz"), dtype=np.float64)
    if len(footbed_faces) != len(footbed.faces):
        raise ValueError(f"{shoe_name}: footbed face count disagrees with preparation")
    return shoe, footbed, dense_foot, footbed_faces, centerline_xz, anatomy_record


def _write_one(
    output_dir: Path,
    payload: dict[str, Any],
    attachment_mesh: TriangleMesh,
    colored: TriangleMesh,
    overlay: TriangleMesh,
    overwrite: bool,
) -> None:
    existing = [output_dir / name for name in ARTIFACT_NAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"lower-leg artifacts already exist in {output_dir}; pass --overwrite"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}-", dir=output_dir.parent))
    try:
        (staging / ARTIFACT_NAMES[0]).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        save_triangle_mesh(staging / ARTIFACT_NAMES[1], attachment_mesh)
        save_triangle_mesh(staging / ARTIFACT_NAMES[2], colored)
        save_triangle_mesh(staging / ARTIFACT_NAMES[3], overlay)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name in ARTIFACT_NAMES:
            os.replace(staging / name, output_dir / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def run(args: argparse.Namespace) -> list[dict[str, Any]]:
    anatomical_root = args.anatomical_surface_root.expanduser().resolve(strict=True)
    preparation_root = args.preparation_root.expanduser().resolve(strict=True)
    support_fit_root = args.support_fit_root.expanduser().resolve(strict=True)
    full_body_model = args.full_body_supr_model.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    reference = load_dense_canonical_supr_reference(anatomical_root)
    canonical = build_extended_canonical_supr_anatomy(reference, full_body_model)
    subdivision = build_supr_mesh_subdivision(
        canonical.lower_leg.mesh.faces,
        len(canonical.lower_leg.mesh.vertices),
        2,
    )
    lower_leg_model = load_posable_supr_lower_leg(
        full_body_model,
        canonical.lower_leg,
        subdivision,
        num_betas=10,
    )
    correspondence = canonical.ankle_correspondence.copy()
    correspondence[:, 1] -= DENSE_VERTEX_COUNT

    names = list(args.shoe_names)
    if not names:
        names = sorted(
            path.name
            for path in anatomical_root.iterdir()
            if path.is_dir() and path.name != "reference"
        )
    if not names:
        raise ValueError("no fitted anatomical-surface directories were found")

    results: list[dict[str, Any]] = []
    for shoe_name in names:
        shoe, footbed, dense_foot, footbed_faces, centerline, anatomy_record = (
            _validate_shoe_inputs(
                shoe_name, anatomical_root, preparation_root, support_fit_root
            )
        )
        if dense_foot.vertices.shape != (DENSE_VERTEX_COUNT, 3) or not np.array_equal(
            dense_foot.faces, reference.faces
        ):
            raise ValueError(f"{shoe_name}: fitted dense topology is not canonical")
        fit = build_lower_leg_collar_fit(
            shoe,
            footbed,
            footbed_faces,
            dense_foot,
            centerline,
            reference.vertices,
            reference.ankle_loop,
            correspondence,
            lower_leg_model,
        )
        attachment = fit.selected.attachment
        colliding_combined_faces = fit.selected.query_face_indices[
            fit.selected.colliding_query_face_indices
        ]
        colored = _mesh_with_colors(
            attachment.mesh,
            attachment.lower_leg_vertex_indices,
            attachment.bridge_face_indices,
            colliding_combined_faces,
        )
        overlay = _overlay(shoe, colored)
        output_dir = output_root / shoe_name
        fit_record = fit.to_dict()
        payload: dict[str, Any] = {
            "schema_version": 2,
            "stage": "fitted_foot_natural_lower_leg_collar_fit",
            "shoe_name": shoe_name,
            "shoe_profile": "normal",
            "status": fit.status,
            "fit": fit_record,
            "inputs": {
                "anatomical_surface": str(anatomical_root / shoe_name),
                "shoe_preparation": str(preparation_root / shoe_name),
                "support_fit": str(support_fit_root / shoe_name),
                "full_body_supr_model": str(full_body_model),
                "fitted_dense_foot_sha256": _file_digest(
                    anatomical_root / shoe_name / "foot_dense.ply"
                ),
                "source_anatomical_schema": anatomy_record["schema_version"],
            },
        }
        _write_one(
            output_dir,
            payload,
            attachment.mesh,
            colored,
            overlay,
            args.overwrite,
        )
        results.append(
            {
                "shoe_name": shoe_name,
                "status": payload["status"],
                "collision_pairs": int(len(fit.selected.collision_pairs)),
                "output_dir": str(output_dir),
            }
        )
    return results


def main() -> None:
    try:
        results = run(parse_args())
    except (FileExistsError, FileNotFoundError, NotADirectoryError, RuntimeError, TypeError, ValueError) as error:
        raise SystemExit(f"lower-leg attachment failed: {error}") from error
    for result in results:
        print(
            f"{result['shoe_name']}: {result['status']}; "
            f"{result['collision_pairs']} exact pairs -> {result['output_dir']}"
        )


if __name__ == "__main__":
    main()
