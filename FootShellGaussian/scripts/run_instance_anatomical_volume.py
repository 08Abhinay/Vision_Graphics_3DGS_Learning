#!/usr/bin/env python3
"""Prepare per-shoe inner-boundary targets for anatomical-volume deformation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import zipfile

import numpy as np

from foot_prior.anatomical_volume import (
    BOUNDARY_KNEE_TRUNCATION,
    CanonicalAnatomicalVolume,
    InstanceBoundaryTarget,
    _ExtendedReference,
    _file_digest,
    _load_extended_reference,
    build_instance_boundary_target,
    load_canonical_anatomical_volume,
    load_fitted_extended_surface,
)
from foot_prior.mesh import TriangleMesh, save_triangle_mesh


ARTIFACT_NAMES = (
    "boundary_target.json",
    "boundary_target.npz",
    "computational_boundary_target.ply",
    "boundary_target_overlay.ply",
)
GRAY = np.asarray((150, 150, 150, 120), dtype=np.uint8)
BLUE = np.asarray((55, 120, 225, 210), dtype=np.uint8)
RED = np.asarray((225, 45, 45, 255), dtype=np.uint8)
PURPLE = np.asarray((175, 95, 205, 230), dtype=np.uint8)


@dataclass(frozen=True)
class _PreparedTarget:
    name: str
    fitted: TriangleMesh
    result: InstanceBoundaryTarget
    payload: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Map the canonical computational inner boundary onto each fitted "
            "foot-and-lower-leg anatomy as a Checkpoint 11-A target."
        )
    )
    parser.add_argument("--anatomical-volume-root", required=True, type=Path)
    parser.add_argument(
        "--extended-anatomical-surface-root", required=True, type=Path
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("shoes", nargs="*")
    return parser.parse_args()


def _write_deterministic_npz(path: Path, **arrays: np.ndarray) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue())


def _shoe_names(root: Path, requested: list[str]) -> list[str]:
    names = requested or [path.name for path in root.iterdir() if path.is_dir()]
    names = [name for name in names if name != "reference"]
    if not names:
        raise ValueError("extended anatomical surface root contains no shoes")
    if len(set(names)) != len(names):
        raise ValueError("shoe names must be unique")
    for name in names:
        if not name or Path(name).name != name or name in {".", "..", "reference"}:
            raise ValueError(f"invalid shoe directory name: {name!r}")
    return sorted(names)


def _target_colored_mesh(result: InstanceBoundaryTarget) -> TriangleMesh:
    vertices = result.vertices[result.faces].reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    face_colors = np.tile(BLUE, (len(result.faces), 1))
    face_colors[result.face_labels == BOUNDARY_KNEE_TRUNCATION] = PURPLE
    face_colors[result.intersecting_face_indices] = RED
    return TriangleMesh(vertices, faces, np.repeat(face_colors, 3, axis=0))


def _overlay(fitted: TriangleMesh, target: TriangleMesh) -> TriangleMesh:
    fitted_colors = np.tile(GRAY, (len(fitted.vertices), 1))
    return TriangleMesh(
        np.vstack((fitted.vertices, target.vertices)),
        np.vstack((fitted.faces, target.faces + len(fitted.vertices))),
        np.vstack((fitted_colors, target.vertex_colors)),
    )


def _prepare(
    name: str,
    volume_root: Path,
    surface_root: Path,
    canonical_volume: CanonicalAnatomicalVolume,
    extended_reference: _ExtendedReference,
) -> _PreparedTarget:
    fitted, metadata = load_fitted_extended_surface(
        surface_root,
        name,
        extended_reference=extended_reference,
    )
    directory = surface_root / name
    json_path = directory / "extended_anatomical_surface.json"
    mesh_path = directory / "foot_lower_leg.ply"
    fitted_digest = str(metadata["geometry"]["geometry_sha256"])
    result = build_instance_boundary_target(
        canonical_volume, extended_reference, fitted
    )
    if result.fitted_surface_digest != fitted_digest:
        raise RuntimeError(f"{name}: target builder changed the fitted-surface digest")

    output_dir = volume_root / name
    payload = result.to_dict()
    payload.update(
        {
            "shoe_name": name,
            "shoe_profile": "normal",
            "inputs": {
                "canonical_volume": str(volume_root / "reference"),
                "canonical_volume_json_sha256": _file_digest(
                    volume_root / "reference" / "canonical_volume.json"
                ),
                "canonical_volume_npz_sha256": _file_digest(
                    volume_root / "reference" / "canonical_volume.npz"
                ),
                "extended_anatomical_surface": str(directory),
                "extended_anatomical_surface_json_sha256": _file_digest(json_path),
                "fitted_surface": str(mesh_path),
                "fitted_surface_file_sha256": _file_digest(mesh_path),
            },
            "artifacts": {
                artifact: str(output_dir / artifact) for artifact in ARTIFACT_NAMES
            },
        }
    )
    return _PreparedTarget(name=name, fitted=fitted, result=result, payload=payload)


def _write_target(directory: Path, prepared: _PreparedTarget) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    try:
        result = prepared.result
        (staging / "boundary_target.json").write_text(
            json.dumps(prepared.payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_deterministic_npz(
            staging / "boundary_target.npz",
            target_inner_vertices=result.vertices,
            intersecting_face_pairs=result.intersecting_face_pairs,
            intersecting_face_indices=result.intersecting_face_indices,
            reverse_reconstructed_vertices=result.reverse_reconstructed_vertices,
            reverse_distances=result.reverse_distances,
            envelope_equation=result.envelope_equation,
        )
        colored = _target_colored_mesh(result)
        save_triangle_mesh(staging / "computational_boundary_target.ply", colored)
        save_triangle_mesh(
            staging / "boundary_target_overlay.ply",
            _overlay(prepared.fitted, colored),
        )
        directory.mkdir(parents=True, exist_ok=True)
        for artifact in ARTIFACT_NAMES:
            os.replace(staging / artifact, directory / artifact)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    volume_root = args.anatomical_volume_root.expanduser().resolve(strict=True)
    surface_root = (
        args.extended_anatomical_surface_root.expanduser().resolve(strict=True)
    )
    if not volume_root.is_dir() or not surface_root.is_dir():
        raise NotADirectoryError("anatomical volume and surface roots must be directories")
    names = _shoe_names(surface_root, list(args.shoes))
    existing = [
        volume_root / name / artifact
        for name in names
        for artifact in ARTIFACT_NAMES
        if (volume_root / name / artifact).exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"instance boundary artifacts already exist; pass --overwrite: {existing[0]}"
        )

    canonical_volume = load_canonical_anatomical_volume(volume_root)
    extended_reference = _load_extended_reference(surface_root)
    prepared = [
        _prepare(
            name,
            volume_root,
            surface_root,
            canonical_volume,
            extended_reference,
        )
        for name in names
    ]
    for target in prepared:
        _write_target(volume_root / target.name, target)
    return {
        "stage": "fitted_anatomical_boundary_target",
        "shoe_count": len(prepared),
        "ready_count": sum(item.result.status == "ready" for item in prepared),
        "requires_untangling_count": sum(
            item.result.status == "ready_requires_untangling" for item in prepared
        ),
        "output_root": str(volume_root),
    }


def main() -> None:
    try:
        result = run(parse_args())
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(f"instance anatomical boundary failed: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
