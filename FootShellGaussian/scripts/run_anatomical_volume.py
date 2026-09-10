#!/usr/bin/env python3
"""Build the shared canonical foot-and-lower-leg anatomical volume."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

import numpy as np

from foot_prior.anatomical_volume import (
    BOUNDARY_LABEL_NAMES,
    BOUNDARY_OUTER_ENVELOPE,
    CanonicalAnatomicalVolume,
    build_canonical_anatomical_volume,
)
from foot_prior.mesh import TriangleMesh, save_triangle_mesh


ARTIFACT_NAMES = (
    "canonical_volume.json",
    "canonical_volume.npz",
    "canonical_volume.vtk",
    "inner_anatomical_boundary.ply",
    "outer_envelope.ply",
    "boundary_regions.ply",
)
BOUNDARY_COLORS = np.asarray(
    (
        (0, 0, 0, 255),
        (45, 185, 95, 255),
        (240, 145, 50, 255),
        (65, 155, 225, 255),
        (175, 95, 205, 255),
        (80, 125, 225, 110),
    ),
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Construct the fixed tetrahedral domain around the canonical SUPR "
            "foot and neutral male lower leg, then solve its harmonic outward "
            "coordinate."
        )
    )
    parser.add_argument("--anatomical-surface-root", required=True, type=Path)
    parser.add_argument("--full-body-supr-model", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the six known canonical-volume artifacts.",
    )
    return parser.parse_args()


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _boundary_regions_mesh(result: CanonicalAnatomicalVolume) -> TriangleMesh:
    vertices = result.volume_vertices[result.boundary_faces].reshape(-1, 3)
    faces = np.arange(len(vertices), dtype=np.int64).reshape(-1, 3)
    colors = np.repeat(BOUNDARY_COLORS[result.boundary_labels], 3, axis=0)
    return TriangleMesh(vertices, faces, colors)


def _write_vtk(path: Path, result: CanonicalAnatomicalVolume) -> None:
    gradient_magnitude = np.linalg.norm(result.harmonic_r_gradient, axis=1)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# vtk DataFile Version 3.0\n")
        stream.write("Canonical foot-and-lower-leg anatomical volume\n")
        stream.write("ASCII\n")
        stream.write("DATASET UNSTRUCTURED_GRID\n")
        stream.write(f"POINTS {len(result.volume_vertices)} double\n")
        np.savetxt(stream, result.volume_vertices, fmt="%.17g")
        stream.write(
            f"CELLS {len(result.tetrahedra)} {5 * len(result.tetrahedra)}\n"
        )
        cells = np.column_stack(
            (np.full(len(result.tetrahedra), 4, dtype=np.int64), result.tetrahedra)
        )
        np.savetxt(stream, cells, fmt="%d")
        stream.write(f"CELL_TYPES {len(result.tetrahedra)}\n")
        np.savetxt(
            stream, np.full(len(result.tetrahedra), 10, dtype=np.int64), fmt="%d"
        )
        stream.write(f"POINT_DATA {len(result.volume_vertices)}\n")
        stream.write("SCALARS harmonic_r double 1\nLOOKUP_TABLE default\n")
        np.savetxt(stream, result.harmonic_r, fmt="%.17g")
        stream.write(f"CELL_DATA {len(result.tetrahedra)}\n")
        stream.write("SCALARS harmonic_r_gradient_magnitude double 1\n")
        stream.write("LOOKUP_TABLE default\n")
        np.savetxt(stream, gradient_magnitude, fmt="%.17g")
        stream.write("SCALARS tetrahedron_mean_ratio double 1\n")
        stream.write("LOOKUP_TABLE default\n")
        np.savetxt(stream, result.tetrahedron_mean_ratio_quality, fmt="%.17g")


def _write_artifacts(
    directory: Path,
    result: CanonicalAnatomicalVolume,
    payload: dict[str, object],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "canonical_volume.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_deterministic_npz(
        directory / "canonical_volume.npz",
        volume_vertices=result.volume_vertices,
        tetrahedra=result.tetrahedra,
        boundary_faces=result.boundary_faces,
        boundary_labels=result.boundary_labels,
        boundary_label_names=np.asarray(BOUNDARY_LABEL_NAMES),
        harmonic_r=result.harmonic_r,
        harmonic_r_gradient=result.harmonic_r_gradient,
        dense_foot_to_volume_indices=result.dense_foot_to_volume_indices,
        lower_leg_to_volume_indices=result.lower_leg_to_volume_indices,
        foot_boundary_face_indices=result.foot_boundary_face_indices,
        ankle_transition_face_indices=result.ankle_transition_face_indices,
        lower_leg_boundary_face_indices=result.lower_leg_boundary_face_indices,
        knee_cap_indices=result.knee_cap_indices,
        knee_cap_vertex_index=np.asarray(result.knee_cap_vertex_index),
        dense_knee_loop_indices=result.dense_knee_loop_indices,
        ankle_loop_correspondence=result.ankle_loop_correspondence,
        lower_leg_source_vertex_indices=result.lower_leg_source_vertex_indices,
        lower_leg_source_face_indices=result.lower_leg_source_face_indices,
        body_to_reference=result.body_to_reference,
        outer_vertex_indices=result.outer_vertex_indices,
        tetrahedron_signed_volumes=result.tetrahedron_signed_volumes,
        tetrahedron_mean_ratio_quality=result.tetrahedron_mean_ratio_quality,
        volume_topology_sha256=np.asarray(result.topology_digest),
        extended_surface_sha256=np.asarray(result.extended_surface_digest),
        outer_envelope_topology_sha256=np.asarray(
            result.envelope_topology_digest
        ),
    )
    _write_vtk(directory / "canonical_volume.vtk", result)
    save_triangle_mesh(
        directory / "inner_anatomical_boundary.ply",
        result.inner_anatomical_mesh,
    )
    first_outer = int(result.outer_vertex_indices[0])
    outer_faces = result.boundary_faces[
        result.boundary_labels == BOUNDARY_OUTER_ENVELOPE
    ]
    outer_mesh = TriangleMesh(
        result.volume_vertices[result.outer_vertex_indices],
        outer_faces - first_outer,
    )
    save_triangle_mesh(directory / "outer_envelope.ply", outer_mesh)
    save_triangle_mesh(
        directory / "boundary_regions.ply", _boundary_regions_mesh(result)
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    surface_root = args.anatomical_surface_root.expanduser().resolve(strict=True)
    if not surface_root.is_dir():
        raise NotADirectoryError(surface_root)
    full_body_model = args.full_body_supr_model.expanduser().resolve(strict=True)
    if not full_body_model.is_file():
        raise FileNotFoundError(full_body_model)
    output_root = args.output_root.expanduser().resolve()
    destination = output_root / "reference"
    existing = [destination / name for name in ARTIFACT_NAMES]
    existing = [path for path in existing if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "canonical-volume artifacts already exist; pass --overwrite to replace them"
        )

    result = build_canonical_anatomical_volume(surface_root, full_body_model)
    reference_root = surface_root / "reference"
    payload = result.to_dict()
    payload["input"] = {
        "anatomical_surface_root": str(surface_root),
        "canonical_surface_json": str(reference_root / "canonical_surface.json"),
        "canonical_surface_npz": str(reference_root / "canonical_surface.npz"),
        "neutral_dense_ply": str(reference_root / "neutral_dense.ply"),
        "canonical_surface_json_sha256": _file_digest(
            reference_root / "canonical_surface.json"
        ),
        "canonical_surface_npz_sha256": _file_digest(
            reference_root / "canonical_surface.npz"
        ),
        "neutral_dense_ply_sha256": _file_digest(
            reference_root / "neutral_dense.ply"
        ),
        "full_body_supr_model": str(full_body_model),
        "full_body_supr_model_sha256": _file_digest(full_body_model),
    }

    output_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(prefix=".anatomical-volume-", dir=output_root.parent)
    )
    try:
        staging = staging_root / "reference"
        _write_artifacts(staging, result, payload)
        destination.mkdir(parents=True, exist_ok=True)
        for name in ARTIFACT_NAMES:
            os.replace(staging / name, destination / name)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    return {
        "output_root": str(output_root),
        "vertex_count": len(result.volume_vertices),
        "tetrahedron_count": len(result.tetrahedra),
        "topology_digest": result.topology_digest,
        "minimum_quality": float(
            np.min(result.tetrahedron_mean_ratio_quality)
        ),
        "harmonic_residual": result.diagnostics["harmonic_r"][
            "linear_system_relative_residual"
        ],
    }


def main() -> None:
    args = parse_args()
    try:
        result = run(args)
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(f"anatomical volume failed: {error}") from error
    print(
        f"wrote canonical anatomical volume to {result['output_root']}: "
        f"{result['vertex_count']} vertices, "
        f"{result['tetrahedron_count']} tetrahedra"
    )
    print(
        f"topology sha256 {result['topology_digest']}; "
        f"minimum quality {result['minimum_quality']:.6g}; "
        f"harmonic residual {result['harmonic_residual']:.3e}"
    )


if __name__ == "__main__":
    main()
