#!/usr/bin/env python3
"""Build canonical and shared dense SUPR anatomical surfaces."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from foot_prior.anatomy import (
    JOINT_NAMES,
    LONGITUDINAL_COLORS,
    LONGITUDINAL_REGION_NAMES,
    SURFACE_COLORS,
    SURFACE_REGION_NAMES,
    CanonicalSuprAnatomy,
    DenseCanonicalSuprAnatomy,
    build_canonical_supr_anatomy,
    build_dense_canonical_supr_anatomy,
    map_surface_coordinates,
)
from foot_prior.mesh import TriangleMesh, load_triangle_mesh, save_triangle_mesh
from foot_prior.normalization import NORMAL_SHOE_PROFILE
from foot_prior.supr_foot import build_supr_mesh_subdivision


REFERENCE_ARTIFACTS = (
    "canonical_surface.json",
    "canonical_surface.npz",
    "neutral_dense.ply",
    "regions_longitudinal.ply",
    "regions_surface.ply",
)
SHOE_ARTIFACTS = (
    "anatomical_surface.json",
    "foot_dense.ply",
    "regions_longitudinal.ply",
    "regions_surface.ply",
)
SUBDIVISION_LEVELS = 2


@dataclass(frozen=True)
class _PreparedShoe:
    name: str
    source_directory: Path
    source_mesh_path: Path
    source_json_path: Path
    source_mesh: TriangleMesh
    dense_mesh: TriangleMesh
    payload: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Transfer canonical SUPR anatomy to accepted native containment "
            "fits and create their shared deterministic subdiv2 surfaces."
        )
    )
    parser.add_argument("--containment-root", required=True, type=Path)
    parser.add_argument("--supr-model", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only the known anatomical-surface artifacts.",
    )
    parser.add_argument(
        "shoes",
        nargs="*",
        help="Optional containment-fit directory names; default is every directory.",
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


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def _write_deterministic_npz(path: Path, **arrays: np.ndarray) -> None:
    """Write an NPZ without wall-clock timestamps in its ZIP members."""

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue())


def _validate_inverse(forward: Any, inverse: Any, label: str) -> None:
    first = np.asarray(forward, dtype=np.float64)
    second = np.asarray(inverse, dtype=np.float64)
    if (
        first.shape != (4, 4)
        or second.shape != (4, 4)
        or not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or not np.allclose(first @ second, np.eye(4), atol=1e-9, rtol=0.0)
        or not np.allclose(second @ first, np.eye(4), atol=1e-9, rtol=0.0)
    ):
        raise ValueError(f"{label} transforms must be finite mutual inverses")


def _shoe_names(root: Path, requested: list[str]) -> list[str]:
    if requested:
        names = requested
    else:
        names = [path.name for path in root.iterdir() if path.is_dir()]
    if not names:
        raise ValueError("containment root contains no shoe directories")
    if len(set(names)) != len(names):
        raise ValueError("shoe names must be unique")
    for name in names:
        if not name or Path(name).name != name or name in {".", ".."}:
            raise ValueError(f"invalid shoe directory name: {name!r}")
    return sorted(names)


def _existing_targets(output_root: Path, names: list[str]) -> list[Path]:
    paths = [output_root / "reference" / name for name in REFERENCE_ARTIFACTS]
    for shoe in names:
        paths.extend(output_root / shoe / name for name in SHOE_ARTIFACTS)
    return [path for path in paths if path.exists()]


def _prepare_shoe(
    name: str,
    containment_root: Path,
    output_root: Path,
    anatomy: CanonicalSuprAnatomy,
    dense_reference: DenseCanonicalSuprAnatomy,
) -> _PreparedShoe:
    directory = (containment_root / name).resolve(strict=True)
    if not directory.is_dir() or directory.parent != containment_root:
        raise ValueError(f"shoe is not directly inside containment root: {name}")
    metadata_path = directory / "containment_fit.json"
    mesh_path = directory / "foot_containment_fitted.ply"
    for path in (metadata_path, mesh_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    metadata = _load_json(metadata_path)
    if metadata.get("schema_version") != 5:
        raise ValueError(f"{name}: containment_fit.json must use schema_version 5")
    if metadata.get("shoe_profile") != NORMAL_SHOE_PROFILE:
        raise ValueError(f"{name}: anatomical surfaces accept normal shoes only")
    fitted = load_triangle_mesh(mesh_path)
    if (
        len(fitted.vertices) != len(anatomy.raw_mesh.vertices)
        or not np.array_equal(fitted.faces, anatomy.raw_mesh.faces)
    ):
        raise ValueError(
            f"{name}: fitted foot must use the native 266-vertex SUPR topology"
        )
    recorded_bounds = np.asarray(
        metadata.get("bounds", {}).get("aligned_foot"), dtype=np.float64
    )
    if (
        recorded_bounds.shape != (2, 3)
        or not np.isfinite(recorded_bounds).all()
        or not np.allclose(fitted.bounds, recorded_bounds, atol=1e-6, rtol=1e-6)
    ):
        raise ValueError(f"{name}: fitted mesh disagrees with recorded bounds")
    supr = metadata.get("supr")
    if not isinstance(supr, dict):
        raise ValueError(f"{name}: containment metadata is missing SUPR parameters")
    pose = np.asarray(supr.get("pose_parameters_radians"), dtype=np.float64)
    betas = np.asarray(supr.get("betas"), dtype=np.float64)
    if pose.shape != (39,) or betas.shape != (10,) or not (
        np.isfinite(pose).all() and np.isfinite(betas).all()
    ):
        raise ValueError(f"{name}: containment SUPR parameters are invalid")
    transforms = metadata.get("transforms")
    if not isinstance(transforms, dict):
        raise ValueError(f"{name}: containment transforms are missing")
    _validate_inverse(
        transforms.get("posed_supr_to_normalized_shoe"),
        transforms.get("normalized_shoe_to_posed_supr"),
        f"{name} posed-SUPR/normalized-shoe",
    )
    _validate_inverse(
        transforms.get("posed_supr_to_original_shoe"),
        transforms.get("original_shoe_to_posed_supr"),
        f"{name} posed-SUPR/original-shoe",
    )

    subdivision = dense_reference.subdivision
    dense_vertices = subdivision.apply_vertices(fitted.vertices)
    mapped_vertices = map_surface_coordinates(
        dense_reference.vertex_chart_face_indices,
        dense_reference.vertex_chart_barycentric,
        fitted.vertices,
        fitted.faces,
    )
    if not np.allclose(dense_vertices, mapped_vertices, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"{name}: dense surface and canonical chart disagree")
    if not np.array_equal(dense_vertices[: len(fitted.vertices)], fitted.vertices):
        raise RuntimeError(f"{name}: subdivision changed original SUPR vertices")
    dense_mesh = TriangleMesh(dense_vertices, subdivision.faces)
    if not np.allclose(dense_mesh.bounds, fitted.bounds, atol=1e-12, rtol=0.0):
        raise RuntimeError(f"{name}: subdivision changed fitted-foot bounds")

    output_directory = output_root / name
    payload = {
        "schema_version": 1,
        "stage": "canonical_dense_supr_anatomical_surface",
        "shoe_profile": NORMAL_SHOE_PROFILE,
        "inputs": {
            "containment_directory": str(directory),
            "containment_fit_json": str(metadata_path),
            "containment_fit_json_sha256": _file_digest(metadata_path),
            "fitted_native_foot": str(mesh_path),
            "fitted_native_foot_sha256": _file_digest(mesh_path),
            "canonical_reference": str(
                output_root / "reference" / "canonical_surface.npz"
            ),
        },
        "source_schema_version": metadata["schema_version"],
        "surface_coordinate": {
            "definition": "canonical SUPR source face plus barycentric weights",
            "chart_face_topology_sha256": _array_digest(anatomy.raw_mesh.faces),
            "mapping": (
                "apply the stored coordinate to the same native SUPR face in "
                "this fitted foot; subdivision uses the shared reference map"
            ),
        },
        "topology": subdivision.to_dict(),
        "geometry": {
            "native_bounds": fitted.bounds.tolist(),
            "dense_bounds": dense_mesh.bounds.tolist(),
            "native_geometry_sha256": _array_digest(fitted.vertices, fitted.faces),
            "dense_geometry_sha256": _array_digest(
                dense_mesh.vertices, dense_mesh.faces
            ),
            "original_vertices_preserved": True,
            "surface_geometry_unchanged": True,
        },
        "anatomical_labels": {
            "source": "canonical neutral right SUPR reference",
            "longitudinal_names": list(LONGITUDINAL_REGION_NAMES),
            "surface_names": list(SURFACE_REGION_NAMES),
            "dense_vertex_labels_shared_by_index": True,
        },
        "supr": {
            "pose_parameters_radians": pose.tolist(),
            "betas": betas.tolist(),
        },
        "artifacts": {
            name: str(output_directory / name) for name in SHOE_ARTIFACTS
        },
    }
    return _PreparedShoe(
        name=name,
        source_directory=directory,
        source_mesh_path=mesh_path,
        source_json_path=metadata_path,
        source_mesh=fitted,
        dense_mesh=dense_mesh,
        payload=payload,
    )


def _reference_payload(
    model_path: Path,
    anatomy: CanonicalSuprAnatomy,
    dense: DenseCanonicalSuprAnatomy,
) -> dict[str, Any]:
    subdivision = dense.subdivision
    return {
        "schema_version": 1,
        "stage": "canonical_dense_supr_anatomical_reference",
        "reference": "F_0 neutral right SUPR foot",
        "input": {
            "supr_model": str(model_path),
            "supr_model_sha256": _file_digest(model_path),
        },
        "surface_coordinate": {
            "type": "triangle_chart_with_barycentric_coordinates",
            "chart_id": "native neutral SUPR face index",
            "local_coordinates": "(u, v, 1-u-v)",
            "boundary_role": "r=0 boundary for the future anatomical volume",
            "volumetric_r_status": "not_implemented",
        },
        **anatomy.to_dict(),
        "subdivision": subdivision.to_dict(),
        "digests": {
            "native_topology_sha256": _array_digest(anatomy.raw_mesh.faces),
            "dense_topology_sha256": subdivision.topology_digest,
            "provenance_sha256": _array_digest(
                subdivision.vertex_source_indices,
                subdivision.vertex_source_weights,
                subdivision.face_parent_indices,
                dense.vertex_chart_face_indices,
                dense.vertex_chart_barycentric,
            ),
            "canonical_dense_surface_sha256": _array_digest(
                dense.mesh.vertices, dense.mesh.faces
            ),
        },
    }


def _write_reference(
    directory: Path,
    payload: dict[str, Any],
    anatomy: CanonicalSuprAnatomy,
    dense: DenseCanonicalSuprAnatomy,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    subdivision = dense.subdivision
    landmarks = anatomy.landmarks
    landmark_names = tuple(landmarks)
    _write_deterministic_npz(
        directory / "canonical_surface.npz",
        raw_vertices=anatomy.raw_mesh.vertices,
        native_faces=anatomy.raw_mesh.faces,
        reference_vertices=anatomy.reference_mesh.vertices,
        dense_reference_vertices=dense.mesh.vertices,
        dense_faces=dense.mesh.faces,
        dense_face_parent_indices=subdivision.face_parent_indices,
        dense_vertex_source_indices=subdivision.vertex_source_indices,
        dense_vertex_source_weights=subdivision.vertex_source_weights,
        dense_vertex_chart_face_indices=dense.vertex_chart_face_indices,
        dense_vertex_chart_barycentric=dense.vertex_chart_barycentric,
        raw_supr_to_reference=anatomy.raw_to_reference,
        reference_to_raw_supr=anatomy.reference_to_raw,
        joint_names=np.asarray(JOINT_NAMES),
        raw_joints=anatomy.raw_joints,
        reference_joints=anatomy.reference_joints,
        ankle_boundary_vertex_indices=anatomy.ankle_boundary_vertex_indices,
        dense_ankle_boundary_vertex_indices=dense.ankle_boundary_vertex_indices,
        landmark_names=np.asarray(landmark_names),
        landmark_primary_vertex_indices=np.asarray(
            [landmarks[name]["primary_vertex_index"] for name in landmark_names],
            dtype=np.int64,
        ),
        landmark_reference_positions=np.asarray(
            [landmarks[name]["point_reference"] for name in landmark_names],
            dtype=np.float64,
        ),
        native_longitudinal_vertex_labels=anatomy.longitudinal_vertex_labels,
        native_longitudinal_face_labels=anatomy.longitudinal_face_labels,
        native_surface_vertex_labels=anatomy.surface_vertex_labels,
        native_surface_face_labels=anatomy.surface_face_labels,
        dense_longitudinal_vertex_labels=dense.longitudinal_vertex_labels,
        dense_longitudinal_face_labels=dense.longitudinal_face_labels,
        dense_surface_vertex_labels=dense.surface_vertex_labels,
        dense_surface_face_labels=dense.surface_face_labels,
    )
    save_triangle_mesh(directory / "neutral_dense.ply", dense.mesh)
    save_triangle_mesh(
        directory / "regions_longitudinal.ply",
        dense.mesh,
        dense.longitudinal_colors,
    )
    save_triangle_mesh(
        directory / "regions_surface.ply",
        dense.mesh,
        dense.surface_colors,
    )
    (directory / "canonical_surface.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_shoe(
    output_root: Path,
    shoe: _PreparedShoe,
    dense: DenseCanonicalSuprAnatomy,
) -> None:
    directory = output_root / shoe.name
    directory.mkdir(parents=True, exist_ok=True)
    save_triangle_mesh(directory / "foot_dense.ply", shoe.dense_mesh)
    save_triangle_mesh(
        directory / "regions_longitudinal.ply",
        shoe.dense_mesh,
        dense.longitudinal_colors,
    )
    save_triangle_mesh(
        directory / "regions_surface.ply",
        shoe.dense_mesh,
        dense.surface_colors,
    )
    (directory / "anatomical_surface.json").write_text(
        json.dumps(shoe.payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    containment_root = args.containment_root.expanduser().resolve(strict=True)
    if not containment_root.is_dir():
        raise NotADirectoryError(containment_root)
    model_path = args.supr_model.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    names = _shoe_names(containment_root, list(args.shoes))
    existing = _existing_targets(output_root, names)
    if existing and not args.overwrite:
        formatted = ", ".join(str(path) for path in existing[:5])
        suffix = " ..." if len(existing) > 5 else ""
        raise FileExistsError(
            f"anatomical-surface artifacts already exist: {formatted}{suffix}; "
            "pass --overwrite to replace them"
        )

    anatomy = build_canonical_supr_anatomy(model_path)
    subdivision = build_supr_mesh_subdivision(
        anatomy.raw_mesh.faces,
        len(anatomy.raw_mesh.vertices),
        SUBDIVISION_LEVELS,
    )
    dense_reference = build_dense_canonical_supr_anatomy(anatomy, subdivision)
    prepared = [
        _prepare_shoe(
            name,
            containment_root,
            output_root,
            anatomy,
            dense_reference,
        )
        for name in names
    ]
    reference_payload = _reference_payload(
        model_path, anatomy, dense_reference
    )

    _write_reference(
        output_root / "reference",
        reference_payload,
        anatomy,
        dense_reference,
    )
    for shoe in prepared:
        _write_shoe(output_root, shoe, dense_reference)
    return {
        "shoe_count": len(prepared),
        "output_root": str(output_root),
        "subdivision": subdivision.to_dict(),
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
        raise SystemExit(f"anatomical surface failed: {error}") from error
    print(
        f"wrote {result['shoe_count']} anatomical surfaces to "
        f"{result['output_root']}"
    )
    subdivision = result["subdivision"]
    print(
        f"shared topology {subdivision['vertex_count']} vertices, "
        f"{subdivision['face_count']} faces, "
        f"sha256 {subdivision['topology_sha256']}"
    )


if __name__ == "__main__":
    main()
