#!/usr/bin/env python3
"""Transfer canonical anatomy across fitted SUPR foot-and-lower-leg surfaces."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import zipfile

import numpy as np

from foot_prior.anatomy import (
    COMPONENT_COLORS,
    COMPONENT_REGION_NAMES,
    DENSE_FOOT_FACE_COUNT,
    DENSE_FOOT_VERTEX_COUNT,
    EXTENDED_FACE_COUNT,
    EXTENDED_LONGITUDINAL_COLORS,
    EXTENDED_LONGITUDINAL_REGION_NAMES,
    EXTENDED_VERTEX_COUNT,
    SURFACE_COLORS,
    SURFACE_REGION_NAMES,
    ExtendedCanonicalSuprAnatomy,
    array_digest,
    build_extended_canonical_supr_anatomy,
    load_dense_canonical_supr_reference,
)
from foot_prior.mesh import TriangleMesh, load_triangle_mesh, save_triangle_mesh


REFERENCE_ARTIFACTS = (
    "canonical_extended_surface.json",
    "canonical_extended_surface.npz",
    "neutral_foot_lower_leg.ply",
    "regions_longitudinal.ply",
    "regions_surface.ply",
    "regions_components.ply",
)
SHOE_ARTIFACTS = (
    "extended_anatomical_surface.json",
    "foot_lower_leg.ply",
    "regions_longitudinal.ply",
    "regions_surface.ply",
    "regions_components.ply",
)


@dataclass(frozen=True)
class _PreparedShoe:
    name: str
    source_mesh_path: Path
    mesh: TriangleMesh
    payload: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one canonical anatomical map for the dense SUPR foot and "
            "lower leg, then transfer it to accepted lower-leg attachments."
        )
    )
    parser.add_argument("--anatomical-surface-root", required=True, type=Path)
    parser.add_argument("--lower-leg-root", required=True, type=Path)
    parser.add_argument("--full-body-supr-model", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("shoes", nargs="*")
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


def _label_record(
    names: tuple[str, ...],
    colors: np.ndarray,
    vertex_labels: np.ndarray,
    face_labels: np.ndarray,
) -> dict[str, Any]:
    return {
        "names": list(names),
        "rgba": {name: colors[index].tolist() for index, name in enumerate(names)},
        "vertex_counts": {
            name: int(np.count_nonzero(vertex_labels == index))
            for index, name in enumerate(names)
        },
        "face_counts": {
            name: int(np.count_nonzero(face_labels == index))
            for index, name in enumerate(names)
        },
    }


def _shoe_names(root: Path, requested: list[str]) -> list[str]:
    names = requested or [path.name for path in root.iterdir() if path.is_dir()]
    if not names:
        raise ValueError("lower-leg root contains no shoe directories")
    if len(set(names)) != len(names):
        raise ValueError("shoe names must be unique")
    for name in names:
        if not name or Path(name).name != name or name in {".", "..", "reference"}:
            raise ValueError(f"invalid shoe directory name: {name!r}")
    return sorted(names)


def _mapped_landmarks(
    mesh: TriangleMesh,
    landmarks: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for name, record in landmarks.items():
        indices = np.asarray(record.get("vertex_indices"), dtype=np.int64)
        primary = int(record.get("primary_vertex_index", -1))
        if (
            indices.ndim != 1
            or len(indices) == 0
            or np.any(indices < 0)
            or np.any(indices >= len(mesh.vertices))
            or primary not in set(indices.tolist())
        ):
            raise ValueError(f"canonical landmark {name!r} is invalid")
        result[name] = {
            "vertex_indices": indices.tolist(),
            "primary_vertex_index": primary,
            "point": mesh.vertices[indices].mean(axis=0).tolist(),
        }
    return result


def _validate_attachment_counts(record: dict[str, Any], name: str) -> None:
    counts = record.get("fit", {}).get("attachment", {}).get("counts")
    expected = {
        "foot_vertices": DENSE_FOOT_VERTEX_COUNT,
        "foot_faces": DENSE_FOOT_FACE_COUNT,
        "lower_leg_vertices": EXTENDED_VERTEX_COUNT - DENSE_FOOT_VERTEX_COUNT,
        "lower_leg_faces": EXTENDED_FACE_COUNT - DENSE_FOOT_FACE_COUNT - 120,
        "bridge_faces": 120,
    }
    if counts != expected:
        raise ValueError(f"{name}: lower-leg attachment counts are not canonical")


def _prepare_shoe(
    name: str,
    anatomical_root: Path,
    lower_leg_root: Path,
    output_root: Path,
    full_body_model: Path,
    canonical: ExtendedCanonicalSuprAnatomy,
) -> _PreparedShoe:
    source_dir = (lower_leg_root / name).resolve(strict=True)
    foot_dir = (anatomical_root / name).resolve(strict=True)
    if source_dir.parent != lower_leg_root or foot_dir.parent != anatomical_root:
        raise ValueError(f"{name}: input must be directly inside its pipeline root")
    metadata_path = source_dir / "lower_leg_attachment.json"
    mesh_path = source_dir / "foot_lower_leg.ply"
    foot_json_path = foot_dir / "anatomical_surface.json"
    foot_mesh_path = foot_dir / "foot_dense.ply"
    for path in (metadata_path, mesh_path, foot_json_path, foot_mesh_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = _load_json(metadata_path)
    foot_metadata = _load_json(foot_json_path)
    if (
        metadata.get("schema_version") != 2
        or metadata.get("stage") != "fitted_foot_natural_lower_leg_collar_fit"
        or metadata.get("shoe_profile") != "normal"
    ):
        raise ValueError(f"{name}: unsupported lower-leg attachment record")
    if (
        foot_metadata.get("schema_version") != 1
        or foot_metadata.get("stage") != "canonical_dense_supr_anatomical_surface"
        or foot_metadata.get("shoe_profile") != "normal"
    ):
        raise ValueError(f"{name}: unsupported foot anatomical-surface record")
    _validate_attachment_counts(metadata, name)

    recorded_anatomy = Path(metadata.get("inputs", {}).get("anatomical_surface", ""))
    if recorded_anatomy.expanduser().resolve() != foot_dir:
        raise ValueError(f"{name}: lower-leg attachment references another foot surface")
    recorded_foot_digest = metadata.get("inputs", {}).get("fitted_dense_foot_sha256")
    if recorded_foot_digest != _file_digest(foot_mesh_path):
        raise ValueError(f"{name}: fitted dense-foot digest does not match")
    recorded_model = Path(metadata.get("inputs", {}).get("full_body_supr_model", ""))
    if recorded_model.expanduser().resolve() != full_body_model:
        raise ValueError(f"{name}: lower-leg attachment used another SUPR donor")

    mesh = load_triangle_mesh(mesh_path)
    foot = load_triangle_mesh(foot_mesh_path)
    if mesh.vertices.shape != (EXTENDED_VERTEX_COUNT, 3) or mesh.faces.shape != (
        EXTENDED_FACE_COUNT,
        3,
    ):
        raise ValueError(f"{name}: joined mesh must use the 6,951/13,832 topology")
    if not np.array_equal(mesh.faces, canonical.faces):
        raise ValueError(f"{name}: joined topology differs from the canonical reference")
    if not np.array_equal(mesh.vertices[:DENSE_FOOT_VERTEX_COUNT], foot.vertices):
        raise ValueError(f"{name}: joined mesh changed the accepted fitted foot")
    if not np.array_equal(mesh.faces[:DENSE_FOOT_FACE_COUNT], foot.faces):
        raise ValueError(f"{name}: joined mesh changed the accepted foot topology")
    if not np.isfinite(mesh.vertices).all():
        raise ValueError(f"{name}: joined mesh contains non-finite vertices")

    selected = metadata.get("fit", {}).get("selected")
    if not isinstance(selected, dict):
        raise ValueError(f"{name}: selected lower-leg parameters are missing")
    output_dir = output_root / name
    payload = {
        "schema_version": 1,
        "stage": "extended_canonical_supr_anatomical_surface",
        "shoe_name": name,
        "shoe_profile": "normal",
        "inputs": {
            "lower_leg_attachment": str(source_dir),
            "lower_leg_attachment_json": str(metadata_path),
            "lower_leg_attachment_json_sha256": _file_digest(metadata_path),
            "joined_mesh": str(mesh_path),
            "joined_mesh_sha256": _file_digest(mesh_path),
            "foot_anatomical_surface": str(foot_dir),
            "canonical_reference": str(
                output_root / "reference" / "canonical_extended_surface.npz"
            ),
        },
        "topology": {
            "vertex_count": EXTENDED_VERTEX_COUNT,
            "face_count": EXTENDED_FACE_COUNT,
            "topology_sha256": canonical.topology_digest,
            "shared_vertex_and_face_ids": True,
            "foot_prefix_preserved": True,
        },
        "geometry": {
            "bounds": mesh.bounds.tolist(),
            "geometry_sha256": array_digest(mesh.vertices, mesh.faces),
            "source_mesh_copied_without_geometry_changes": True,
        },
        "anatomical_correspondence": {
            "method": "shared canonical vertex and face indices",
            "labels_transferred_without_reclassification": True,
            "landmarks": _mapped_landmarks(mesh, canonical.landmarks),
        },
        "lower_leg_fit": {
            "status": metadata.get("status"),
            "selected_parameters": selected,
            "parameters_or_geometry_changed_by_this_stage": False,
        },
        "artifacts": {artifact: str(output_dir / artifact) for artifact in SHOE_ARTIFACTS},
    }
    return _PreparedShoe(name=name, source_mesh_path=mesh_path, mesh=mesh, payload=payload)


def _reference_payload(
    anatomical_root: Path,
    full_body_model: Path,
    canonical: ExtendedCanonicalSuprAnatomy,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "stage": "canonical_dense_supr_foot_lower_leg_reference",
        "reference": "neutral right SUPR foot with neutral male right lower leg",
        "inputs": {
            "checkpoint9_reference": str(anatomical_root / "reference"),
            "full_body_supr_model": str(full_body_model),
            "full_body_supr_model_sha256": _file_digest(full_body_model),
        },
        "counts": {
            "vertices": len(canonical.vertices),
            "faces": len(canonical.faces),
            "foot_vertices": len(canonical.dense_foot_indices),
            "foot_faces": len(canonical.foot_face_indices),
            "lower_leg_vertices": len(canonical.lower_leg_indices),
            "lower_leg_faces": len(canonical.lower_leg_face_indices),
            "ankle_transition_faces": len(canonical.bridge_face_indices),
            "knee_boundary_vertices": len(canonical.knee_loop),
        },
        "coordinate_convention": {
            "x": "heel_to_toe; lower-leg anterior/posterior reference",
            "y": "positive_down; lower leg extends primarily toward negative Y",
            "z": "right-foot medial_negative_lateral_positive",
        },
        "surface_coordinates": {
            "foot": "preserved Checkpoint 9 native face plus barycentric weights",
            "extended": "joined dense face ID plus three barycentric weights",
            "mapping": "apply the coordinate to the same face ID in any fitted joined mesh",
        },
        "regions": {
            "interpretation": "longitudinal, surface and component maps are complementary",
            "longitudinal": _label_record(
                EXTENDED_LONGITUDINAL_REGION_NAMES,
                EXTENDED_LONGITUDINAL_COLORS,
                canonical.longitudinal_vertex_labels,
                canonical.longitudinal_face_labels,
            ),
            "surface": _label_record(
                SURFACE_REGION_NAMES,
                SURFACE_COLORS,
                canonical.surface_vertex_labels,
                canonical.surface_face_labels,
            ),
            "component": _label_record(
                COMPONENT_REGION_NAMES,
                COMPONENT_COLORS,
                canonical.component_vertex_labels,
                canonical.component_face_labels,
            ),
        },
        "joints": {
            "foot": {
                "names": list(canonical.foot_joint_names),
                "reference": canonical.foot_reference_joints.tolist(),
            },
            "lower_leg": {
                "names": list(canonical.lower_leg_joint_names),
                "full_body_source_indices": canonical.lower_leg_joint_source_indices.tolist(),
                "reference": canonical.lower_leg_reference_joints.tolist(),
            },
        },
        "landmarks": canonical.landmarks,
        "boundaries": {
            "ankle_correspondence": canonical.ankle_correspondence.tolist(),
            "knee_loop": canonical.knee_loop.tolist(),
        },
        "lower_leg": canonical.lower_leg_metadata,
        "ankle_attachment": canonical.attachment_diagnostics,
        "digests": {
            "topology_sha256": canonical.topology_digest,
            "canonical_geometry_sha256": canonical.digest,
        },
        "scope": {
            "completed": "canonical and fitted foot-and-lower-leg surface correspondence",
            "deferred": [
                "knee closure",
                "tetrahedral anatomical volume",
                "harmonic outward coordinate",
                "per-instance volume mapping",
            ],
        },
    }


def _write_known(directory: Path, artifacts: dict[str, Any]) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    try:
        artifacts["write"](staging)
        directory.mkdir(parents=True, exist_ok=True)
        for name in artifacts["names"]:
            os.replace(staging / name, directory / name)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _write_reference(
    directory: Path,
    payload: dict[str, Any],
    canonical: ExtendedCanonicalSuprAnatomy,
) -> None:
    def write(staging: Path) -> None:
        subdivision = canonical.lower_leg_subdivision
        landmark_names = tuple(canonical.landmarks)
        _write_deterministic_npz(
            staging / "canonical_extended_surface.npz",
            extended_reference_vertices=canonical.vertices,
            extended_faces=canonical.faces,
            foot_vertex_indices=canonical.dense_foot_indices,
            foot_face_indices=canonical.foot_face_indices,
            lower_leg_vertex_indices=canonical.lower_leg_indices,
            lower_leg_face_indices=canonical.lower_leg_face_indices,
            ankle_transition_face_indices=canonical.bridge_face_indices,
            knee_boundary_vertex_indices=canonical.knee_loop,
            ankle_loop_correspondence=canonical.ankle_correspondence,
            extended_vertex_chart_face_indices=canonical.vertex_chart_face_indices,
            extended_vertex_chart_barycentric=canonical.vertex_chart_barycentric,
            foot_native_chart_face_indices=canonical.foot_native_chart_face_indices,
            foot_native_chart_barycentric=canonical.foot_native_chart_barycentric,
            foot_native_reference_vertices=canonical.foot_native_vertices,
            foot_native_faces=canonical.foot_native_faces,
            raw_supr_to_reference=canonical.raw_supr_to_reference,
            reference_to_raw_supr=canonical.reference_to_raw_supr,
            foot_dense_vertex_source_indices=canonical.foot_dense_vertex_source_indices,
            foot_dense_vertex_source_weights=canonical.foot_dense_vertex_source_weights,
            foot_dense_face_parent_indices=canonical.foot_dense_face_parent_indices,
            longitudinal_vertex_labels=canonical.longitudinal_vertex_labels,
            longitudinal_face_labels=canonical.longitudinal_face_labels,
            surface_vertex_labels=canonical.surface_vertex_labels,
            surface_face_labels=canonical.surface_face_labels,
            component_vertex_labels=canonical.component_vertex_labels,
            component_face_labels=canonical.component_face_labels,
            lower_leg_dense_vertex_source_indices=subdivision.vertex_source_indices,
            lower_leg_dense_vertex_source_weights=subdivision.vertex_source_weights,
            lower_leg_dense_face_parent_indices=subdivision.face_parent_indices,
            lower_leg_native_reference_vertices=canonical.lower_leg.mesh.vertices,
            lower_leg_native_faces=canonical.lower_leg.mesh.faces,
            lower_leg_source_full_body_vertex_indices=canonical.lower_leg.source_vertex_indices,
            lower_leg_source_full_body_face_indices=canonical.lower_leg.source_face_indices,
            foot_joint_names=np.asarray(canonical.foot_joint_names),
            foot_reference_joints=canonical.foot_reference_joints,
            lower_leg_joint_names=np.asarray(canonical.lower_leg_joint_names),
            lower_leg_joint_source_indices=canonical.lower_leg_joint_source_indices,
            lower_leg_reference_joints=canonical.lower_leg_reference_joints,
            anatomical_frame=canonical.anatomical_frame,
            landmark_names=np.asarray(landmark_names),
            landmark_primary_vertex_indices=np.asarray(
                [canonical.landmarks[name]["primary_vertex_index"] for name in landmark_names],
                dtype=np.int64,
            ),
            landmark_reference_positions=np.asarray(
                [canonical.landmarks[name]["point_reference"] for name in landmark_names],
                dtype=np.float64,
            ),
        )
        mesh = canonical.mesh
        save_triangle_mesh(staging / "neutral_foot_lower_leg.ply", mesh)
        save_triangle_mesh(
            staging / "regions_longitudinal.ply", mesh, canonical.longitudinal_colors
        )
        save_triangle_mesh(staging / "regions_surface.ply", mesh, canonical.surface_colors)
        save_triangle_mesh(
            staging / "regions_components.ply", mesh, canonical.component_colors
        )
        (staging / "canonical_extended_surface.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    _write_known(directory, {"names": REFERENCE_ARTIFACTS, "write": write})


def _write_shoe(
    directory: Path,
    prepared: _PreparedShoe,
    canonical: ExtendedCanonicalSuprAnatomy,
) -> None:
    def write(staging: Path) -> None:
        shutil.copyfile(prepared.source_mesh_path, staging / "foot_lower_leg.ply")
        save_triangle_mesh(
            staging / "regions_longitudinal.ply",
            prepared.mesh,
            canonical.longitudinal_colors,
        )
        save_triangle_mesh(
            staging / "regions_surface.ply", prepared.mesh, canonical.surface_colors
        )
        save_triangle_mesh(
            staging / "regions_components.ply", prepared.mesh, canonical.component_colors
        )
        (staging / "extended_anatomical_surface.json").write_text(
            json.dumps(prepared.payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    _write_known(directory, {"names": SHOE_ARTIFACTS, "write": write})


def run(args: argparse.Namespace) -> dict[str, Any]:
    anatomical_root = args.anatomical_surface_root.expanduser().resolve(strict=True)
    lower_leg_root = args.lower_leg_root.expanduser().resolve(strict=True)
    full_body_model = args.full_body_supr_model.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    names = _shoe_names(lower_leg_root, list(args.shoes))
    known = [output_root / "reference" / name for name in REFERENCE_ARTIFACTS]
    for shoe in names:
        known.extend(output_root / shoe / name for name in SHOE_ARTIFACTS)
    existing = [path for path in known if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"extended anatomical artifacts already exist; pass --overwrite: {existing[0]}"
        )

    reference = load_dense_canonical_supr_reference(anatomical_root)
    canonical = build_extended_canonical_supr_anatomy(reference, full_body_model)
    prepared = [
        _prepare_shoe(
            name,
            anatomical_root,
            lower_leg_root,
            output_root,
            full_body_model,
            canonical,
        )
        for name in names
    ]
    payload = _reference_payload(anatomical_root, full_body_model, canonical)
    _write_reference(output_root / "reference", payload, canonical)
    for shoe in prepared:
        _write_shoe(output_root / shoe.name, shoe, canonical)
    return {
        "stage": payload["stage"],
        "shoe_count": len(prepared),
        "vertex_count": len(canonical.vertices),
        "face_count": len(canonical.faces),
        "topology_sha256": canonical.topology_digest,
        "output_root": str(output_root),
    }


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
