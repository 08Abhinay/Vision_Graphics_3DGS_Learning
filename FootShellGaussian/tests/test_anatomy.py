"""Focused checks for canonical and dense SUPR anatomical correspondence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from foot_prior.anatomy import (
    JOINT_NAMES,
    LONGITUDINAL_REGION_NAMES,
    SURFACE_REGION_NAMES,
    build_canonical_supr_anatomy,
    build_dense_canonical_supr_anatomy,
    map_surface_coordinates,
)
from foot_prior.mesh import load_triangle_mesh, save_triangle_mesh
from foot_prior.supr_foot import build_supr_mesh_subdivision


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPR_MODEL = REPOSITORY_ROOT / "baselines/SUPR/data/supr_male_right_foot.npy"
RUNNER = PROJECT_ROOT / "scripts/run_anatomical_surface.py"


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_canonical_reference_has_reversible_frame_and_complete_labels() -> None:
    first = build_canonical_supr_anatomy(SUPR_MODEL)
    second = build_canonical_supr_anatomy(SUPR_MODEL)

    np.testing.assert_allclose(
        first.raw_to_reference @ first.reference_to_raw,
        np.eye(4),
        atol=1e-12,
        rtol=0.0,
    )
    assert np.isclose(first.reference_mesh.bounds[0, 0], 0.0)
    assert np.isclose(first.reference_mesh.bounds[1, 1], 0.0)
    assert len(first.ankle_boundary_vertex_indices) == 15
    assert len(set(first.ankle_boundary_vertex_indices.tolist())) == 15
    assert len(JOINT_NAMES) == 13
    assert set(np.unique(first.longitudinal_vertex_labels)) == set(
        range(len(LONGITUDINAL_REGION_NAMES))
    )
    assert set(np.unique(first.surface_vertex_labels)) == set(
        range(len(SURFACE_REGION_NAMES))
    )
    np.testing.assert_array_equal(
        first.longitudinal_vertex_labels, second.longitudinal_vertex_labels
    )
    np.testing.assert_array_equal(
        first.surface_vertex_labels, second.surface_vertex_labels
    )
    assert first.landmarks == second.landmarks


def test_subdiv2_provenance_and_surface_coordinates_are_exact() -> None:
    anatomy = build_canonical_supr_anatomy(SUPR_MODEL)
    subdivision = build_supr_mesh_subdivision(
        anatomy.raw_mesh.faces, len(anatomy.raw_mesh.vertices), 2
    )
    dense = build_dense_canonical_supr_anatomy(anatomy, subdivision)

    assert dense.mesh.vertices.shape == (4151, 3)
    assert dense.mesh.faces.shape == (8240, 3)
    np.testing.assert_array_equal(
        dense.mesh.vertices[:266], anatomy.reference_mesh.vertices
    )
    np.testing.assert_allclose(
        subdivision.vertex_source_weights.sum(axis=1), 1.0, atol=1e-15
    )
    safe_indices = np.where(
        subdivision.vertex_source_indices < 0,
        0,
        subdivision.vertex_source_indices,
    )
    reconstructed = np.sum(
        anatomy.reference_mesh.vertices[safe_indices]
        * subdivision.vertex_source_weights[..., None],
        axis=1,
    )
    np.testing.assert_allclose(reconstructed, dense.mesh.vertices, atol=1e-15)
    assert subdivision.face_parent_indices.shape == (8240,)
    assert subdivision.face_parent_indices.min() == 0
    assert subdivision.face_parent_indices.max() == 514

    changed = anatomy.reference_mesh.vertices.copy()
    changed[:, 0] *= 1.07
    changed[:, 1] += 0.03 * changed[:, 0]
    expected = subdivision.apply_vertices(changed)
    mapped = map_surface_coordinates(
        dense.vertex_chart_face_indices,
        dense.vertex_chart_barycentric,
        changed,
        anatomy.raw_mesh.faces,
    )
    np.testing.assert_allclose(mapped, expected, atol=1e-15)


def test_runner_writes_repeatable_shared_topology_and_rejects_high_heels(
    tmp_path: Path,
) -> None:
    anatomy = build_canonical_supr_anatomy(SUPR_MODEL)
    containment_root = tmp_path / "containment"
    source_directory = containment_root / "normal_shoe"
    source_directory.mkdir(parents=True)
    mesh_path = source_directory / "foot_containment_fitted.ply"
    save_triangle_mesh(mesh_path, anatomy.reference_mesh)
    saved_mesh = load_triangle_mesh(mesh_path)
    identity = np.eye(4).tolist()
    metadata = {
        "schema_version": 5,
        "shoe_profile": "normal",
        "bounds": {"aligned_foot": saved_mesh.bounds.tolist()},
        "supr": {
            "pose_parameters_radians": np.zeros(39).tolist(),
            "betas": np.zeros(10).tolist(),
        },
        "transforms": {
            "posed_supr_to_normalized_shoe": identity,
            "normalized_shoe_to_posed_supr": identity,
            "posed_supr_to_original_shoe": identity,
            "original_shoe_to_posed_supr": identity,
        },
    }
    metadata_path = source_directory / "containment_fit.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    output_root = tmp_path / "anatomical"
    command = [
        sys.executable,
        str(RUNNER),
        "--containment-root",
        str(containment_root),
        "--supr-model",
        str(SUPR_MODEL),
        "--output-root",
        str(output_root),
        "--overwrite",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True)

    known_paths = [
        output_root / "reference/canonical_surface.json",
        output_root / "reference/canonical_surface.npz",
        output_root / "reference/neutral_dense.ply",
        output_root / "normal_shoe/anatomical_surface.json",
        output_root / "normal_shoe/foot_dense.ply",
    ]
    first_digests = [_digest(path) for path in known_paths]
    unrelated = output_root / "normal_shoe/keep.txt"
    unrelated.write_text("keep", encoding="utf-8")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, capture_output=True)
    assert first_digests == [_digest(path) for path in known_paths]
    assert unrelated.read_text(encoding="utf-8") == "keep"

    reference = np.load(output_root / "reference/canonical_surface.npz")
    fitted = load_triangle_mesh(output_root / "normal_shoe/foot_dense.ply")
    np.testing.assert_array_equal(fitted.faces, reference["dense_faces"])
    np.testing.assert_array_equal(
        fitted.vertices[:266], saved_mesh.vertices
    )

    metadata["shoe_profile"] = "high_heel"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    rejected_root = tmp_path / "rejected"
    rejected_command = command.copy()
    rejected_command[rejected_command.index(str(output_root))] = str(rejected_root)
    failed = subprocess.run(
        rejected_command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )
    assert failed.returncode != 0
    assert "normal shoes only" in failed.stderr
    assert not rejected_root.exists()
