"""Focused checks for the canonical foot-and-lower-leg anatomical volume."""

from __future__ import annotations

from pathlib import Path
import inspect

import numpy as np
import pytest

from foot_prior.anatomical_volume import (
    DENSE_FACE_COUNT,
    DENSE_VERTEX_COUNT,
    ENVELOPE_CENTER,
    ENVELOPE_POWER,
    ENVELOPE_RADII,
    _build_extended_anatomical_surface,
    _close_truncation,
    _directed_boundary_loop,
    _load_checkpoint_nine_reference,
    build_outer_envelope,
    map_volume_coordinates,
    orient_tetrahedra_positive,
    solve_harmonic_r,
)
from foot_prior.mesh import TriangleMesh
from foot_prior.supr_lower_leg import attach_lower_leg_to_fitted_dense_foot


REFERENCE_ROOT = Path(
    "/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/anatomical_surface"
)
FULL_BODY_MODEL = Path(
    "/storage/Abhinay/Shell_Gaussian/baselines/SUPR/data/supr_male.npy"
)


def test_orientation_barycentric_mapping_and_harmonic_boundaries() -> None:
    vertices = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.2, 0.2, 0.2),
        )
    )
    tetrahedra = np.asarray(
        ((4, 2, 1, 3), (0, 2, 4, 3), (0, 4, 1, 3), (0, 2, 1, 4)),
        dtype=np.int64,
    )
    oriented, volumes = orient_tetrahedra_positive(vertices, tetrahedra)
    assert np.all(volumes > 0.0)

    weights = np.asarray(((0.1, 0.2, 0.3, 0.4), (1.0, 0.0, 0.0, 0.0)))
    mapped = map_volume_coordinates(
        np.asarray((0, 1)), weights, vertices, oriented
    )
    expected = np.einsum("ni,nij->nj", weights, vertices[oriented[[0, 1]]])
    np.testing.assert_allclose(mapped, expected, atol=1.0e-15, rtol=0.0)

    values, gradients, residual = solve_harmonic_r(
        vertices,
        oriented,
        np.asarray((0,), dtype=np.int64),
        np.asarray((1, 2, 3), dtype=np.int64),
    )
    assert values[0] == 0.0
    np.testing.assert_array_equal(values[[1, 2, 3]], np.ones(3))
    assert 0.0 < values[4] < 1.0
    assert gradients.shape == (4, 3)
    assert np.isfinite(gradients).all()
    assert residual < 1.0e-12


def test_outer_envelope_is_frozen_and_deterministic() -> None:
    first = build_outer_envelope()
    second = build_outer_envelope()
    np.testing.assert_array_equal(first.vertices, second.vertices)
    np.testing.assert_array_equal(first.faces, second.faces)
    equation = np.sum(
        np.abs((first.vertices - ENVELOPE_CENTER) / ENVELOPE_RADII)
        ** ENVELOPE_POWER,
        axis=1,
    )
    np.testing.assert_allclose(equation, np.ones(len(equation)), atol=2.0e-14)


@pytest.mark.skipif(
    not REFERENCE_ROOT.is_dir() or not FULL_BODY_MODEL.is_file(),
    reason="Checkpoint 9 reference or full-body SUPR donor absent",
)
def test_extended_surface_preserves_foot_and_has_one_knee_boundary() -> None:
    reference = _load_checkpoint_nine_reference(REFERENCE_ROOT)
    assert reference.vertices.shape == (DENSE_VERTEX_COUNT, 3)
    assert reference.faces.shape == (DENSE_FACE_COUNT, 3)
    first = _build_extended_anatomical_surface(reference, FULL_BODY_MODEL)
    second = _build_extended_anatomical_surface(reference, FULL_BODY_MODEL)
    np.testing.assert_array_equal(first.vertices, second.vertices)
    np.testing.assert_array_equal(first.faces, second.faces)
    np.testing.assert_array_equal(first.vertices[:DENSE_VERTEX_COUNT], reference.vertices)
    np.testing.assert_array_equal(first.faces[:DENSE_FACE_COUNT], reference.faces)
    assert len(first.lower_leg.distal_boundary_vertex_indices) == 15
    assert len(first.lower_leg.proximal_boundary_vertex_indices) == 17
    assert len(first.lower_leg_indices) == 2_800
    assert len(first.bridge_face_indices) == 120
    assert len(first.knee_loop) == 68
    np.testing.assert_array_equal(_directed_boundary_loop(first.faces), first.knee_loop)

    closed_vertices, cap_faces, cap_index = _close_truncation(
        first.vertices, first.faces, first.knee_loop
    )
    np.testing.assert_array_equal(closed_vertices[:DENSE_VERTEX_COUNT], reference.vertices)
    np.testing.assert_array_equal(
        np.vstack((first.faces, cap_faces))[:DENSE_FACE_COUNT], reference.faces
    )
    assert cap_index == len(first.vertices)
    assert len(cap_faces) == 68

    broken = reference.faces[:-1]
    with pytest.raises(ValueError):
        _directed_boundary_loop(broken)

    envelope_value = np.sum(
        np.abs((closed_vertices - ENVELOPE_CENTER) / ENVELOPE_RADII)
        ** ENVELOPE_POWER,
        axis=1,
    )
    assert float(np.max(envelope_value)) < 1.0

    leg_offset = DENSE_VERTEX_COUNT
    correspondence = first.ankle_correspondence.copy()
    correspondence[:, 1] -= leg_offset
    attached = attach_lower_leg_to_fitted_dense_foot(
        TriangleMesh(reference.vertices, reference.faces),
        reference.vertices,
        first.vertices[first.lower_leg_indices],
        first.faces[first.lower_leg_face_indices] - leg_offset,
        reference.ankle_loop,
        correspondence,
    )
    np.testing.assert_array_equal(
        attached.mesh.vertices[:DENSE_VERTEX_COUNT], reference.vertices
    )
    np.testing.assert_allclose(
        attached.mesh.vertices[DENSE_VERTEX_COUNT:],
        first.vertices[DENSE_VERTEX_COUNT:],
        atol=1.0e-14,
        rtol=0.0,
    )
    np.testing.assert_array_equal(attached.mesh.faces, first.faces)
    np.testing.assert_allclose(
        attached.canonical_leg_to_fitted_ankle,
        np.eye(4),
        atol=1.0e-14,
        rtol=0.0,
    )


def test_installed_gmsh_classification_api_matches_checkpoint() -> None:
    gmsh = pytest.importorskip("gmsh")
    assert gmsh.__version__ == "4.15.2"
    parameters = inspect.signature(gmsh.model.mesh.classifySurfaces).parameters
    assert "boundary" in parameters
    assert "forReparametrization" in parameters
    assert "curveAngle" in parameters
