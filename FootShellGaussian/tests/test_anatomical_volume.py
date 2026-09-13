"""Focused checks for the canonical foot-and-lower-leg anatomical volume."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import inspect
import shutil
import sys

import numpy as np
import pytest

from foot_prior.anatomical_volume import (
    BOUNDARY_FOOT_SKIN,
    ENVELOPE_CENTER,
    ENVELOPE_POWER,
    ENVELOPE_RADII,
    EXTENDED_FACE_COUNT,
    EXTENDED_VERTEX_COUNT,
    _build_computational_boundary,
    _close_truncation,
    _load_extended_reference,
    _self_intersection_pairs,
    _surface_coordinates,
    build_instance_boundary_target,
    build_outer_envelope,
    continue_instance_volume,
    load_canonical_anatomical_volume,
    load_instance_volume_problem,
    map_volume_coordinates,
    orient_tetrahedra_positive,
    solve_harmonic_r,
)
from foot_prior.mesh import load_triangle_mesh
from foot_prior.instance_volume_optimization import (
    _CollisionConstraints,
    _build_collision_constraints,
    _deformation_quality,
    _evaluate_objective,
    _evaluate_surface_objective,
    _grow_surface_repair_region,
    _full_vertices_with_inner,
    _initial_surface_repair_region,
    _triangle_triangle_closest_features,
    build_instance_optimization_system,
    optimization_configuration,
)
from scripts.run_instance_volume_deformation import (
    _load_resumable_continuation,
    _preflight_resume_state,
    parse_args as parse_instance_volume_args,
)


REFERENCE_ROOT = Path(
    "/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/extended_anatomical_surface"
)
VOLUME_ROOT = Path(
    "/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/anatomical_volume"
)
SANDAL_B3_PILOT = Path(
    "/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/temp/"
    "checkpoint_11_b3_pilot/sandal_1/instance_volume.npz"
)
SANDAL_B3_PILOT_ROOT = SANDAL_B3_PILOT.parent
FAST_SANDAL_PILOT = Path(
    "/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/temp/"
    "checkpoint_11_b3_fast_pilot/sandal_1"
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
    not REFERENCE_ROOT.is_dir(),
    reason="extended canonical anatomical reference absent",
)
def test_computational_boundary_repairs_without_changing_reference() -> None:
    reference = _load_extended_reference(REFERENCE_ROOT)
    original_vertices = reference.vertices.copy()
    original_faces = reference.faces.copy()
    assert reference.vertices.shape == (EXTENDED_VERTEX_COUNT, 3)
    assert reference.faces.shape == (EXTENDED_FACE_COUNT, 3)

    computational = _build_computational_boundary(reference)
    np.testing.assert_array_equal(reference.vertices, original_vertices)
    np.testing.assert_array_equal(reference.faces, original_faces)
    assert computational.vertices.shape == (6_904, 3)
    assert computational.faces.shape == (13_804, 3)
    assert len(computational.knee_cap_face_indices) == 68

    canonical_vertices, cap_faces, _ = _close_truncation(
        reference.vertices, reference.faces, reference.knee_loop
    )
    canonical_faces = np.vstack((reference.faces, cap_faces))
    closest, distances, face_indices, barycentric = _surface_coordinates(
        computational.vertices, canonical_vertices, canonical_faces
    )
    reconstructed = np.einsum(
        "ni,nij->nj",
        barycentric,
        canonical_vertices[canonical_faces[face_indices]],
    )
    np.testing.assert_allclose(reconstructed, closest, atol=1.0e-12, rtol=0.0)
    np.testing.assert_allclose(barycentric.sum(axis=1), np.ones(len(barycentric)))
    assert float(np.max(distances)) < 0.014


def test_installed_gmsh_classification_api_matches_checkpoint() -> None:
    gmsh = pytest.importorskip("gmsh")
    assert gmsh.__version__ == "4.15.2"
    parameters = inspect.signature(gmsh.model.mesh.classifySurfaces).parameters
    assert "boundary" in parameters
    assert "forReparametrization" in parameters
    assert "curveAngle" in parameters


def test_installed_pymeshfix_api_matches_checkpoint() -> None:
    pymeshfix = pytest.importorskip("pymeshfix")
    assert pymeshfix.__version__ == "0.18.1"
    assert hasattr(pymeshfix.PyTMesh, "strong_intersection_removal")


@pytest.mark.skipif(
    not (VOLUME_ROOT / "reference" / "canonical_volume.npz").is_file(),
    reason="canonical anatomical volume absent",
)
def test_saved_canonical_volume_loads_with_exact_boundary_topology() -> None:
    volume = load_canonical_anatomical_volume(VOLUME_ROOT)
    assert volume.computational_inner_vertex_indices.shape == (8_224,)
    assert volume.computational_inner_faces.shape == (16_444, 3)
    np.testing.assert_array_equal(
        volume.computational_inner_vertex_indices,
        np.arange(8_224, dtype=np.int64),
    )


@pytest.mark.skipif(
    not (VOLUME_ROOT / "reference" / "canonical_volume.npz").is_file(),
    reason="canonical anatomical volume absent",
)
def test_b3_identity_quality_and_objective_gradient() -> None:
    volume = load_canonical_anatomical_volume(VOLUME_ROOT)
    system = build_instance_optimization_system(volume)
    quality = _deformation_quality(system, volume.volume_vertices)
    np.testing.assert_allclose(
        quality.determinants, np.ones(len(volume.tetrahedra)), atol=2.0e-13
    )
    np.testing.assert_allclose(
        quality.singular_values, np.ones((len(volume.tetrahedra), 3)), atol=2.0e-13
    )

    rng = np.random.default_rng(7)
    movable = volume.volume_vertices[system.movable_vertex_indices].copy()
    movable += rng.normal(scale=1.0e-6, size=movable.shape)
    inner = volume.volume_vertices[system.inner_vertex_indices]
    target_edges = (
        inner[system.surface_edges[:, 1]] - inner[system.surface_edges[:, 0]]
    )
    empty = _CollisionConstraints(
        np.empty((0, 3), dtype=np.int64),
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.int64),
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.float64),
    )
    arguments = {
        "system": system,
        "fixed_vertices": volume.volume_vertices,
        "target_inner": inner,
        "target_edges": target_edges,
        "baseline": volume.volume_vertices,
        "surface_resolution": 0.013,
        "collision_constraints": empty,
        "barrier_multiplier": 1.0,
    }
    flat = movable.reshape(-1)
    _, gradient = _evaluate_objective(flat, **arguments)
    direction = rng.normal(size=flat.shape)
    direction /= np.linalg.norm(direction)
    epsilon = 3.0e-7
    plus, _ = _evaluate_objective(flat + epsilon * direction, **arguments)
    minus, _ = _evaluate_objective(flat - epsilon * direction, **arguments)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    analytical = float(np.dot(gradient, direction))
    assert analytical == pytest.approx(finite_difference, rel=1.0e-5, abs=1.0e-9)


def test_b3_triangle_closest_features_are_deterministic() -> None:
    first = np.asarray(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)))
    second = first + np.asarray((0.0, 0.0, 0.1))

    distance, first_weights, second_weights, normal = (
        _triangle_triangle_closest_features(first, second)
    )

    assert distance == pytest.approx(0.1)
    assert float(np.sum(first_weights)) == pytest.approx(1.0)
    assert float(np.sum(second_weights)) == pytest.approx(1.0)
    np.testing.assert_allclose(normal, np.asarray((0.0, 0.0, -1.0)))


@pytest.mark.skipif(
    not (REFERENCE_ROOT / "canvas_shoe" / "foot_lower_leg.ply").is_file(),
    reason="fitted canvas anatomy absent",
)
def test_canvas_boundary_target_preserves_correspondence_and_reports_toes() -> None:
    volume = load_canonical_anatomical_volume(VOLUME_ROOT)
    reference = _load_extended_reference(REFERENCE_ROOT)
    fitted = load_triangle_mesh(REFERENCE_ROOT / "canvas_shoe" / "foot_lower_leg.ply")
    target = build_instance_boundary_target(volume, reference, fitted)

    assert target.status == "ready_requires_untangling"
    assert target.vertices.shape == (8_224, 3)
    np.testing.assert_array_equal(target.faces, volume.computational_inner_faces)
    assert len(target.intersecting_face_pairs) > 0
    np.testing.assert_array_equal(
        np.unique(target.intersecting_face_pairs), target.intersecting_face_indices
    )
    assert np.all(
        target.face_labels[target.intersecting_face_indices] == BOUNDARY_FOOT_SKIN
    )
    assert float(np.max(target.reverse_distances)) <= 2.0 * target.surface_resolution


@pytest.mark.skipif(
    not (VOLUME_ROOT / "sandal_1" / "boundary_target.npz").is_file(),
    reason="saved sandal boundary target absent",
)
def test_sandal_instance_volume_problem_loads_recorded_intersections() -> None:
    problem = load_instance_volume_problem(VOLUME_ROOT, REFERENCE_ROOT, "sandal_1")

    assert problem.shoe_name == "sandal_1"
    assert problem.boundary_target.status == "ready_requires_untangling"
    assert problem.boundary_target.vertices.shape == (8_224, 3)
    assert problem.boundary_target.intersecting_face_pairs.shape == (2, 2)
    np.testing.assert_array_equal(
        problem.boundary_target.intersecting_face_indices,
        np.asarray((5732, 5816, 5826), dtype=np.int64),
    )


@pytest.mark.skipif(
    not (VOLUME_ROOT / "sandal_1" / "boundary_target.npz").is_file(),
    reason="saved sandal boundary target absent",
)
def test_sandal_localized_surface_region_is_deterministic() -> None:
    problem = load_instance_volume_problem(VOLUME_ROOT, REFERENCE_ROOT, "sandal_1")
    volume = problem.canonical_volume
    system = build_instance_optimization_system(volume)
    first = _initial_surface_repair_region(problem, system)
    repeated = _initial_surface_repair_region(problem, system)

    assert first.expansion_rings == 2
    assert not first.full_surface
    assert len(first.active_inner_vertex_indices) == 275
    np.testing.assert_array_equal(
        first.active_inner_vertex_indices, repeated.active_inner_vertex_indices
    )
    assert set(np.unique(problem.boundary_target.intersecting_face_pairs)).issubset(
        first.active_face_indices
    )

    canonical_inner = volume.volume_vertices[system.inner_vertex_indices]
    target_edges = (
        canonical_inner[system.surface_edge_inner_indices[:, 1]]
        - canonical_inner[system.surface_edge_inner_indices[:, 0]]
    )
    empty = _CollisionConstraints(
        np.empty((0, 3), dtype=np.int64),
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.int64),
        np.empty((0, 3), dtype=np.float64),
        np.empty((0, 3), dtype=np.float64),
    )
    template = volume.volume_vertices.copy()
    value, gradient = _evaluate_surface_objective(
        canonical_inner[first.active_inner_vertex_indices].reshape(-1),
        system=system,
        template_vertices=template,
        repair_region=first,
        target_inner=canonical_inner,
        target_edges=target_edges,
        surface_resolution=problem.boundary_target.surface_resolution,
        collision_constraints=empty,
        barrier_multiplier=1.0,
    )
    assert np.isfinite(value)
    assert gradient.shape == (3 * len(first.active_inner_vertex_indices),)
    np.testing.assert_array_equal(template, volume.volume_vertices)

    beta = 0.7342248141765595
    target_beta = (
        (1.0 - beta) * canonical_inner
        + beta * problem.boundary_target.vertices
    )
    target_complete = _full_vertices_with_inner(
        system, volume.volume_vertices, target_beta
    )
    constraints = _build_collision_constraints(
        system,
        target_complete,
        problem.boundary_target.surface_resolution,
        problem.boundary_target.intersecting_face_pairs,
        first,
    )
    target_edges = (
        target_beta[system.surface_edge_inner_indices[:, 1]]
        - target_beta[system.surface_edge_inner_indices[:, 0]]
    )
    arguments = {
        "system": system,
        "template_vertices": target_complete,
        "repair_region": first,
        "target_inner": target_beta,
        "target_edges": target_edges,
        "surface_resolution": problem.boundary_target.surface_resolution,
        "collision_constraints": constraints,
        "barrier_multiplier": 1.0,
    }
    flat = target_beta[first.active_inner_vertex_indices].reshape(-1)
    _, analytical_gradient = _evaluate_surface_objective(flat, **arguments)
    direction = -analytical_gradient / np.linalg.norm(analytical_gradient)
    epsilon = 1.0e-7
    plus, _ = _evaluate_surface_objective(flat + epsilon * direction, **arguments)
    minus, _ = _evaluate_surface_objective(flat - epsilon * direction, **arguments)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    analytical = float(np.dot(analytical_gradient, direction))
    assert analytical == pytest.approx(finite_difference, rel=1.0e-5, abs=1.0e-8)

    region = first
    for expected_rings in (3, 4, 5, 6):
        region = _grow_surface_repair_region(
            system, region, np.empty(0, dtype=np.int64)
        )
        assert region.expansion_rings == expected_rings
        assert not region.full_surface
    region = _grow_surface_repair_region(
        system, region, np.empty(0, dtype=np.int64)
    )
    assert region.full_surface
    assert len(region.active_inner_vertex_indices) == 8_224


@pytest.mark.skipif(
    not SANDAL_B3_PILOT.is_file(), reason="successful sandal B3 pilot absent"
)
def test_accelerated_exact_search_matches_saved_sandal_states() -> None:
    problem = load_instance_volume_problem(VOLUME_ROOT, REFERENCE_ROOT, "sandal_1")
    target_pairs = _self_intersection_pairs(
        problem.boundary_target.vertices, problem.boundary_target.faces
    )
    np.testing.assert_array_equal(
        target_pairs, problem.boundary_target.intersecting_face_pairs
    )
    with np.load(SANDAL_B3_PILOT, allow_pickle=False) as archive:
        final_vertices = archive["volume_vertices"]
    final_inner = final_vertices[
        problem.canonical_volume.computational_inner_vertex_indices
    ]
    assert not len(
        _self_intersection_pairs(final_inner, problem.boundary_target.faces)
    )


@pytest.mark.skipif(
    not (VOLUME_ROOT / "sandal_1" / "boundary_target.npz").is_file(),
    reason="saved sandal boundary target absent",
)
def test_instance_volume_problem_rejects_mismatched_target_digest(
    tmp_path: Path,
) -> None:
    volume_root = tmp_path / "anatomical_volume"
    volume_root.mkdir()
    (volume_root / "reference").symlink_to(
        VOLUME_ROOT / "reference", target_is_directory=True
    )
    target = volume_root / "sandal_1"
    target.mkdir()
    for name in ("boundary_target.json", "boundary_target.npz"):
        shutil.copyfile(VOLUME_ROOT / "sandal_1" / name, target / name)
    payload = json.loads((target / "boundary_target.json").read_text(encoding="utf-8"))
    payload["geometry"]["geometry_sha256"] = "0" * 64
    (target / "boundary_target.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="digest validation failed"):
        load_instance_volume_problem(volume_root, REFERENCE_ROOT, "sandal_1")


@pytest.mark.skipif(
    not (SANDAL_B3_PILOT_ROOT / "continuation_state.npz").is_file(),
    reason="saved sandal B2 pilot absent",
)
def test_explicit_resume_validates_b2_and_retries_failed_b3(
    tmp_path: Path,
) -> None:
    problem = load_instance_volume_problem(VOLUME_ROOT, REFERENCE_ROOT, "sandal_1")
    directory = tmp_path / "sandal_1"
    directory.mkdir()
    for name in ("continuation_state.json", "continuation_state.npz"):
        shutil.copyfile(SANDAL_B3_PILOT_ROOT / name, directory / name)

    continuation = _load_resumable_continuation(directory, problem)
    assert continuation.reached_alpha == pytest.approx(0.76904296875)
    assert continuation.status == "needs_11_b3"

    continuation_payload = json.loads(
        (directory / "continuation_state.json").read_text(encoding="utf-8")
    )
    failed_payload = {
        "schema_version": 2,
        "stage": "instance_volume_optimization",
        "shoe_name": "sandal_1",
        "status": "failed_11_b3",
        "configuration": optimization_configuration(),
        "digests": {
            "canonical_volume_topology_sha256": (
                problem.canonical_volume.topology_digest
            ),
            "boundary_target_geometry_sha256": (
                problem.boundary_target.geometry_digest
            ),
            "fitted_extended_surface_sha256": (
                problem.boundary_target.fitted_surface_digest
            ),
            "continuation_vertices_sha256": continuation_payload["digests"][
                "continuation_vertices_sha256"
            ],
        },
    }
    (directory / "instance_volume.json").write_text(
        json.dumps(failed_payload), encoding="utf-8"
    )
    system = build_instance_optimization_system(problem.canonical_volume)
    resumed = _preflight_resume_state(directory, problem, system)
    assert resumed.continuation is not None
    assert resumed.final_status is None

    continuation_payload["digests"]["boundary_target_geometry_sha256"] = "0" * 64
    (directory / "continuation_state.json").write_text(
        json.dumps(continuation_payload), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="input digests mismatch"):
        _load_resumable_continuation(directory, problem)


@pytest.mark.skipif(
    not (SANDAL_B3_PILOT_ROOT / "continuation_state.json").is_file(),
    reason="saved sandal B2 pilot absent",
)
def test_explicit_resume_rejects_partial_state(tmp_path: Path) -> None:
    problem = load_instance_volume_problem(VOLUME_ROOT, REFERENCE_ROOT, "sandal_1")
    directory = tmp_path / "sandal_1"
    directory.mkdir()
    shutil.copyfile(
        SANDAL_B3_PILOT_ROOT / "continuation_state.json",
        directory / "continuation_state.json",
    )
    with pytest.raises(ValueError, match="partial B2 artifacts"):
        _preflight_resume_state(directory, problem, None)


@pytest.mark.skipif(
    not (FAST_SANDAL_PILOT / "instance_volume.npz").is_file(),
    reason="accelerated sandal B3 pilot absent",
)
def test_explicit_resume_revalidates_completed_b3() -> None:
    problem = load_instance_volume_problem(VOLUME_ROOT, REFERENCE_ROOT, "sandal_1")
    system = build_instance_optimization_system(problem.canonical_volume)
    resumed = _preflight_resume_state(FAST_SANDAL_PILOT, problem, system)
    assert resumed.continuation is not None
    assert resumed.final_status == "final_corrected_target"


def test_resume_and_overwrite_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_instance_volume_deformation.py",
            "--anatomical-volume-root",
            "volume",
            "--extended-anatomical-surface-root",
            "surface",
            "--output-root",
            "output",
            "--resume",
            "--overwrite",
        ],
    )
    with pytest.raises(SystemExit):
        parse_instance_volume_args()


@pytest.mark.skipif(
    not (VOLUME_ROOT / "sandal_1" / "boundary_target.npz").is_file(),
    reason="saved sandal boundary target absent",
)
def test_identity_instance_volume_continuation_is_exact() -> None:
    problem = load_instance_volume_problem(VOLUME_ROOT, REFERENCE_ROOT, "sandal_1")
    volume = problem.canonical_volume
    identity_target = replace(
        problem.boundary_target,
        vertices=volume.volume_vertices[
            volume.computational_inner_vertex_indices
        ].copy(),
        intersecting_face_pairs=np.empty((0, 2), dtype=np.int64),
        intersecting_face_indices=np.empty(0, dtype=np.int64),
        status="ready",
        geometry_digest="identity",
    )
    identity_problem = replace(problem, boundary_target=identity_target)

    result = continue_instance_volume(identity_problem)

    assert result.status == "baseline_reached_target"
    assert result.reached_alpha == 1.0
    np.testing.assert_array_equal(result.volume_vertices, volume.volume_vertices)
    np.testing.assert_array_equal(
        result.jacobian_determinants,
        np.ones(len(volume.tetrahedra), dtype=np.float64),
    )
