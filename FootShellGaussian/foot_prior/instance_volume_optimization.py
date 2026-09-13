"""Bounded-distortion Checkpoint 11-B3 instance-volume optimization."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any, Callable

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .anatomical_volume import (
    BOUNDARY_OUTER_ENVELOPE,
    ENVELOPE_CENTER,
    ENVELOPE_POWER,
    ENVELOPE_RADII,
    INSTANCE_JACOBIAN_DETERMINANT_FLOOR,
    CanonicalAnatomicalVolume,
    InstanceVolumeContinuation,
    InstanceVolumeProblem,
    _InstanceDeformationSystem,
    _self_intersection_pairs,
    _tetrahedral_stiffness,
    _validate_continuation_candidate,
)
from .cavity import (
    _TriangleBroadPhase,
    _build_triangle_broad_phase,
    _find_collision_pairs,
)


B3_INITIAL_BETA_STEP = 0.05
B3_MINIMUM_BETA_STEP = 0.001
B3_OPTIMIZER_BLOCKS = 4
B3_ITERATIONS_PER_BLOCK = 25
B3_BARRIER_MULTIPLIERS = (1.0, 10.0, 100.0, 100.0)
B3_TARGET_WEIGHT = 10.0
B3_SURFACE_WEIGHT = 1.0
B3_FEM_WEIGHT = 1.0
B3_DISTORTION_WEIGHT = 0.1
B3_DETERMINANT_ACTIVATION = 0.1
B3_FINAL_MINIMUM_DETERMINANT = 0.02
B3_FINAL_MINIMUM_SINGULAR_VALUE = 0.05
B3_FINAL_MAXIMUM_SINGULAR_VALUE = 5.0
B3_FINAL_MAXIMUM_CONDITION_NUMBER = 20.0
B3_MAXIMUM_CORRECTION_RESOLUTIONS = 0.5
B3_COLLISION_ACTIVATION_RESOLUTIONS = 0.25
B3_COLLISION_MINIMUM_RESOLUTIONS = 1.0e-4
B3_MINIMUM_TRIANGLE_AREA_RATIO = 0.5
B3_MAXIMUM_TRIANGLE_AREA_RATIO = 2.0
B3_EXACT_CORRECTION_RESOLUTIONS = 1.0e-10
B3_SURFACE_VARIABLE_SCALE_RESOLUTIONS = 0.1
B3_INITIAL_REPAIR_RINGS = 2
B3_MAXIMUM_REPAIR_RINGS = 6
B3_MAXIMUM_LOCAL_VERTEX_FRACTION = 0.25


@dataclass(frozen=True)
class _CollisionConstraints:
    """Fixed closest-feature linearizations for one optimizer block."""

    first_vertex_indices: np.ndarray
    first_weights: np.ndarray
    second_vertex_indices: np.ndarray
    second_weights: np.ndarray
    normals: np.ndarray

    @property
    def count(self) -> int:
        return int(len(self.normals))


@dataclass(frozen=True)
class _InstanceOptimizationSystem:
    """Canonical arrays reused by every Checkpoint 11-B3 instance."""

    topology_digest: str
    movable_vertex_indices: np.ndarray
    global_to_movable: np.ndarray
    global_to_inner: np.ndarray
    inner_vertex_indices: np.ndarray
    outer_vertex_indices: np.ndarray
    interior_vertex_indices: np.ndarray
    tetrahedra: np.ndarray
    canonical_inverse_matrices: np.ndarray
    canonical_signed_volumes: np.ndarray
    canonical_volume_weights: np.ndarray
    stiffness: sparse.csr_matrix
    inner_faces: np.ndarray
    inner_global_faces: np.ndarray
    outer_faces: np.ndarray
    surface_edges: np.ndarray
    surface_edge_inner_indices: np.ndarray
    surface_area_weights: np.ndarray
    inner_vertex_incident_faces: tuple[np.ndarray, ...]
    inner_face_adjacency: frozenset[tuple[int, int]]
    boundary_tetrahedron_indices: np.ndarray
    outer_collision_index: _TriangleBroadPhase
    canonical_shell_volume: float


@dataclass(frozen=True)
class _SurfaceRepairRegion:
    """Deterministic local degrees of freedom for surface untangling."""

    active_inner_vertex_indices: np.ndarray
    active_face_indices: np.ndarray
    active_edge_indices: np.ndarray
    affected_boundary_tetrahedron_indices: np.ndarray
    expansion_rings: int
    seed_intersection_face_count: int
    seed_bad_tetrahedron_count: int
    full_surface: bool


@dataclass(frozen=True)
class _QualityDiagnostics:
    determinants: np.ndarray
    singular_values: np.ndarray
    condition_numbers: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "jacobian_determinant": _summary(self.determinants),
            "minimum_singular_value": _summary(self.singular_values[:, 2]),
            "maximum_singular_value": _summary(self.singular_values[:, 0]),
            "condition_number": _summary(self.condition_numbers),
        }


@dataclass(frozen=True)
class _FinalValidation:
    accepted: bool
    failure: str | None
    quality: _QualityDiagnostics
    target_correction_vectors: np.ndarray
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class _BoundaryValidation:
    """Safety result for a computational boundary without a volume solve."""

    accepted: bool
    failure: str | None
    self_intersection_count: int
    inner_outer_intersection_count: int
    envelope_outside_vertex_count: int
    degenerate_face_count: int
    self_intersection_pairs: np.ndarray | None = None
    inner_outer_intersection_pairs: np.ndarray | None = None


@dataclass(frozen=True)
class _SurfaceUntangling:
    """Collision-free computational target produced before volume fitting."""

    vertices: np.ndarray
    reached_beta: float
    history: tuple[dict[str, Any], ...]
    failure: str | None
    optimizer_iterations: int
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class InstanceVolumeOptimization:
    """Final or last-valid Checkpoint 11-B3 state for one shoe."""

    shoe_name: str
    status: str
    volume_vertices: np.ndarray
    jacobian_determinants: np.ndarray
    jacobian_singular_values: np.ndarray
    condition_numbers: np.ndarray
    target_correction_vectors: np.ndarray
    reached_beta: float
    optimization_history: tuple[dict[str, Any], ...]
    diagnostics: dict[str, Any]
    canonical_volume_topology_digest: str
    boundary_target_geometry_digest: str

    def to_dict(self) -> dict[str, Any]:
        """Return portable metadata without embedding large numerical arrays."""

        return {
            "schema_version": 2,
            "stage": "instance_volume_optimization",
            "status": self.status,
            "meaning": (
                "usable Checkpoint 11-B3 instance volume"
                if self.status != "failed_11_b3"
                else "last valid state only; not usable for chi_i or Phi_i"
            ),
            "shoe_name": self.shoe_name,
            "reached_beta": self.reached_beta,
            "configuration": optimization_configuration(),
            "counts": {
                "volume_vertices": int(len(self.volume_vertices)),
                "tetrahedra": int(len(self.jacobian_determinants)),
                "optimization_history": int(len(self.optimization_history)),
                "corrected_inner_vertices": int(
                    np.count_nonzero(
                        np.linalg.norm(self.target_correction_vectors, axis=1)
                        > B3_EXACT_CORRECTION_RESOLUTIONS
                        * float(self.diagnostics["surface_resolution"])
                    )
                ),
            },
            "final": {
                "jacobian_determinant": _summary(self.jacobian_determinants),
                "minimum_singular_value": _summary(
                    self.jacobian_singular_values[:, 2]
                ),
                "maximum_singular_value": _summary(
                    self.jacobian_singular_values[:, 0]
                ),
                "condition_number": _summary(self.condition_numbers),
                **self.diagnostics,
            },
            "history": list(self.optimization_history),
            "digests": {
                "canonical_volume_topology_sha256": (
                    self.canonical_volume_topology_digest
                ),
                "boundary_target_geometry_sha256": (
                    self.boundary_target_geometry_digest
                ),
            },
        }


def optimization_configuration() -> dict[str, Any]:
    """Return the fixed, shoe-independent Checkpoint 11-B3 policy."""

    return {
        "backend": "scipy_lbfgsb_cpu_explicit_numpy_gradients",
        "phases": [
            "topology_preserving_computational_surface_untangling",
            "fixed_inner_boundary_tetrahedral_optimization",
        ],
        "initial_beta_step": B3_INITIAL_BETA_STEP,
        "minimum_beta_step": B3_MINIMUM_BETA_STEP,
        "optimizer_blocks_per_beta": B3_OPTIMIZER_BLOCKS,
        "iterations_per_block": B3_ITERATIONS_PER_BLOCK,
        "robust_seed_bisection_iterations": 14,
        "accepted_block_minimum_line_step": 2.0**-20,
        "barrier_multipliers": list(B3_BARRIER_MULTIPLIERS),
        "energy_weights": {
            "target": B3_TARGET_WEIGHT,
            "surface": B3_SURFACE_WEIGHT,
            "fem": B3_FEM_WEIGHT,
            "distortion": B3_DISTORTION_WEIGHT,
        },
        "determinant": {
            "emergency_floor": INSTANCE_JACOBIAN_DETERMINANT_FLOOR,
            "barrier_activation": B3_DETERMINANT_ACTIVATION,
            "final_minimum": B3_FINAL_MINIMUM_DETERMINANT,
        },
        "singular_values": {
            "final_minimum": B3_FINAL_MINIMUM_SINGULAR_VALUE,
            "final_maximum": B3_FINAL_MAXIMUM_SINGULAR_VALUE,
            "final_maximum_condition_number": (
                B3_FINAL_MAXIMUM_CONDITION_NUMBER
            ),
        },
        "boundary": {
            "maximum_correction_surface_resolutions": (
                B3_MAXIMUM_CORRECTION_RESOLUTIONS
            ),
            "minimum_triangle_area_ratio": B3_MINIMUM_TRIANGLE_AREA_RATIO,
            "maximum_triangle_area_ratio": B3_MAXIMUM_TRIANGLE_AREA_RATIO,
            "outer_envelope": "fixed exactly",
        },
        "collision": {
            "activation_surface_resolutions": (
                B3_COLLISION_ACTIVATION_RESOLUTIONS
            ),
            "minimum_surface_resolutions": B3_COLLISION_MINIMUM_RESOLUTIONS,
            "exact_intersection_acceptance": True,
        },
        "localized_surface_repair": {
            "variable_scale_surface_resolutions": (
                B3_SURFACE_VARIABLE_SCALE_RESOLUTIONS
            ),
            "initial_face_rings": B3_INITIAL_REPAIR_RINGS,
            "maximum_face_rings": B3_MAXIMUM_REPAIR_RINGS,
            "maximum_local_vertex_fraction": (
                B3_MAXIMUM_LOCAL_VERTEX_FRACTION
            ),
            "full_surface_fallback": True,
        },
        "per_shoe_tuning": False,
        "randomness": False,
    }


def _summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not len(array) or not np.isfinite(array).all():
        raise ValueError("diagnostic array must be finite and nonempty")
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95.0)),
        "p99": float(np.percentile(array, 99.0)),
        "maximum": float(np.max(array)),
    }


def build_instance_optimization_system(
    canonical_volume: CanonicalAnatomicalVolume,
) -> _InstanceOptimizationSystem:
    """Build the immutable B3 system once for a complete instance batch."""

    volume = canonical_volume
    tetrahedra = np.asarray(volume.tetrahedra, dtype=np.int64)
    canonical_points = volume.volume_vertices[tetrahedra]
    canonical_matrices = np.stack(
        (
            canonical_points[:, 1] - canonical_points[:, 0],
            canonical_points[:, 2] - canonical_points[:, 0],
            canonical_points[:, 3] - canonical_points[:, 0],
        ),
        axis=2,
    )
    inverse_matrices = np.linalg.inv(canonical_matrices)
    if not np.isfinite(inverse_matrices).all():
        raise ValueError("canonical tetrahedron inverse matrices are invalid")

    cells, volumes, _, stiffness = _tetrahedral_stiffness(
        volume.volume_vertices, tetrahedra
    )
    if not np.array_equal(cells, tetrahedra) or not np.allclose(
        volumes,
        volume.tetrahedron_signed_volumes,
        atol=1.0e-15,
        rtol=1.0e-12,
    ):
        raise ValueError("canonical optimization tetrahedra are inconsistent")

    movable_mask = np.ones(len(volume.volume_vertices), dtype=bool)
    movable_mask[volume.outer_vertex_indices] = False
    movable = np.flatnonzero(movable_mask)
    interior_mask = movable_mask.copy()
    interior_mask[volume.computational_inner_vertex_indices] = False
    interior = np.flatnonzero(interior_mask)
    global_to_movable = np.full(len(volume.volume_vertices), -1, dtype=np.int64)
    global_to_movable[movable] = np.arange(len(movable), dtype=np.int64)

    inner = np.asarray(volume.computational_inner_vertex_indices, dtype=np.int64)
    global_to_inner = np.full(len(volume.volume_vertices), -1, dtype=np.int64)
    global_to_inner[inner] = np.arange(len(inner), dtype=np.int64)
    inner_faces = np.asarray(volume.computational_inner_faces, dtype=np.int64)
    inner_global_faces = inner[inner_faces]
    edge_pairs = np.concatenate(
        (
            inner_global_faces[:, (0, 1)],
            inner_global_faces[:, (1, 2)],
            inner_global_faces[:, (2, 0)],
        ),
        axis=0,
    )
    edges = np.unique(np.sort(edge_pairs, axis=1), axis=0)
    local_edges = global_to_inner[edges]
    if np.any(local_edges < 0):
        raise ValueError("computational-boundary edge contains a non-inner vertex")

    incident_faces: list[list[int]] = [[] for _ in range(len(inner))]
    for face_index, face in enumerate(inner_faces):
        for vertex_index in face:
            incident_faces[int(vertex_index)].append(face_index)
    adjacent_faces: set[tuple[int, int]] = set()
    for incident in incident_faces:
        for first_offset, first_face in enumerate(incident):
            for second_face in incident[first_offset + 1 :]:
                adjacent_faces.add(
                    (min(first_face, second_face), max(first_face, second_face))
                )

    triangles = volume.volume_vertices[inner_global_faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    vertex_areas = np.zeros(len(inner), dtype=np.float64)
    for corner in range(3):
        np.add.at(vertex_areas, inner_faces[:, corner], areas / 3.0)
    if np.any(vertex_areas <= 0.0) or not np.isfinite(vertex_areas).all():
        raise ValueError("computational boundary has invalid vertex areas")
    vertex_areas /= np.sum(vertex_areas)

    outer_faces = volume.boundary_faces[
        volume.boundary_labels == BOUNDARY_OUTER_ENVELOPE
    ]
    outer_faces = np.asarray(outer_faces, dtype=np.int64)
    outer_collision_index = _build_triangle_broad_phase(
        volume.volume_vertices[outer_faces],
        np.arange(len(outer_faces), dtype=np.int64),
    )
    boundary_tetrahedra = np.flatnonzero(
        np.all(global_to_inner[tetrahedra] >= 0, axis=1)
    )
    if not len(boundary_tetrahedra):
        raise ValueError("canonical volume has no inner-boundary tetrahedra")
    shell_volume = float(np.sum(volumes))
    return _InstanceOptimizationSystem(
        topology_digest=volume.topology_digest,
        movable_vertex_indices=movable,
        global_to_movable=global_to_movable,
        global_to_inner=global_to_inner,
        inner_vertex_indices=inner,
        outer_vertex_indices=np.asarray(volume.outer_vertex_indices, dtype=np.int64),
        interior_vertex_indices=interior,
        tetrahedra=tetrahedra,
        canonical_inverse_matrices=inverse_matrices,
        canonical_signed_volumes=np.asarray(volumes, dtype=np.float64),
        canonical_volume_weights=volumes / shell_volume,
        stiffness=stiffness[movable][:, movable].tocsr(),
        inner_faces=inner_faces,
        inner_global_faces=inner_global_faces,
        outer_faces=np.asarray(outer_faces, dtype=np.int64),
        surface_edges=edges,
        surface_edge_inner_indices=local_edges,
        surface_area_weights=vertex_areas,
        inner_vertex_incident_faces=tuple(
            np.asarray(items, dtype=np.int64) for items in incident_faces
        ),
        inner_face_adjacency=frozenset(adjacent_faces),
        boundary_tetrahedron_indices=boundary_tetrahedra,
        outer_collision_index=outer_collision_index,
        canonical_shell_volume=shell_volume,
    )


def _quality_failure_mask(quality: _QualityDiagnostics) -> np.ndarray:
    return (
        ~np.isfinite(quality.determinants)
        | (quality.determinants < B3_FINAL_MINIMUM_DETERMINANT)
        | ~np.isfinite(quality.singular_values).all(axis=1)
        | (quality.singular_values[:, 2] < B3_FINAL_MINIMUM_SINGULAR_VALUE)
        | (quality.singular_values[:, 0] > B3_FINAL_MAXIMUM_SINGULAR_VALUE)
        | ~np.isfinite(quality.condition_numbers)
        | (quality.condition_numbers > B3_FINAL_MAXIMUM_CONDITION_NUMBER)
    )


def _expanded_surface_faces(
    system: _InstanceOptimizationSystem,
    seed_face_indices: np.ndarray,
    rings: int,
) -> np.ndarray:
    selected = set(np.asarray(seed_face_indices, dtype=np.int64).tolist())
    for _ in range(rings):
        if not selected:
            break
        vertices = np.unique(
            system.inner_faces[np.asarray(sorted(selected), dtype=np.int64)]
        )
        selected.update(
            int(face_index)
            for vertex_index in vertices
            for face_index in system.inner_vertex_incident_faces[int(vertex_index)]
        )
    return np.asarray(sorted(selected), dtype=np.int64)


def _surface_repair_region(
    system: _InstanceOptimizationSystem,
    face_indices: np.ndarray,
    *,
    expansion_rings: int,
    seed_intersection_face_count: int,
    seed_bad_tetrahedron_count: int,
    full_surface: bool = False,
) -> _SurfaceRepairRegion:
    if full_surface:
        active_faces = np.arange(len(system.inner_faces), dtype=np.int64)
        active_vertices = np.arange(len(system.inner_vertex_indices), dtype=np.int64)
        active_edges = np.arange(len(system.surface_edges), dtype=np.int64)
        affected_tetrahedra = system.boundary_tetrahedron_indices.copy()
    else:
        active_faces = np.unique(np.asarray(face_indices, dtype=np.int64))
        active_vertices = (
            np.unique(system.inner_faces[active_faces])
            if len(active_faces)
            else np.empty(0, dtype=np.int64)
        )
        vertex_mask = np.zeros(len(system.inner_vertex_indices), dtype=bool)
        vertex_mask[active_vertices] = True
        active_edges = np.flatnonzero(
            np.any(vertex_mask[system.surface_edge_inner_indices], axis=1)
        )
        boundary_local = system.global_to_inner[
            system.tetrahedra[system.boundary_tetrahedron_indices]
        ]
        affected_tetrahedra = system.boundary_tetrahedron_indices[
            np.any(vertex_mask[boundary_local], axis=1)
        ]
    return _SurfaceRepairRegion(
        active_inner_vertex_indices=active_vertices,
        active_face_indices=active_faces,
        active_edge_indices=active_edges,
        affected_boundary_tetrahedron_indices=affected_tetrahedra,
        expansion_rings=expansion_rings,
        seed_intersection_face_count=seed_intersection_face_count,
        seed_bad_tetrahedron_count=seed_bad_tetrahedron_count,
        full_surface=full_surface,
    )


def _initial_surface_repair_region(
    problem: InstanceVolumeProblem,
    system: _InstanceOptimizationSystem,
) -> _SurfaceRepairRegion:
    volume = problem.canonical_volume
    target_complete = _full_vertices_with_inner(
        system, volume.volume_vertices, problem.boundary_target.vertices
    )
    quality = _selected_deformation_quality(
        system, target_complete, system.boundary_tetrahedron_indices
    )
    bad_tetrahedra = system.boundary_tetrahedron_indices[
        _quality_failure_mask(quality)
    ]
    intersection_faces = (
        np.unique(problem.boundary_target.intersecting_face_pairs)
        if len(problem.boundary_target.intersecting_face_pairs)
        else np.empty(0, dtype=np.int64)
    )
    seed_faces = set(intersection_faces.tolist())
    if len(bad_tetrahedra):
        bad_vertices = np.unique(system.tetrahedra[bad_tetrahedra])
        bad_inner = system.global_to_inner[bad_vertices]
        for vertex_index in bad_inner:
            seed_faces.update(
                int(face_index)
                for face_index in system.inner_vertex_incident_faces[
                    int(vertex_index)
                ]
            )
    expanded = _expanded_surface_faces(
        system,
        np.asarray(sorted(seed_faces), dtype=np.int64),
        B3_INITIAL_REPAIR_RINGS,
    )
    return _surface_repair_region(
        system,
        expanded,
        expansion_rings=B3_INITIAL_REPAIR_RINGS,
        seed_intersection_face_count=int(len(intersection_faces)),
        seed_bad_tetrahedron_count=int(len(bad_tetrahedra)),
    )


def _grow_surface_repair_region(
    system: _InstanceOptimizationSystem,
    region: _SurfaceRepairRegion,
    additional_face_indices: np.ndarray,
) -> _SurfaceRepairRegion:
    if region.full_surface:
        return region
    seeds = np.unique(
        np.concatenate(
            (
                region.active_face_indices,
                np.asarray(additional_face_indices, dtype=np.int64),
            )
        )
    )
    expanded = _expanded_surface_faces(system, seeds, 1)
    next_rings = region.expansion_rings + 1
    maximum_vertices = int(
        math.floor(
            B3_MAXIMUM_LOCAL_VERTEX_FRACTION * len(system.inner_vertex_indices)
        )
    )
    active_count = (
        len(np.unique(system.inner_faces[expanded])) if len(expanded) else 0
    )
    use_full_surface = bool(
        next_rings > B3_MAXIMUM_REPAIR_RINGS
        or active_count > maximum_vertices
    )
    return _surface_repair_region(
        system,
        expanded,
        expansion_rings=next_rings,
        seed_intersection_face_count=region.seed_intersection_face_count,
        seed_bad_tetrahedron_count=region.seed_bad_tetrahedron_count,
        full_surface=use_full_surface,
    )


def _bad_boundary_face_indices(
    system: _InstanceOptimizationSystem,
    vertices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    quality = _selected_deformation_quality(
        system, vertices, system.boundary_tetrahedron_indices
    )
    bad_tetrahedra = system.boundary_tetrahedron_indices[
        _quality_failure_mask(quality)
    ]
    faces: set[int] = set()
    if len(bad_tetrahedra):
        inner_vertices = system.global_to_inner[
            np.unique(system.tetrahedra[bad_tetrahedra])
        ]
        for vertex_index in inner_vertices:
            faces.update(
                int(face_index)
                for face_index in system.inner_vertex_incident_faces[
                    int(vertex_index)
                ]
            )
    return np.asarray(sorted(faces), dtype=np.int64), bad_tetrahedra


def _deformation_quality(
    system: _InstanceOptimizationSystem,
    vertices: np.ndarray,
) -> _QualityDiagnostics:
    points = np.asarray(vertices, dtype=np.float64)[system.tetrahedra]
    matrices = np.stack(
        (
            points[:, 1] - points[:, 0],
            points[:, 2] - points[:, 0],
            points[:, 3] - points[:, 0],
        ),
        axis=2,
    )
    deformation = matrices @ system.canonical_inverse_matrices
    determinants = np.linalg.det(deformation)
    singular_values = np.linalg.svd(deformation, compute_uv=False)
    condition = singular_values[:, 0] / singular_values[:, 2]
    return _QualityDiagnostics(determinants, singular_values, condition)


def _selected_deformation_quality(
    system: _InstanceOptimizationSystem,
    vertices: np.ndarray,
    tetrahedron_indices: np.ndarray,
) -> _QualityDiagnostics:
    selected = np.asarray(tetrahedron_indices, dtype=np.int64)
    tetrahedra = system.tetrahedra[selected]
    points = np.asarray(vertices, dtype=np.float64)[tetrahedra]
    matrices = np.stack(
        (
            points[:, 1] - points[:, 0],
            points[:, 2] - points[:, 0],
            points[:, 3] - points[:, 0],
        ),
        axis=2,
    )
    deformation = matrices @ system.canonical_inverse_matrices[selected]
    determinants = np.linalg.det(deformation)
    singular_values = np.linalg.svd(deformation, compute_uv=False)
    return _QualityDiagnostics(
        determinants,
        singular_values,
        singular_values[:, 0] / singular_values[:, 2],
    )


def _quality_is_robust(quality: _QualityDiagnostics) -> bool:
    return bool(
        np.isfinite(quality.determinants).all()
        and np.isfinite(quality.singular_values).all()
        and float(np.min(quality.determinants))
        >= B3_FINAL_MINIMUM_DETERMINANT
        and float(np.min(quality.singular_values[:, 2]))
        >= B3_FINAL_MINIMUM_SINGULAR_VALUE
        and float(np.max(quality.singular_values[:, 0]))
        <= B3_FINAL_MAXIMUM_SINGULAR_VALUE
        and float(np.max(quality.condition_numbers))
        <= B3_FINAL_MAXIMUM_CONDITION_NUMBER
    )


def _lower_barrier(
    values: np.ndarray,
    floor: float,
    activation: float,
) -> tuple[float, np.ndarray] | None:
    """Return a C1 barrier and derivative for values that must stay above floor."""

    array = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(array)) or np.any(array <= floor):
        return None
    derivative = np.zeros_like(array)
    active = array < activation
    if not np.any(active):
        return 0.0, derivative
    span = activation - floor
    y = (array[active] - floor) / span
    one_minus = 1.0 - y
    terms = -(one_minus**2) * np.log(y)
    derivative[active] = (
        2.0 * one_minus * np.log(y) - one_minus**2 / y
    ) / span
    return float(np.sum(terms) / len(array)), derivative / len(array)


def _upper_barrier(
    values: np.ndarray,
    activation: float,
    ceiling: float,
) -> tuple[float, np.ndarray] | None:
    """Return a C1 barrier and derivative for values below a ceiling."""

    array = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(array)) or np.any(array >= ceiling):
        return None
    derivative = np.zeros_like(array)
    active = array > activation
    if not np.any(active):
        return 0.0, derivative
    span = ceiling - activation
    y = (ceiling - array[active]) / span
    one_minus = 1.0 - y
    terms = -(one_minus**2) * np.log(y)
    derivative[active] = -(
        2.0 * one_minus * np.log(y) - one_minus**2 / y
    ) / span
    return float(np.sum(terms) / len(array)), derivative / len(array)


def _point_triangle_closest(
    point: np.ndarray,
    triangle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the closest point and barycentric weights on one triangle."""

    a, b, c = triangle
    ab = b - a
    ac = c - a
    ap = point - a
    d1 = float(np.dot(ab, ap))
    d2 = float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        return a, np.asarray((1.0, 0.0, 0.0))
    bp = point - b
    d3 = float(np.dot(ab, bp))
    d4 = float(np.dot(ac, bp))
    if d3 >= 0.0 and d4 <= d3:
        return b, np.asarray((0.0, 1.0, 0.0))
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return a + v * ab, np.asarray((1.0 - v, v, 0.0))
    cp = point - c
    d5 = float(np.dot(ab, cp))
    d6 = float(np.dot(ac, cp))
    if d6 >= 0.0 and d5 <= d6:
        return c, np.asarray((0.0, 0.0, 1.0))
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return a + w * ac, np.asarray((1.0 - w, 0.0, w))
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        return b + w * (c - b), np.asarray((0.0, 1.0 - w, w))
    denominator = 1.0 / (va + vb + vc)
    v = vb * denominator
    w = vc * denominator
    return a + ab * v + ac * w, np.asarray((1.0 - v - w, v, w))


def _segment_segment_closest(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return closest points and segment parameters for two finite segments."""

    p0, p1 = first
    q0, q1 = second
    u = p1 - p0
    v = q1 - q0
    w = p0 - q0
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denominator = a * c - b * b
    tolerance = np.finfo(np.float64).eps * max(a * c, 1.0)
    s = 0.0 if denominator <= tolerance else np.clip((b * e - c * d) / denominator, 0.0, 1.0)
    t = np.clip((b * s + e) / c, 0.0, 1.0) if c > tolerance else 0.0
    s = np.clip((b * t - d) / a, 0.0, 1.0) if a > tolerance else 0.0
    t = np.clip((b * s + e) / c, 0.0, 1.0) if c > tolerance else 0.0
    return p0 + s * u, q0 + t * v, float(s), float(t)


def _triangle_triangle_closest_features(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return distance, barycentric weights, and a deterministic separating normal."""

    candidates: list[tuple[float, int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for index in range(3):
        closest, weights = _point_triangle_closest(first[index], second)
        first_weights = np.zeros(3, dtype=np.float64)
        first_weights[index] = 1.0
        delta = first[index] - closest
        candidates.append((float(np.dot(delta, delta)), index, first_weights, weights, first[index], closest))
    for index in range(3):
        closest, weights = _point_triangle_closest(second[index], first)
        second_weights = np.zeros(3, dtype=np.float64)
        second_weights[index] = 1.0
        delta = closest - second[index]
        candidates.append((float(np.dot(delta, delta)), 3 + index, weights, second_weights, closest, second[index]))
    edges = ((0, 1), (1, 2), (2, 0))
    candidate_index = 6
    for first_edge in edges:
        for second_edge in edges:
            p, q, s, t = _segment_segment_closest(
                first[np.asarray(first_edge)], second[np.asarray(second_edge)]
            )
            first_weights = np.zeros(3, dtype=np.float64)
            second_weights = np.zeros(3, dtype=np.float64)
            first_weights[list(first_edge)] = (1.0 - s, s)
            second_weights[list(second_edge)] = (1.0 - t, t)
            delta = p - q
            candidates.append((float(np.dot(delta, delta)), candidate_index, first_weights, second_weights, p, q))
            candidate_index += 1
    squared, _, first_weights, second_weights, p, q = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    distance = math.sqrt(max(squared, 0.0))
    if distance > 1.0e-14:
        normal = (p - q) / distance
    else:
        first_normal = np.cross(first[1] - first[0], first[2] - first[0])
        second_normal = np.cross(second[1] - second[0], second[2] - second[0])
        normal = first_normal - second_normal
        norm = float(np.linalg.norm(normal))
        if norm <= 1.0e-14:
            normal = first_normal
            norm = float(np.linalg.norm(normal))
        if norm <= 1.0e-14:
            normal = np.asarray((1.0, 0.0, 0.0))
        else:
            normal /= norm
    return distance, first_weights, second_weights, normal


def _near_face_pairs(
    vertices: np.ndarray,
    first_faces: np.ndarray,
    second_faces: np.ndarray,
    threshold: float,
    *,
    self_pairs: bool,
    forced_pairs: np.ndarray | None = None,
    excluded_pairs: frozenset[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Find a conservative deterministic set of nearby triangle pairs."""

    first_triangles = vertices[first_faces]
    second_triangles = vertices[second_faces]
    first_minimum = first_triangles.min(axis=1)
    first_maximum = first_triangles.max(axis=1)
    second_minimum = second_triangles.min(axis=1)
    second_maximum = second_triangles.max(axis=1)
    first_centers = 0.5 * (first_minimum + first_maximum)
    second_centers = 0.5 * (second_minimum + second_maximum)
    first_radii = np.linalg.norm(first_maximum - first_minimum, axis=1) * 0.5
    second_radii = np.linalg.norm(second_maximum - second_minimum, axis=1) * 0.5
    maximum_second_radius = float(np.max(second_radii))
    tree = cKDTree(second_centers)
    pairs: set[tuple[int, int]] = set()
    for first_index, center in enumerate(first_centers):
        radius = float(first_radii[first_index]) + maximum_second_radius + threshold
        for second_index in tree.query_ball_point(center, radius):
            second_index = int(second_index)
            if self_pairs and first_index >= second_index:
                continue
            if (
                self_pairs
                and excluded_pairs is not None
                and (first_index, second_index) in excluded_pairs
            ):
                continue
            if (
                self_pairs
                and excluded_pairs is None
                and np.intersect1d(
                    first_faces[first_index], second_faces[second_index]
                ).size
            ):
                continue
            separation = np.maximum(
                np.maximum(
                    second_minimum[second_index] - first_maximum[first_index],
                    first_minimum[first_index] - second_maximum[second_index],
                ),
                0.0,
            )
            if float(np.linalg.norm(separation)) <= threshold:
                pairs.add((first_index, second_index))
    if forced_pairs is not None:
        for first_index, second_index in np.asarray(forced_pairs, dtype=np.int64):
            if first_index != second_index:
                pairs.add((int(first_index), int(second_index)))
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(sorted(pairs), dtype=np.int64)


def _build_collision_constraints(
    system: _InstanceOptimizationSystem,
    vertices: np.ndarray,
    surface_resolution: float,
    forced_inner_pairs: np.ndarray,
    repair_region: _SurfaceRepairRegion | None = None,
) -> _CollisionConstraints:
    activation = B3_COLLISION_ACTIVATION_RESOLUTIONS * surface_resolution
    self_pairs = _near_face_pairs(
        vertices,
        system.inner_global_faces,
        system.inner_global_faces,
        activation,
        self_pairs=True,
        forced_pairs=forced_inner_pairs,
        excluded_pairs=system.inner_face_adjacency,
    )
    outer_pairs = _near_face_pairs(
        vertices,
        system.inner_global_faces,
        system.outer_faces,
        activation,
        self_pairs=False,
    )
    if repair_region is not None and not repair_region.full_surface:
        active_faces = np.zeros(len(system.inner_faces), dtype=bool)
        active_faces[repair_region.active_face_indices] = True
        if len(self_pairs):
            self_pairs = self_pairs[
                active_faces[self_pairs[:, 0]] | active_faces[self_pairs[:, 1]]
            ]
        if len(outer_pairs):
            outer_pairs = outer_pairs[active_faces[outer_pairs[:, 0]]]
    first_indices: list[np.ndarray] = []
    first_weights: list[np.ndarray] = []
    second_indices: list[np.ndarray] = []
    second_weights: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    pair_sets = (
        (self_pairs, system.inner_global_faces, system.inner_global_faces),
        (outer_pairs, system.inner_global_faces, system.outer_faces),
    )
    for pairs, first_faces, second_faces in pair_sets:
        for first_face, second_face in pairs:
            first_ids = first_faces[first_face]
            second_ids = second_faces[second_face]
            distance, first_barycentric, second_barycentric, normal = (
                _triangle_triangle_closest_features(
                    vertices[first_ids], vertices[second_ids]
                )
            )
            if distance <= activation or pairs is self_pairs:
                first_indices.append(first_ids)
                first_weights.append(first_barycentric)
                second_indices.append(second_ids)
                second_weights.append(second_barycentric)
                normals.append(normal)
    if not normals:
        return _CollisionConstraints(
            np.empty((0, 3), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.int64),
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 3), dtype=np.float64),
        )
    return _CollisionConstraints(
        np.asarray(first_indices, dtype=np.int64),
        np.asarray(first_weights, dtype=np.float64),
        np.asarray(second_indices, dtype=np.int64),
        np.asarray(second_weights, dtype=np.float64),
        np.asarray(normals, dtype=np.float64),
    )


def _tetrahedral_energy_gradient(
    system: _InstanceOptimizationSystem,
    vertices: np.ndarray,
    barrier_multiplier: float,
    tetrahedron_indices: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    """Return distortion/barrier energies and the full vertex gradient."""

    selected = (
        np.arange(len(system.tetrahedra), dtype=np.int64)
        if tetrahedron_indices is None
        else np.asarray(tetrahedron_indices, dtype=np.int64)
    )
    tetrahedra = system.tetrahedra[selected]
    points = vertices[tetrahedra]
    matrices = np.stack(
        (
            points[:, 1] - points[:, 0],
            points[:, 2] - points[:, 0],
            points[:, 3] - points[:, 0],
        ),
        axis=2,
    )
    deformation = matrices @ system.canonical_inverse_matrices[selected]
    determinants = np.linalg.det(deformation)
    determinant_barrier = _lower_barrier(
        determinants,
        INSTANCE_JACOBIAN_DETERMINANT_FLOOR,
        B3_DETERMINANT_ACTIVATION,
    )
    if determinant_barrier is None:
        return None
    barrier_energy, barrier_derivative = determinant_barrier

    inverse_transpose = np.swapaxes(np.linalg.inv(deformation), 1, 2)
    cofactors = determinants[:, None, None] * inverse_transpose
    squared_norm = np.sum(deformation * deformation, axis=(1, 2))
    determinant_two_thirds = determinants ** (2.0 / 3.0)
    log_determinant = np.log(determinants)
    distortion_terms = (
        squared_norm / (3.0 * determinant_two_thirds)
        - 1.0
        + 0.1 * log_determinant**2
    )
    weights = system.canonical_volume_weights[selected]
    weights = weights / np.sum(weights)
    distortion_energy = float(np.sum(weights * distortion_terms))
    distortion_deformation_gradient = weights[:, None, None] * (
        2.0 * deformation / (3.0 * determinant_two_thirds[:, None, None])
        - (
            2.0
            * squared_norm
            / (9.0 * determinants ** (5.0 / 3.0))
        )[:, None, None]
        * cofactors
        + (0.2 * log_determinant / determinants)[:, None, None] * cofactors
    )
    barrier_deformation_gradient = (
        barrier_multiplier * barrier_derivative[:, None, None] * cofactors
    )
    distortion_matrix_gradient = (
        distortion_deformation_gradient
        @ np.swapaxes(system.canonical_inverse_matrices[selected], 1, 2)
    )
    barrier_matrix_gradient = (
        barrier_deformation_gradient
        @ np.swapaxes(system.canonical_inverse_matrices[selected], 1, 2)
    )
    distortion_gradient = np.zeros_like(vertices)
    barrier_gradient = np.zeros_like(vertices)
    np.add.at(
        distortion_gradient, tetrahedra[:, 1], distortion_matrix_gradient[:, :, 0]
    )
    np.add.at(
        distortion_gradient, tetrahedra[:, 2], distortion_matrix_gradient[:, :, 1]
    )
    np.add.at(
        distortion_gradient, tetrahedra[:, 3], distortion_matrix_gradient[:, :, 2]
    )
    np.add.at(
        distortion_gradient,
        tetrahedra[:, 0],
        -np.sum(distortion_matrix_gradient, axis=2),
    )
    np.add.at(
        barrier_gradient, tetrahedra[:, 1], barrier_matrix_gradient[:, :, 0]
    )
    np.add.at(
        barrier_gradient, tetrahedra[:, 2], barrier_matrix_gradient[:, :, 1]
    )
    np.add.at(
        barrier_gradient, tetrahedra[:, 3], barrier_matrix_gradient[:, :, 2]
    )
    np.add.at(
        barrier_gradient,
        tetrahedra[:, 0],
        -np.sum(barrier_matrix_gradient, axis=2),
    )
    return (
        distortion_energy,
        barrier_energy,
        distortion_gradient,
        barrier_gradient,
    )


def _evaluate_objective(
    flat_movable: np.ndarray,
    *,
    system: _InstanceOptimizationSystem,
    fixed_vertices: np.ndarray,
    target_inner: np.ndarray,
    target_edges: np.ndarray,
    baseline: np.ndarray,
    surface_resolution: float,
    collision_constraints: _CollisionConstraints,
    barrier_multiplier: float,
) -> tuple[float, np.ndarray]:
    """Evaluate the normalized B3 objective and its analytical gradient."""

    vertices = np.asarray(fixed_vertices, dtype=np.float64).copy()
    movable = np.asarray(flat_movable, dtype=np.float64).reshape(-1, 3)
    vertices[system.movable_vertex_indices] = movable
    inner = system.inner_vertex_indices
    gradient = np.zeros_like(vertices)
    terms: dict[str, float] = {}

    correction = vertices[inner] - target_inner
    correction_squared = np.sum(correction * correction, axis=1)
    cap = B3_MAXIMUM_CORRECTION_RESOLUTIONS * surface_resolution
    cap_barrier = _upper_barrier(
        correction_squared,
        0.25 * cap * cap,
        cap * cap,
    )
    if cap_barrier is None:
        return math.inf, np.zeros_like(flat_movable)
    cap_energy, cap_derivative = cap_barrier
    terms["target"] = float(
        np.sum(system.surface_area_weights * correction_squared)
        / surface_resolution**2
    )
    gradient[inner] += (
        2.0
        * B3_TARGET_WEIGHT
        * system.surface_area_weights[:, None]
        * correction
        / surface_resolution**2
    )
    gradient[inner] += (
        barrier_multiplier * 2.0 * cap_derivative[:, None] * correction
    )

    edges = system.surface_edges
    edge_delta = (
        vertices[edges[:, 1]] - vertices[edges[:, 0]] - target_edges
    )
    terms["surface"] = float(
        np.mean(np.sum(edge_delta * edge_delta, axis=1))
        / surface_resolution**2
    )
    edge_gradient = (
        2.0
        * B3_SURFACE_WEIGHT
        * edge_delta
        / (len(edges) * surface_resolution**2)
    )
    np.add.at(gradient, edges[:, 1], edge_gradient)
    np.add.at(gradient, edges[:, 0], -edge_gradient)

    baseline_delta = movable - baseline[system.movable_vertex_indices]
    stiffness_delta = system.stiffness @ baseline_delta
    terms["fem"] = float(
        np.sum(baseline_delta * stiffness_delta)
        / system.canonical_shell_volume
    )
    gradient[system.movable_vertex_indices] += (
        2.0
        * B3_FEM_WEIGHT
        * stiffness_delta
        / system.canonical_shell_volume
    )

    tetrahedral = _tetrahedral_energy_gradient(
        system, vertices, barrier_multiplier
    )
    if tetrahedral is None:
        return math.inf, np.zeros_like(flat_movable)
    (
        distortion_energy,
        determinant_energy,
        distortion_gradient,
        determinant_gradient,
    ) = tetrahedral
    terms["distortion"] = distortion_energy
    terms["determinant_barrier"] = determinant_energy
    gradient += (
        B3_DISTORTION_WEIGHT * distortion_gradient + determinant_gradient
    )

    normalized = np.abs((vertices[inner] - ENVELOPE_CENTER) / ENVELOPE_RADII)
    envelope_values = np.sum(normalized**ENVELOPE_POWER, axis=1)
    envelope_barrier = _upper_barrier(
        envelope_values, 0.9, 1.0 - 1.0e-10
    )
    if envelope_barrier is None:
        return math.inf, np.zeros_like(flat_movable)
    envelope_energy, envelope_derivative = envelope_barrier
    envelope_gradient = (
        ENVELOPE_POWER
        * (vertices[inner] - ENVELOPE_CENTER) ** 3
        / ENVELOPE_RADII**4
    )
    gradient[inner] += (
        barrier_multiplier
        * envelope_derivative[:, None]
        * envelope_gradient
    )

    collision_energy = 0.0
    if collision_constraints.count:
        first_points = np.einsum(
            "ni,nij->nj",
            collision_constraints.first_weights,
            vertices[collision_constraints.first_vertex_indices],
        )
        second_points = np.einsum(
            "ni,nij->nj",
            collision_constraints.second_weights,
            vertices[collision_constraints.second_vertex_indices],
        )
        gaps = np.einsum(
            "ni,ni->n",
            collision_constraints.normals,
            first_points - second_points,
        )
        collision_barrier = _lower_barrier(
            gaps,
            B3_COLLISION_MINIMUM_RESOLUTIONS * surface_resolution,
            B3_COLLISION_ACTIVATION_RESOLUTIONS * surface_resolution,
        )
        if collision_barrier is None:
            return math.inf, np.zeros_like(flat_movable)
        collision_energy, collision_derivative = collision_barrier
        pair_gradient = (
            barrier_multiplier
            * collision_derivative[:, None]
            * collision_constraints.normals
        )
        for corner in range(3):
            np.add.at(
                gradient,
                collision_constraints.first_vertex_indices[:, corner],
                collision_constraints.first_weights[:, corner, None]
                * pair_gradient,
            )
            np.add.at(
                gradient,
                collision_constraints.second_vertex_indices[:, corner],
                -collision_constraints.second_weights[:, corner, None]
                * pair_gradient,
            )

    terms["correction_barrier"] = cap_energy
    terms["envelope_barrier"] = envelope_energy
    terms["collision_barrier"] = collision_energy
    total = (
        B3_TARGET_WEIGHT * terms["target"]
        + B3_SURFACE_WEIGHT * terms["surface"]
        + B3_FEM_WEIGHT * terms["fem"]
        + B3_DISTORTION_WEIGHT * terms["distortion"]
        + barrier_multiplier
        * (
            terms["determinant_barrier"]
            + terms["correction_barrier"]
            + terms["envelope_barrier"]
            + terms["collision_barrier"]
        )
    )
    movable_gradient = gradient[system.movable_vertex_indices].reshape(-1)
    if not np.isfinite(total) or not np.isfinite(movable_gradient).all():
        return math.inf, np.zeros_like(flat_movable)
    return float(total), movable_gradient


def _interpolated_baseline(
    volume: CanonicalAnatomicalVolume,
    full_displacement: np.ndarray,
    target: np.ndarray,
    beta: float,
) -> np.ndarray:
    vertices = volume.volume_vertices + beta * full_displacement
    inner = volume.computational_inner_vertex_indices
    vertices[inner] = (
        (1.0 - beta) * volume.volume_vertices[inner] + beta * target
    )
    vertices[volume.outer_vertex_indices] = volume.volume_vertices[
        volume.outer_vertex_indices
    ]
    return vertices


def _final_validation(
    problem: InstanceVolumeProblem,
    system: _InstanceOptimizationSystem,
    vertices: np.ndarray,
    *,
    beta: float,
) -> _FinalValidation:
    volume = problem.canonical_volume
    target = problem.boundary_target.vertices
    inner = system.inner_vertex_indices
    validation, _ = _validate_continuation_candidate(volume, vertices)
    quality = _deformation_quality(system, vertices)
    correction = vertices[inner] - target
    correction_magnitudes = np.linalg.norm(correction, axis=1)
    resolution = problem.boundary_target.surface_resolution

    target_triangles = target[system.inner_faces]
    final_triangles = vertices[system.inner_global_faces]
    target_areas = 0.5 * np.linalg.norm(
        np.cross(
            target_triangles[:, 1] - target_triangles[:, 0],
            target_triangles[:, 2] - target_triangles[:, 0],
        ),
        axis=1,
    )
    final_areas = 0.5 * np.linalg.norm(
        np.cross(
            final_triangles[:, 1] - final_triangles[:, 0],
            final_triangles[:, 2] - final_triangles[:, 0],
        ),
        axis=1,
    )
    area_ratios = final_areas / target_areas

    reverse_vertices = np.einsum(
        "ni,nij->nj",
        volume.canonical_to_computational_barycentric,
        vertices[inner][
            system.inner_faces[volume.canonical_to_computational_face_indices]
        ],
    )
    reverse_distances = np.linalg.norm(
        reverse_vertices - problem.fitted_surface.vertices, axis=1
    )
    repair_zone = np.zeros(len(reverse_distances), dtype=bool)
    repair_zone[volume.repair_zone_canonical_vertex_indices] = True
    preserved = reverse_distances[~repair_zone]
    repaired = reverse_distances[repair_zone]
    fidelity_ok = bool(
        float(np.percentile(preserved, 99.0)) <= 0.5 * resolution
        and float(np.max(preserved)) <= resolution
        and float(np.percentile(repaired, 99.0)) <= 2.0 * resolution
        and float(np.max(repaired)) <= 2.0 * resolution
    )

    failure: str | None = None
    if beta != 1.0:
        failure = "target_progress_incomplete"
    elif not validation.accepted:
        failure = str(validation.failure)
    elif not _quality_is_robust(quality):
        failure = "bounded_distortion"
    elif float(np.max(correction_magnitudes)) > (
        B3_MAXIMUM_CORRECTION_RESOLUTIONS * resolution
    ):
        failure = "target_correction_limit"
    elif (
        float(np.min(area_ratios)) < B3_MINIMUM_TRIANGLE_AREA_RATIO
        or float(np.max(area_ratios)) > B3_MAXIMUM_TRIANGLE_AREA_RATIO
    ):
        failure = "surface_triangle_area_ratio"
    elif not fidelity_ok:
        failure = "authoritative_anatomy_fidelity"

    diagnostics = {
        "surface_resolution": float(resolution),
        "target_correction_magnitude": _summary(correction_magnitudes),
        "corrected_triangle_area_ratio": _summary(area_ratios),
        "reverse_fitted_anatomy_distance": _summary(reverse_distances),
        "preserved_reverse_distance": _summary(preserved),
        "repair_zone_reverse_distance": _summary(repaired),
        "authoritative_anatomy_fidelity_passed": fidelity_ok,
        "outer_boundary_exact": bool(
            np.array_equal(
                vertices[system.outer_vertex_indices],
                volume.volume_vertices[system.outer_vertex_indices],
            )
        ),
        "inner_self_intersection_count": int(
            validation.inner_self_intersection_count or 0
        ),
        "inner_outer_intersection_count": int(
            validation.inner_outer_intersection_count or 0
        ),
        **quality.summary(),
    }
    return _FinalValidation(
        accepted=failure is None,
        failure=failure,
        quality=quality,
        target_correction_vectors=correction,
        diagnostics=diagnostics,
    )


def _full_vertices_with_inner(
    system: _InstanceOptimizationSystem,
    canonical_vertices: np.ndarray,
    inner_vertices: np.ndarray,
) -> np.ndarray:
    vertices = np.asarray(canonical_vertices, dtype=np.float64).copy()
    vertices[system.inner_vertex_indices] = inner_vertices
    return vertices


def _validate_computational_boundary_cheap(
    system: _InstanceOptimizationSystem,
    inner_vertices: np.ndarray,
) -> _BoundaryValidation:
    """Apply inexpensive surface checks before exact intersection queries."""

    inner = np.asarray(inner_vertices, dtype=np.float64)
    if inner.shape != (len(system.inner_vertex_indices), 3) or not np.isfinite(
        inner
    ).all():
        return _BoundaryValidation(False, "non_finite_vertices", 0, 0, 0, 0)
    triangles = inner[system.inner_faces]
    twice_area = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    scale = max(float(np.ptp(inner, axis=0).max()), 1.0)
    area_tolerance = 128.0 * np.finfo(np.float64).eps * scale**2
    degenerate = int(np.count_nonzero(twice_area <= area_tolerance))
    if degenerate:
        return _BoundaryValidation(
            False, "degenerate_inner_face", 0, 0, 0, degenerate
        )

    envelope = np.sum(
        np.abs((inner - ENVELOPE_CENTER) / ENVELOPE_RADII) ** ENVELOPE_POWER,
        axis=1,
    )
    outside = int(np.count_nonzero(envelope >= 1.0 - 1.0e-10))
    if outside:
        return _BoundaryValidation(
            False, "outside_frozen_envelope", 0, 0, outside, 0
        )
    return _BoundaryValidation(True, None, 0, 0, 0, 0)


def _validate_computational_boundary(
    system: _InstanceOptimizationSystem,
    canonical_vertices: np.ndarray,
    inner_vertices: np.ndarray,
    *,
    timing: Callable[[str, float], None] | None = None,
) -> _BoundaryValidation:
    """Apply cheap checks followed by authoritative exact intersections."""

    cheap = _validate_computational_boundary_cheap(system, inner_vertices)
    if not cheap.accepted:
        return cheap
    inner = np.asarray(inner_vertices, dtype=np.float64)
    triangles = inner[system.inner_faces]

    started = time.perf_counter()
    self_pairs = _self_intersection_pairs(inner, system.inner_faces)
    if timing is not None:
        timing("exact_inner_self_intersections", time.perf_counter() - started)
    if len(self_pairs):
        return _BoundaryValidation(
            False,
            "inner_self_intersection",
            int(len(self_pairs)),
            0,
            0,
            0,
            self_intersection_pairs=self_pairs,
        )

    complete = _full_vertices_with_inner(system, canonical_vertices, inner)
    outer_triangles = complete[system.outer_faces]
    tolerance = 512.0 * np.finfo(np.float64).eps * max(
        float(np.max(np.abs(complete))), 1.0
    )
    started = time.perf_counter()
    inner_outer_pairs = _find_collision_pairs(
        triangles,
        np.arange(len(system.inner_faces), dtype=np.int64),
        outer_triangles,
        np.arange(len(system.outer_faces), dtype=np.int64),
        tolerance,
        obstacle_broad_phase=system.outer_collision_index,
    )
    if timing is not None:
        timing("exact_inner_outer_intersections", time.perf_counter() - started)
    if len(inner_outer_pairs):
        return _BoundaryValidation(
            False,
            "inner_outer_intersection",
            0,
            int(len(inner_outer_pairs)),
            0,
            0,
            inner_outer_intersection_pairs=inner_outer_pairs,
        )
    return _BoundaryValidation(True, None, 0, 0, 0, 0)


def _evaluate_surface_objective(
    flat_active: np.ndarray,
    *,
    system: _InstanceOptimizationSystem,
    template_vertices: np.ndarray,
    repair_region: _SurfaceRepairRegion,
    target_inner: np.ndarray,
    target_edges: np.ndarray,
    surface_resolution: float,
    collision_constraints: _CollisionConstraints,
    barrier_multiplier: float,
) -> tuple[float, np.ndarray]:
    """Optimize the active topology-preserving computational-surface patch."""

    vertices = np.asarray(template_vertices, dtype=np.float64).copy()
    active = repair_region.active_inner_vertex_indices
    inner_vertices = vertices[system.inner_vertex_indices].copy()
    inner_vertices[active] = np.asarray(flat_active, dtype=np.float64).reshape(-1, 3)
    vertices[system.inner_vertex_indices] = inner_vertices
    gradient = np.zeros_like(vertices)
    correction = inner_vertices[active] - target_inner[active]
    correction_squared = np.sum(correction * correction, axis=1)
    cap = B3_MAXIMUM_CORRECTION_RESOLUTIONS * surface_resolution
    cap_barrier = _upper_barrier(
        correction_squared, 0.25 * cap * cap, cap * cap
    )
    if cap_barrier is None:
        return math.inf, np.zeros_like(flat_active)
    cap_energy, cap_derivative = cap_barrier
    target_energy = float(
        np.sum(system.surface_area_weights[active] * correction_squared)
        / surface_resolution**2
    )
    gradient[system.inner_vertex_indices[active]] += (
        2.0
        * B3_TARGET_WEIGHT
        * system.surface_area_weights[active, None]
        * correction
        / surface_resolution**2
        + barrier_multiplier * 2.0 * cap_derivative[:, None] * correction
    )

    edge_indices = repair_region.active_edge_indices
    edges = system.surface_edges[edge_indices]
    edge_delta = (
        vertices[edges[:, 1]]
        - vertices[edges[:, 0]]
        - target_edges[edge_indices]
    )
    surface_energy = float(
        np.sum(np.sum(edge_delta * edge_delta, axis=1))
        / len(system.surface_edges)
        / surface_resolution**2
    )
    edge_gradient = (
        2.0
        * B3_SURFACE_WEIGHT
        * edge_delta
        / (len(system.surface_edges) * surface_resolution**2)
    )
    np.add.at(gradient, edges[:, 1], edge_gradient)
    np.add.at(gradient, edges[:, 0], -edge_gradient)

    active_vertices = inner_vertices[active]
    normalized = np.abs((active_vertices - ENVELOPE_CENTER) / ENVELOPE_RADII)
    envelope_values = np.sum(normalized**ENVELOPE_POWER, axis=1)
    envelope_barrier = _upper_barrier(
        envelope_values, 0.9, 1.0 - 1.0e-10
    )
    if envelope_barrier is None:
        return math.inf, np.zeros_like(flat_active)
    envelope_energy, envelope_derivative = envelope_barrier
    envelope_gradient = (
        ENVELOPE_POWER
        * (active_vertices - ENVELOPE_CENTER) ** 3
        / ENVELOPE_RADII**4
    )
    gradient[system.inner_vertex_indices[active]] += (
        barrier_multiplier
        * envelope_derivative[:, None]
        * envelope_gradient
    )

    collision_energy = 0.0
    if collision_constraints.count:
        first_points = np.einsum(
            "ni,nij->nj",
            collision_constraints.first_weights,
            vertices[collision_constraints.first_vertex_indices],
        )
        second_points = np.einsum(
            "ni,nij->nj",
            collision_constraints.second_weights,
            vertices[collision_constraints.second_vertex_indices],
        )
        gaps = np.einsum(
            "ni,ni->n",
            collision_constraints.normals,
            first_points - second_points,
        )
        collision_barrier = _lower_barrier(
            gaps,
            B3_COLLISION_MINIMUM_RESOLUTIONS * surface_resolution,
            B3_COLLISION_ACTIVATION_RESOLUTIONS * surface_resolution,
        )
        if collision_barrier is None:
            return math.inf, np.zeros_like(flat_active)
        collision_energy, collision_derivative = collision_barrier
        pair_gradient = collision_derivative[:, None] * collision_constraints.normals
        for corner in range(3):
            np.add.at(
                gradient,
                collision_constraints.first_vertex_indices[:, corner],
                barrier_multiplier
                * collision_constraints.first_weights[:, corner, None]
                * pair_gradient,
            )
            np.add.at(
                gradient,
                collision_constraints.second_vertex_indices[:, corner],
                -barrier_multiplier
                * collision_constraints.second_weights[:, corner, None]
                * pair_gradient,
            )

    if len(repair_region.affected_boundary_tetrahedron_indices):
        boundary_tetrahedral = _tetrahedral_energy_gradient(
            system,
            vertices,
            barrier_multiplier,
            repair_region.affected_boundary_tetrahedron_indices,
        )
        if boundary_tetrahedral is None:
            return math.inf, np.zeros_like(flat_active)
        (
            boundary_distortion_energy,
            boundary_determinant_energy,
            boundary_distortion_gradient,
            boundary_determinant_gradient,
        ) = boundary_tetrahedral
        gradient += (
            B3_DISTORTION_WEIGHT * boundary_distortion_gradient
            + boundary_determinant_gradient
        )
    else:
        boundary_distortion_energy = 0.0
        boundary_determinant_energy = 0.0

    total = (
        B3_TARGET_WEIGHT * target_energy
        + B3_SURFACE_WEIGHT * surface_energy
        + B3_DISTORTION_WEIGHT * boundary_distortion_energy
        + barrier_multiplier
        * (
            cap_energy
            + envelope_energy
            + collision_energy
            + boundary_determinant_energy
        )
    )
    inner_gradient = gradient[system.inner_vertex_indices[active]].reshape(-1)
    if not np.isfinite(total) or not np.isfinite(inner_gradient).all():
        return math.inf, np.zeros_like(flat_active)
    return float(total), inner_gradient


def _untangle_computational_boundary(
    problem: InstanceVolumeProblem,
    continuation: InstanceVolumeContinuation,
    system: _InstanceOptimizationSystem,
    progress: Callable[[dict[str, Any]], None] | None,
    timing: Callable[[str, float], None] | None,
) -> _SurfaceUntangling:
    """Create a collision-free target before deforming tetrahedral interiors."""

    volume = problem.canonical_volume
    canonical = volume.volume_vertices[system.inner_vertex_indices]
    target = problem.boundary_target.vertices
    resolution = problem.boundary_target.surface_resolution
    region = _initial_surface_repair_region(problem, system)
    initial_active_count = int(len(region.active_inner_vertex_indices))
    exact_target_validation = _validate_computational_boundary(
        system, volume.volume_vertices, target, timing=timing
    )
    exact_target_complete = _full_vertices_with_inner(
        system, volume.volume_vertices, target
    )
    exact_boundary_quality = _selected_deformation_quality(
        system,
        exact_target_complete,
        system.boundary_tetrahedron_indices,
    )
    if exact_target_validation.accepted and _quality_is_robust(
        exact_boundary_quality
    ):
        return _SurfaceUntangling(
            target.copy(),
            1.0,
            (),
            None,
            0,
            {
                "initial_active_inner_vertices": initial_active_count,
                "final_active_inner_vertices": initial_active_count,
                "initial_expansion_rings": region.expansion_rings,
                "final_expansion_rings": region.expansion_rings,
                "seed_intersection_faces": region.seed_intersection_face_count,
                "seed_bad_boundary_tetrahedra": (
                    region.seed_bad_tetrahedron_count
                ),
                "region_expansions": 0,
                "full_surface_fallback": False,
            },
        )

    if not len(region.active_inner_vertex_indices):
        unexpected_faces = []
        if exact_target_validation.self_intersection_pairs is not None:
            unexpected_faces.extend(
                np.unique(
                    exact_target_validation.self_intersection_pairs
                ).tolist()
            )
        if exact_target_validation.inner_outer_intersection_pairs is not None:
            unexpected_faces.extend(
                exact_target_validation.inner_outer_intersection_pairs[:, 0].tolist()
            )
        bad_faces, _ = _bad_boundary_face_indices(
            system, exact_target_complete
        )
        unexpected_faces.extend(bad_faces.tolist())
        region = _surface_repair_region(
            system,
            _expanded_surface_faces(
                system,
                np.asarray(sorted(set(unexpected_faces)), dtype=np.int64),
                B3_INITIAL_REPAIR_RINGS,
            ),
            expansion_rings=B3_INITIAL_REPAIR_RINGS,
            seed_intersection_face_count=len(set(unexpected_faces)),
            seed_bad_tetrahedron_count=0,
        )
    if not len(region.active_inner_vertex_indices):
        region = _surface_repair_region(
            system,
            np.arange(len(system.inner_faces), dtype=np.int64),
            expansion_rings=B3_MAXIMUM_REPAIR_RINGS + 1,
            seed_intersection_face_count=0,
            seed_bad_tetrahedron_count=0,
            full_surface=True,
        )

    upper = float(continuation.reached_alpha)
    lower = 0.0
    for _ in range(14):
        middle = 0.5 * (lower + upper)
        middle_inner = (1.0 - middle) * canonical + middle * target
        middle_complete = _full_vertices_with_inner(
            system, volume.volume_vertices, middle_inner
        )
        middle_quality = _selected_deformation_quality(
            system,
            middle_complete,
            system.boundary_tetrahedron_indices,
        )
        if _quality_is_robust(middle_quality):
            lower = middle
        else:
            upper = middle
    beta = lower
    current = (1.0 - beta) * canonical + beta * target
    seed_validation = _validate_computational_boundary(
        system, volume.volume_vertices, current, timing=timing
    )
    if not seed_validation.accepted:
        beta = 0.0
        current = canonical.copy()
    beta_step = B3_INITIAL_BETA_STEP
    history: list[dict[str, Any]] = []
    last_failure = "surface_seed"
    cap = B3_MAXIMUM_CORRECTION_RESOLUTIONS * resolution
    total_iterations = 0
    region_expansions = 0
    forced_pairs = {
        (int(first), int(second))
        for first, second in problem.boundary_target.intersecting_face_pairs
    }

    while beta < 1.0:
        trial_beta = min(1.0, beta + beta_step)
        previous_target = (1.0 - beta) * canonical + beta * target
        target_beta = (1.0 - trial_beta) * canonical + trial_beta * target
        predictor = current + (target_beta - previous_target)
        inactive = np.ones(len(system.inner_vertex_indices), dtype=bool)
        inactive[region.active_inner_vertex_indices] = False
        predictor[inactive] = target_beta[inactive]
        predictor_validation = _validate_computational_boundary_cheap(
            system, predictor
        )
        predictor_complete = _full_vertices_with_inner(
            system, volume.volume_vertices, predictor
        )
        predictor_quality = _selected_deformation_quality(
            system,
            predictor_complete,
            system.boundary_tetrahedron_indices,
        )
        predictor_usable = bool(
            predictor_validation.accepted
            and np.isfinite(predictor_quality.determinants).all()
            and float(np.min(predictor_quality.determinants))
            > INSTANCE_JACOBIAN_DETERMINANT_FLOOR
        )
        candidate = predictor if predictor_usable else current.copy()
        candidate[inactive] = target_beta[inactive]
        started_from_predictor = predictor_usable
        target_edges = (
            target_beta[system.surface_edge_inner_indices[:, 1]]
            - target_beta[system.surface_edge_inner_indices[:, 0]]
        )
        stage_records: list[dict[str, Any]] = []
        improved = False
        candidate_validation: _BoundaryValidation | None = None

        for block, multiplier in enumerate(B3_BARRIER_MULTIPLIERS):
            complete = _full_vertices_with_inner(
                system, volume.volume_vertices, candidate
            )
            constraints = _build_collision_constraints(
                system,
                complete,
                resolution,
                (
                    np.asarray(sorted(forced_pairs), dtype=np.int64)
                    if forced_pairs
                    else np.empty((0, 2), dtype=np.int64)
                ),
                region,
            )
            arguments = {
                "system": system,
                "template_vertices": complete,
                "repair_region": region,
                "target_inner": target_beta,
                "target_edges": target_edges,
                "surface_resolution": resolution,
                "collision_constraints": constraints,
                "barrier_multiplier": multiplier,
            }
            active = region.active_inner_vertex_indices
            initial_flat = candidate[active].reshape(-1)
            initial_value, _ = _evaluate_surface_objective(
                initial_flat, **arguments
            )
            if not np.isfinite(initial_value):
                last_failure = "infeasible_surface_step"
                stage_records.append(
                    {
                        "block": block,
                        "barrier_multiplier": multiplier,
                        "accepted": False,
                        "failure": last_failure,
                        "active_collision_constraints": constraints.count,
                    }
                )
                break

            def normalized_objective(
                offsets: np.ndarray,
            ) -> tuple[float, np.ndarray]:
                variable_scale = (
                    B3_SURFACE_VARIABLE_SCALE_RESOLUTIONS * resolution
                )
                value, coordinate_gradient = _evaluate_surface_objective(
                    initial_flat + variable_scale * offsets, **arguments
                )
                return value, variable_scale * coordinate_gradient

            optimized = minimize(
                normalized_objective,
                np.zeros_like(initial_flat),
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxiter": B3_ITERATIONS_PER_BLOCK,
                    "maxcor": 10,
                    "ftol": 1.0e-12,
                    "gtol": 1.0e-8,
                    "maxls": 30,
                },
            )
            total_iterations += int(optimized.nit)
            variable_scale = B3_SURFACE_VARIABLE_SCALE_RESOLUTIONS * resolution
            proposal = candidate.copy()
            proposal[active] = (
                initial_flat + variable_scale * optimized.x
            ).reshape(-1, 3)
            proposal[inactive] = target_beta[inactive]
            accepted_step = 0.0
            accepted_value = initial_value
            failure = "objective_not_improved"
            line_step = 1.0
            while line_step >= 2.0**-20:
                trial = candidate + line_step * (proposal - candidate)
                trial[inactive] = target_beta[inactive]
                trial_flat = trial[active].reshape(-1)
                trial_value, _ = _evaluate_surface_objective(
                    trial_flat, **arguments
                )
                tolerance = 1.0e-12 * max(1.0, abs(initial_value))
                correction = np.linalg.norm(trial - target_beta, axis=1)
                if (
                    np.isfinite(trial_value)
                    and trial_value < initial_value - tolerance
                    and float(np.max(correction)) < cap
                ):
                    cheap_validation = _validate_computational_boundary_cheap(
                        system, trial
                    )
                    trial_quality = _selected_deformation_quality(
                        system,
                        _full_vertices_with_inner(
                            system, volume.volume_vertices, trial
                        ),
                        system.boundary_tetrahedron_indices,
                    )
                    if (
                        cheap_validation.accepted
                        and np.isfinite(trial_quality.determinants).all()
                        and float(np.min(trial_quality.determinants))
                        > INSTANCE_JACOBIAN_DETERMINANT_FLOOR
                    ):
                        exact_validation = _validate_computational_boundary(
                            system,
                            volume.volume_vertices,
                            trial,
                            timing=timing,
                        )
                        if exact_validation.accepted:
                            candidate = trial
                            candidate_validation = exact_validation
                            accepted_step = line_step
                            accepted_value = trial_value
                            failure = None
                            improved = True
                            break
                        failure = str(exact_validation.failure)
                        candidate_validation = exact_validation
                    else:
                        failure = str(
                            cheap_validation.failure or "jacobian_determinant"
                        )
                line_step *= 0.5
            stage_records.append(
                {
                    "block": block,
                    "barrier_multiplier": multiplier,
                    "active_collision_constraints": constraints.count,
                    "optimizer_iterations": int(optimized.nit),
                    "optimizer_success": bool(optimized.success),
                    "optimizer_message": str(optimized.message),
                    "initial_objective": initial_value,
                    "optimizer_objective": float(optimized.fun),
                    "accepted_objective": accepted_value,
                    "accepted_line_step": accepted_step,
                    "accepted": bool(accepted_step),
                    "failure": failure,
                }
            )

        validation = candidate_validation or _validate_computational_boundary(
            system, volume.volume_vertices, candidate, timing=timing
        )
        maximum_correction = float(
            np.max(np.linalg.norm(candidate - target_beta, axis=1))
        )
        boundary_quality = _selected_deformation_quality(
            system,
            _full_vertices_with_inner(
                system, volume.volume_vertices, candidate
            ),
            system.boundary_tetrahedron_indices,
        )
        accepted = bool(
            validation.accepted
            and _quality_is_robust(boundary_quality)
            and maximum_correction <= cap
            and (started_from_predictor or improved)
        )
        record = {
            "phase": "surface_untangling",
            "from_beta": beta,
            "trial_beta": trial_beta,
            "beta_step": beta_step,
            "active_inner_vertices": int(
                len(region.active_inner_vertex_indices)
            ),
            "active_faces": int(len(region.active_face_indices)),
            "repair_expansion_rings": int(region.expansion_rings),
            "full_surface_fallback": bool(region.full_surface),
            "accepted": accepted,
            "failure": None if accepted else str(validation.failure or last_failure),
            "maximum_target_correction": maximum_correction,
            "self_intersection_count": validation.self_intersection_count,
            "inner_outer_intersection_count": (
                validation.inner_outer_intersection_count
            ),
            "minimum_boundary_tetrahedron_determinant": float(
                np.min(boundary_quality.determinants)
            ),
            "maximum_boundary_tetrahedron_condition_number": float(
                np.max(boundary_quality.condition_numbers)
            ),
            "blocks": stage_records,
        }
        history.append(record)
        if progress is not None:
            progress(record)
        if accepted:
            current = candidate
            beta = trial_beta
            continue
        last_failure = str(validation.failure or last_failure)
        additional_faces: set[int] = set()
        if validation.self_intersection_pairs is not None:
            forced_pairs.update(
                (int(first), int(second))
                for first, second in validation.self_intersection_pairs
            )
            additional_faces.update(
                int(index)
                for index in np.unique(validation.self_intersection_pairs)
            )
        if validation.inner_outer_intersection_pairs is not None:
            additional_faces.update(
                int(index)
                for index in validation.inner_outer_intersection_pairs[:, 0]
            )
        bad_faces, _ = _bad_boundary_face_indices(
            system,
            _full_vertices_with_inner(
                system, volume.volume_vertices, candidate
            ),
        )
        additional_faces.update(int(index) for index in bad_faces)
        outside_region = additional_faces.difference(
            region.active_face_indices.tolist()
        )
        if outside_region:
            region = _grow_surface_repair_region(
                system,
                region,
                np.asarray(sorted(outside_region), dtype=np.int64),
            )
            region_expansions += 1
            beta_step = B3_INITIAL_BETA_STEP
            record["repair_region_expanded"] = True
            continue
        beta_step *= 0.5
        if beta_step < B3_MINIMUM_BETA_STEP:
            if region.full_surface:
                break
            region = _grow_surface_repair_region(
                system, region, np.empty(0, dtype=np.int64)
            )
            region_expansions += 1
            beta_step = B3_INITIAL_BETA_STEP
            record["repair_region_expanded"] = True

    failure = None if beta == 1.0 else last_failure
    return _SurfaceUntangling(
        current,
        beta,
        tuple(history),
        failure,
        total_iterations,
        {
            "initial_active_inner_vertices": initial_active_count,
            "final_active_inner_vertices": int(
                len(region.active_inner_vertex_indices)
            ),
            "initial_expansion_rings": B3_INITIAL_REPAIR_RINGS,
            "final_expansion_rings": int(region.expansion_rings),
            "seed_intersection_faces": region.seed_intersection_face_count,
            "seed_bad_boundary_tetrahedra": (
                region.seed_bad_tetrahedron_count
            ),
            "region_expansions": region_expansions,
            "full_surface_fallback": bool(region.full_surface),
        },
    )


def _solve_smooth_displacement_to_target(
    problem: InstanceVolumeProblem,
    system: _InstanceDeformationSystem,
    target_inner: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Reuse the canonical factorization for a corrected inner boundary."""

    volume = problem.canonical_volume
    expected_boundary = np.concatenate(
        (volume.computational_inner_vertex_indices, volume.outer_vertex_indices)
    )
    if (
        system.topology_digest != volume.topology_digest
        or not np.array_equal(system.boundary_vertex_indices, expected_boundary)
    ):
        raise ValueError("deformation system belongs to another canonical volume")
    boundary_displacement = np.vstack(
        (
            target_inner
            - volume.volume_vertices[volume.computational_inner_vertex_indices],
            np.zeros((len(volume.outer_vertex_indices), 3), dtype=np.float64),
        )
    )
    right_hand_side = -(system.free_to_boundary @ boundary_displacement)
    free_displacement = np.asarray(
        system.factorization.solve(np.asarray(right_hand_side)), dtype=np.float64
    )
    displacement = np.zeros_like(volume.volume_vertices)
    displacement[system.boundary_vertex_indices] = boundary_displacement
    displacement[system.free_vertex_indices] = free_displacement
    residual_vector = system.free_matrix @ free_displacement - right_hand_side
    residual = float(
        np.linalg.norm(residual_vector) / max(np.linalg.norm(right_hand_side), 1.0)
    )
    if not np.isfinite(displacement).all() or residual > 1.0e-8:
        raise RuntimeError(f"{problem.shoe_name}: corrected FEM solve failed")
    return displacement, residual


def _evaluate_volume_objective(
    flat_interior: np.ndarray,
    *,
    system: _InstanceOptimizationSystem,
    fixed_vertices: np.ndarray,
    baseline: np.ndarray,
    barrier_multiplier: float,
) -> tuple[float, np.ndarray]:
    """Optimize tetrahedral interiors while both boundaries remain fixed."""

    vertices = np.asarray(fixed_vertices, dtype=np.float64).copy()
    vertices[system.interior_vertex_indices] = np.asarray(
        flat_interior, dtype=np.float64
    ).reshape(-1, 3)
    movable = system.movable_vertex_indices
    delta = vertices[movable] - baseline[movable]
    stiffness_delta = system.stiffness @ delta
    fem_energy = float(
        np.sum(delta * stiffness_delta) / system.canonical_shell_volume
    )
    gradient = np.zeros_like(vertices)
    gradient[movable] += (
        2.0 * B3_FEM_WEIGHT * stiffness_delta / system.canonical_shell_volume
    )
    tetrahedral = _tetrahedral_energy_gradient(
        system, vertices, barrier_multiplier
    )
    if tetrahedral is None:
        return math.inf, np.zeros_like(flat_interior)
    distortion_energy, determinant_energy, distortion_gradient, determinant_gradient = (
        tetrahedral
    )
    gradient += B3_DISTORTION_WEIGHT * distortion_gradient + determinant_gradient
    total = (
        B3_FEM_WEIGHT * fem_energy
        + B3_DISTORTION_WEIGHT * distortion_energy
        + barrier_multiplier * determinant_energy
    )
    interior_gradient = gradient[system.interior_vertex_indices].reshape(-1)
    if not np.isfinite(total) or not np.isfinite(interior_gradient).all():
        return math.inf, np.zeros_like(flat_interior)
    return float(total), interior_gradient


def _robust_seed_for_target(
    problem: InstanceVolumeProblem,
    system: _InstanceOptimizationSystem,
    full_displacement: np.ndarray,
    target_inner: np.ndarray,
) -> tuple[float, np.ndarray, _QualityDiagnostics]:
    volume = problem.canonical_volume
    upper = 1.0
    full = _interpolated_baseline(
        volume, full_displacement, target_inner, upper
    )
    full_quality = _deformation_quality(system, full)
    if _quality_is_robust(full_quality):
        validation, _ = _validate_continuation_candidate(volume, full)
        if validation.accepted:
            return 1.0, full, full_quality
    lower = 0.0
    for _ in range(14):
        middle = 0.5 * (lower + upper)
        candidate = _interpolated_baseline(
            volume, full_displacement, target_inner, middle
        )
        if _quality_is_robust(_deformation_quality(system, candidate)):
            lower = middle
        else:
            upper = middle
    candidate = _interpolated_baseline(
        volume, full_displacement, target_inner, lower
    )
    validation, _ = _validate_continuation_candidate(volume, candidate)
    quality = _deformation_quality(system, candidate)
    if validation.accepted and _quality_is_robust(quality):
        return lower, candidate, quality
    identity = volume.volume_vertices.copy()
    return 0.0, identity, _deformation_quality(system, identity)


def _optimize_volume_to_boundary(
    problem: InstanceVolumeProblem,
    system: _InstanceOptimizationSystem,
    deformation_system: _InstanceDeformationSystem,
    clean_target: np.ndarray,
    progress: Callable[[dict[str, Any]], None] | None,
) -> tuple[
    float,
    np.ndarray,
    _QualityDiagnostics,
    tuple[dict[str, Any], ...],
    int,
    str | None,
]:
    """Move the fixed clean boundary while optimizing only interior nodes."""

    volume = problem.canonical_volume
    full_displacement, _ = _solve_smooth_displacement_to_target(
        problem, deformation_system, clean_target
    )
    beta, current, seed_quality = _robust_seed_for_target(
        problem, system, full_displacement, clean_target
    )
    seed_beta = beta
    beta_step = B3_INITIAL_BETA_STEP
    history: list[dict[str, Any]] = []
    total_iterations = 0
    last_failure = "volume_seed"
    resolution = problem.boundary_target.surface_resolution
    canonical_inner = volume.volume_vertices[system.inner_vertex_indices]

    while beta < 1.0:
        trial_beta = min(1.0, beta + beta_step)
        delta_beta = trial_beta - beta
        target_beta = (1.0 - trial_beta) * canonical_inner + trial_beta * clean_target
        baseline = _interpolated_baseline(
            volume, full_displacement, clean_target, trial_beta
        )
        transported = current + delta_beta * full_displacement
        transported[system.inner_vertex_indices] = target_beta
        transported[system.outer_vertex_indices] = volume.volume_vertices[
            system.outer_vertex_indices
        ]
        held_interior = current.copy()
        held_interior[system.inner_vertex_indices] = target_beta
        held_interior[system.outer_vertex_indices] = volume.volume_vertices[
            system.outer_vertex_indices
        ]
        candidate = transported
        for option in (transported, held_interior):
            initial, _ = _evaluate_volume_objective(
                option[system.interior_vertex_indices].reshape(-1),
                system=system,
                fixed_vertices=option,
                baseline=baseline,
                barrier_multiplier=B3_BARRIER_MULTIPLIERS[0],
            )
            if np.isfinite(initial):
                candidate = option.copy()
                break
        else:
            record = {
                "phase": "volume_optimization",
                "from_beta": beta,
                "trial_beta": trial_beta,
                "beta_step": beta_step,
                "accepted": False,
                "failure": "nonpositive_predictor",
                "blocks": [],
            }
            history.append(record)
            if progress is not None:
                progress(record)
            beta_step *= 0.5
            if beta_step < B3_MINIMUM_BETA_STEP:
                last_failure = "nonpositive_predictor"
                break
            continue

        stage_records: list[dict[str, Any]] = []
        stage_improved = False
        for block, multiplier in enumerate(B3_BARRIER_MULTIPLIERS):
            arguments = {
                "system": system,
                "fixed_vertices": candidate,
                "baseline": baseline,
                "barrier_multiplier": multiplier,
            }
            initial_flat = candidate[system.interior_vertex_indices].reshape(-1)
            initial_value, _ = _evaluate_volume_objective(
                initial_flat, **arguments
            )
            if not np.isfinite(initial_value):
                last_failure = "infeasible_volume_step"
                stage_records.append(
                    {
                        "block": block,
                        "barrier_multiplier": multiplier,
                        "accepted": False,
                        "failure": last_failure,
                    }
                )
                break

            def normalized_objective(
                offsets: np.ndarray,
            ) -> tuple[float, np.ndarray]:
                value, coordinate_gradient = _evaluate_volume_objective(
                    initial_flat + resolution * offsets, **arguments
                )
                return value, resolution * coordinate_gradient

            optimized = minimize(
                normalized_objective,
                np.zeros_like(initial_flat),
                method="L-BFGS-B",
                jac=True,
                options={
                    "maxiter": B3_ITERATIONS_PER_BLOCK,
                    "maxcor": 10,
                    "ftol": 1.0e-12,
                    "gtol": 1.0e-8,
                    "maxls": 30,
                },
            )
            total_iterations += int(optimized.nit)
            proposal = candidate.copy()
            proposal[system.interior_vertex_indices] = (
                initial_flat + resolution * optimized.x
            ).reshape(-1, 3)
            accepted_step = 0.0
            accepted_value = initial_value
            failure = "objective_not_improved"
            line_step = 1.0
            while line_step >= 2.0**-20:
                trial = candidate + line_step * (proposal - candidate)
                trial[system.inner_vertex_indices] = target_beta
                trial[system.outer_vertex_indices] = volume.volume_vertices[
                    system.outer_vertex_indices
                ]
                trial_value, _ = _evaluate_volume_objective(
                    trial[system.interior_vertex_indices].reshape(-1), **arguments
                )
                tolerance = 1.0e-12 * max(1.0, abs(initial_value))
                if (
                    np.isfinite(trial_value)
                    and trial_value < initial_value - tolerance
                ):
                    candidate = trial
                    accepted_step = line_step
                    accepted_value = trial_value
                    failure = None
                    stage_improved = True
                    break
                line_step *= 0.5
            stage_records.append(
                {
                    "block": block,
                    "barrier_multiplier": multiplier,
                    "optimizer_iterations": int(optimized.nit),
                    "optimizer_success": bool(optimized.success),
                    "optimizer_message": str(optimized.message),
                    "initial_objective": initial_value,
                    "optimizer_objective": float(optimized.fun),
                    "accepted_objective": accepted_value,
                    "accepted_line_step": accepted_step,
                    "accepted": bool(accepted_step),
                    "failure": failure,
                }
            )

        quality = _deformation_quality(system, candidate)
        validation, _ = _validate_continuation_candidate(volume, candidate)
        accepted = bool(validation.accepted and _quality_is_robust(quality))
        record = {
            "phase": "volume_optimization",
            "from_beta": beta,
            "trial_beta": trial_beta,
            "beta_step": beta_step,
            "accepted": accepted,
            "failure": None if accepted else str(validation.failure or "bounded_distortion"),
            "minimum_jacobian_determinant": float(np.min(quality.determinants)),
            "maximum_condition_number": float(np.max(quality.condition_numbers)),
            "optimizer_changed_interior": stage_improved,
            "blocks": stage_records,
        }
        history.append(record)
        if progress is not None:
            progress(record)
        if accepted:
            current = candidate
            beta = trial_beta
            continue
        beta_step *= 0.5
        last_failure = str(record["failure"])
        if beta_step < B3_MINIMUM_BETA_STEP:
            break

    failure = None if beta == 1.0 else f"seed_beta={seed_beta:.17g}; {last_failure}"
    return (
        beta,
        current,
        seed_quality,
        tuple(history),
        total_iterations,
        failure,
    )


def optimize_instance_volume(
    problem: InstanceVolumeProblem,
    continuation: InstanceVolumeContinuation,
    *,
    optimization_system: _InstanceOptimizationSystem,
    deformation_system: _InstanceDeformationSystem,
    progress: Callable[[dict[str, Any]], None] | None = None,
    timing: Callable[[str, float], None] | None = None,
) -> InstanceVolumeOptimization:
    """Untangle the computational surface, then fit a valid volume to it."""

    volume = problem.canonical_volume
    system = optimization_system
    if (
        system.topology_digest != volume.topology_digest
        or continuation.canonical_volume_topology_digest != volume.topology_digest
        or continuation.boundary_target_geometry_digest
        != problem.boundary_target.geometry_digest
        or continuation.shoe_name != problem.shoe_name
        or not np.array_equal(
            system.interior_vertex_indices, deformation_system.free_vertex_indices
        )
    ):
        raise ValueError(f"{problem.shoe_name}: B3 inputs belong to another problem")

    started = time.perf_counter()
    promoted = _final_validation(
        problem, system, continuation.volume_vertices, beta=continuation.reached_alpha
    )
    if timing is not None:
        timing("initial_final_validation", time.perf_counter() - started)
    if continuation.reached_alpha == 1.0 and promoted.accepted:
        return InstanceVolumeOptimization(
            shoe_name=problem.shoe_name,
            status="final_exact_target",
            volume_vertices=continuation.volume_vertices.copy(),
            jacobian_determinants=promoted.quality.determinants,
            jacobian_singular_values=promoted.quality.singular_values,
            condition_numbers=promoted.quality.condition_numbers,
            target_correction_vectors=promoted.target_correction_vectors,
            reached_beta=1.0,
            optimization_history=(),
            diagnostics={
                **promoted.diagnostics,
                "source_b2_status": continuation.status,
                "source_b2_alpha": continuation.reached_alpha,
                "surface_untangling_beta": 1.0,
                "robust_seed_quality": promoted.quality.summary(),
                "stopping_reason": "b2_target_promoted_without_optimization",
                "surface_optimizer_iterations": 0,
                "volume_optimizer_iterations": 0,
                "optimization_iterations": 0,
                "initial_active_inner_vertices": 0,
                "final_active_inner_vertices": 0,
                "region_expansions": 0,
                "full_surface_fallback": False,
                "corrected_inner_vertex_indices": [],
            },
            canonical_volume_topology_digest=volume.topology_digest,
            boundary_target_geometry_digest=problem.boundary_target.geometry_digest,
        )

    started = time.perf_counter()
    surface = _untangle_computational_boundary(
        problem, continuation, system, progress, timing
    )
    if timing is not None:
        timing("surface_untangling", time.perf_counter() - started)
    history = list(surface.history)
    if surface.reached_beta == 1.0:
        started = time.perf_counter()
        (
            beta,
            current,
            seed_quality,
            volume_history,
            volume_iterations,
            volume_failure,
        ) = _optimize_volume_to_boundary(
            problem,
            system,
            deformation_system,
            surface.vertices,
            progress,
        )
        if timing is not None:
            timing("volume_deformation", time.perf_counter() - started)
        history.extend(volume_history)
    else:
        beta = 0.0
        current = volume.volume_vertices.copy()
        seed_quality = _deformation_quality(system, current)
        volume_iterations = 0
        volume_failure = str(surface.failure or "surface_untangling_failed")

    started = time.perf_counter()
    final = _final_validation(problem, system, current, beta=beta)
    if timing is not None:
        timing("final_validation", time.perf_counter() - started)
    if final.accepted:
        maximum_correction = float(
            np.max(np.linalg.norm(final.target_correction_vectors, axis=1))
        )
        status = (
            "final_exact_target"
            if maximum_correction
            <= B3_EXACT_CORRECTION_RESOLUTIONS
            * problem.boundary_target.surface_resolution
            else "final_corrected_target"
        )
        stopping_reason = "target_reached_and_validated"
    else:
        status = "failed_11_b3"
        stopping_reason = str(final.failure or volume_failure)
    corrected_indices = np.flatnonzero(
        np.linalg.norm(final.target_correction_vectors, axis=1)
        > B3_EXACT_CORRECTION_RESOLUTIONS
        * problem.boundary_target.surface_resolution
    )
    return InstanceVolumeOptimization(
        shoe_name=problem.shoe_name,
        status=status,
        volume_vertices=current,
        jacobian_determinants=final.quality.determinants,
        jacobian_singular_values=final.quality.singular_values,
        condition_numbers=final.quality.condition_numbers,
        target_correction_vectors=final.target_correction_vectors,
        reached_beta=beta,
        optimization_history=tuple(history),
        diagnostics={
            **final.diagnostics,
            "source_b2_status": continuation.status,
            "source_b2_alpha": continuation.reached_alpha,
            "surface_untangling_beta": surface.reached_beta,
            "surface_untangling_failure": surface.failure,
            "robust_seed_quality": seed_quality.summary(),
            "stopping_reason": stopping_reason,
            "volume_failure": volume_failure,
            "surface_optimizer_iterations": surface.optimizer_iterations,
            "volume_optimizer_iterations": volume_iterations,
            "optimization_iterations": (
                surface.optimizer_iterations + volume_iterations
            ),
            **surface.diagnostics,
            "corrected_inner_vertex_indices": corrected_indices.tolist(),
        },
        canonical_volume_topology_digest=volume.topology_digest,
        boundary_target_geometry_digest=problem.boundary_target.geometry_digest,
    )
