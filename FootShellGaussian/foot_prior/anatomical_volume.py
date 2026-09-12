"""Canonical tetrahedral volume around the neutral SUPR foot and lower leg."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import splu, spsolve
import trimesh

from .anatomy import (
    DENSE_FOOT_FACE_COUNT as DENSE_FACE_COUNT,
    DENSE_FOOT_VERTEX_COUNT as DENSE_VERTEX_COUNT,
    EXTENDED_FACE_COUNT,
    EXTENDED_VERTEX_COUNT,
    array_digest as _array_digest,
    directed_boundary_loop as _directed_boundary_loop,
    topology_digest as _topology_digest,
)
from .cavity import _closest_points_on_triangles, _find_collision_pairs
from .mesh import TriangleMesh, load_triangle_mesh
from .supr_foot import build_supr_mesh_subdivision


BOUNDARY_FOOT_SKIN = 1
BOUNDARY_ANKLE_TRANSITION = 2
BOUNDARY_LOWER_LEG_SKIN = 3
BOUNDARY_KNEE_TRUNCATION = 4
BOUNDARY_OUTER_ENVELOPE = 5
BOUNDARY_LABEL_NAMES = (
    "unused",
    "foot_skin",
    "ankle_transition",
    "lower_leg_skin",
    "knee_truncation",
    "outer_envelope",
)
ENVELOPE_CENTER = np.asarray((0.45, -0.825, 0.0), dtype=np.float64)
ENVELOPE_RADII = np.asarray((0.85, 1.225, 0.48), dtype=np.float64)
ENVELOPE_POWER = 4.0
ENVELOPE_SUBDIVISIONS = 3
GMSH_OPTIONS = {
    "General.NumThreads": 1.0,
    "Mesh.MaxNumThreads1D": 1.0,
    "Mesh.MaxNumThreads2D": 1.0,
    "Mesh.MaxNumThreads3D": 1.0,
    "Mesh.Algorithm3D": 1.0,
    "Mesh.ElementOrder": 1.0,
    "Mesh.MeshSizeFromPoints": 0.0,
    "Mesh.MeshSizeFromCurvature": 0.0,
    "Mesh.MeshSizeExtendFromBoundary": 0.0,
    # The inner proxy and outer envelope are already the exact conforming
    # boundary meshes.  Only the empty 3-D entity may be meshed: allowing Gmsh
    # to revisit the 1-D/2-D entities silently replaces supplied triangles and
    # drops boundary vertices.
    "Mesh.MeshOnlyEmpty": 1.0,
    "Mesh.MeshSizeMin": 0.025,
    "Mesh.MeshSizeMax": 0.075,
    "Mesh.RandomFactor": 1.0e-9,
}
GMSH_INNER_REMESH_MIN_SIZE = 0.01
GMSH_INNER_REMESH_MAX_SIZE = 0.02
PYMESHFIX_MAX_ITERATIONS = 10
REPAIR_NEIGHBOUR_RINGS = 2
INSTANCE_CONTINUATION_INITIAL_STEP = 0.25
INSTANCE_CONTINUATION_MINIMUM_STEP = 1.0e-4
INSTANCE_JACOBIAN_DETERMINANT_FLOOR = 1.0e-6


@dataclass(frozen=True)
class _ExtendedReference:
    """Validated, immutable Checkpoint 10-A reference artifacts."""

    root: Path
    vertices: np.ndarray
    faces: np.ndarray
    foot_vertex_indices: np.ndarray
    foot_face_indices: np.ndarray
    lower_leg_vertex_indices: np.ndarray
    lower_leg_face_indices: np.ndarray
    ankle_transition_face_indices: np.ndarray
    knee_loop: np.ndarray
    ankle_correspondence: np.ndarray
    component_face_labels: np.ndarray
    native_foot_vertices: np.ndarray
    native_foot_faces: np.ndarray
    dense_foot_face_parent_indices: np.ndarray
    geometry_digest: str
    topology_digest: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _ComputationalBoundary:
    vertices: np.ndarray
    faces: np.ndarray
    expected_face_labels: np.ndarray
    knee_cap_face_indices: np.ndarray
    repair_zone_canonical_vertex_indices: np.ndarray
    native_intersection_pairs: np.ndarray
    subdivision_metadata: dict[str, Any]


@dataclass(frozen=True)
class CanonicalAnatomicalVolume:
    """Shared foot-and-lower-leg domain and its harmonic outward coordinate."""

    volume_vertices: np.ndarray
    tetrahedra: np.ndarray
    boundary_faces: np.ndarray
    boundary_labels: np.ndarray
    harmonic_r: np.ndarray
    harmonic_r_gradient: np.ndarray
    computational_inner_vertex_indices: np.ndarray
    computational_inner_faces: np.ndarray
    computational_inner_face_labels: np.ndarray
    zero_boundary_vertex_indices: np.ndarray
    knee_cap_face_indices: np.ndarray
    knee_cap_natural_vertex_indices: np.ndarray
    computational_to_canonical_face_indices: np.ndarray
    computational_to_canonical_barycentric: np.ndarray
    computational_to_canonical_distances: np.ndarray
    canonical_to_computational_face_indices: np.ndarray
    canonical_to_computational_barycentric: np.ndarray
    canonical_to_computational_distances: np.ndarray
    initial_self_intersection_pairs: np.ndarray
    repair_zone_canonical_vertex_indices: np.ndarray
    outer_vertex_indices: np.ndarray
    tetrahedron_signed_volumes: np.ndarray
    tetrahedron_mean_ratio_quality: np.ndarray
    topology_digest: str
    envelope_topology_digest: str
    extended_surface_digest: str
    diagnostics: dict[str, Any]

    @property
    def computational_inner_mesh(self) -> TriangleMesh:
        if not np.array_equal(
            self.computational_inner_vertex_indices,
            np.arange(len(self.computational_inner_vertex_indices), dtype=np.int64),
        ):
            raise RuntimeError("computational inner vertices are not stored first")
        return TriangleMesh(
            self.volume_vertices[self.computational_inner_vertex_indices],
            self.computational_inner_faces,
        )

    @property
    def outer_envelope_mesh(self) -> TriangleMesh:
        faces = self.boundary_faces[
            self.boundary_labels == BOUNDARY_OUTER_ENVELOPE
        ]
        return TriangleMesh(self.volume_vertices, faces)

    def to_dict(self) -> dict[str, Any]:
        """Return the human-readable metadata for this reference volume."""

        return {
            "schema_version": 3,
            "stage": "canonical_foot_lower_leg_anatomical_volume",
            "reference_domain": "A",
            "coordinate_convention": {
                "surface": (
                    "authoritative extended face ID plus three barycentric weights; "
                    "the r=0 computational boundary maps to this chart explicitly"
                ),
                "volume": "tetrahedron ID plus four barycentric weights",
                "r": (
                    "normalized harmonic layer coordinate; r=0 on anatomical skin "
                    "and r=1 on the outer envelope; not a metric distance"
                ),
            },
            "counts": {
                "volume_vertices": int(len(self.volume_vertices)),
                "tetrahedra": int(len(self.tetrahedra)),
                "boundary_faces": int(len(self.boundary_faces)),
                "authoritative_surface_vertices": EXTENDED_VERTEX_COUNT,
                "authoritative_surface_faces": EXTENDED_FACE_COUNT,
                "computational_inner_vertices": int(
                    len(self.computational_inner_vertex_indices)
                ),
                "computational_inner_faces": int(len(self.computational_inner_faces)),
                "knee_cap_faces": int(len(self.knee_cap_face_indices)),
                "outer_vertices": int(len(self.outer_vertex_indices)),
                "outer_faces": int(
                    np.count_nonzero(
                        self.boundary_labels == BOUNDARY_OUTER_ENVELOPE
                    )
                ),
            },
            "boundary_labels": {
                str(index): name
                for index, name in enumerate(BOUNDARY_LABEL_NAMES)
                if index > 0
            },
            "computational_boundary": self.diagnostics["repair"],
            "knee_truncation": {
                "method": "ordered_dense_knee_loop_centroid_fan",
                "boundary_face_indices": self.knee_cap_face_indices.tolist(),
                "natural_boundary_vertex_indices": (
                    self.knee_cap_natural_vertex_indices.tolist()
                ),
                "harmonic_boundary_condition": "natural_neumann",
                "anatomical_skin": False,
            },
            "outer_envelope": {
                "type": "fourth_order_superellipsoid",
                "center": ENVELOPE_CENTER.tolist(),
                "radii": ENVELOPE_RADII.tolist(),
                "bounds": [
                    (ENVELOPE_CENTER - ENVELOPE_RADII).tolist(),
                    (ENVELOPE_CENTER + ENVELOPE_RADII).tolist(),
                ],
                "power": ENVELOPE_POWER,
                "source": "deterministic subdivision-3 icosphere",
                "subdivisions": ENVELOPE_SUBDIVISIONS,
                "topology_sha256": self.envelope_topology_digest,
            },
            "tetrahedralization": {
                "engine": "Gmsh Python API",
                "version": self.diagnostics["gmsh_version"],
                "surface_policy": (
                    "conforming mesh with the registered computational boundary "
                    "preserved exactly; authoritative anatomy remains unchanged"
                ),
                "options": self.diagnostics["gmsh_options"],
                "single_threaded": True,
            },
            "harmonic_r": self.diagnostics["harmonic_r"],
            "quality": self.diagnostics["quality"],
            "connectivity": self.diagnostics["connectivity"],
            "digests": {
                "volume_topology_sha256": self.topology_digest,
                "extended_surface_sha256": self.extended_surface_digest,
                "computational_inner_sha256": self.diagnostics[
                    "computational_inner_sha256"
                ],
            },
            "scope": {
                "supported": (
                    "foot and lower-leg anatomical space from toes to the "
                    "canonical knee truncation"
                ),
                "deferred": [
                    "per-fitted-foot volume deformation",
                    "shoe-to-domain mapping",
                    "volume-wide u and v fields",
                ],
            },
        }


@dataclass(frozen=True)
class InstanceBoundaryTarget:
    """Per-instance target positions for the canonical computational boundary."""

    vertices: np.ndarray
    faces: np.ndarray
    face_labels: np.ndarray
    intersecting_face_pairs: np.ndarray
    intersecting_face_indices: np.ndarray
    reverse_reconstructed_vertices: np.ndarray
    reverse_distances: np.ndarray
    envelope_equation: np.ndarray
    surface_resolution: float
    signed_volume: float
    geometry_digest: str
    canonical_volume_topology_digest: str
    canonical_computational_boundary_digest: str
    fitted_surface_digest: str
    status: str
    diagnostics: dict[str, Any]

    @property
    def mesh(self) -> TriangleMesh:
        return TriangleMesh(self.vertices, self.faces)

    def to_dict(self) -> dict[str, Any]:
        """Return the portable Checkpoint 11-A metadata."""

        return {
            "schema_version": 1,
            "stage": "fitted_anatomical_boundary_target",
            "status": self.status,
            "meaning": (
                "soft target for Checkpoint 11-B; not a tetrahedral boundary or "
                "a forward/inverse anatomical-volume map"
            ),
            "counts": {
                "target_vertices": int(len(self.vertices)),
                "target_faces": int(len(self.faces)),
                "intersecting_face_pairs": int(len(self.intersecting_face_pairs)),
                "intersecting_faces": int(len(self.intersecting_face_indices)),
                "envelope_outside_vertices": int(
                    np.count_nonzero(self.envelope_equation >= 1.0 - 1.0e-10)
                ),
            },
            "correspondence": {
                "target": (
                    "canonical computational vertex to authoritative extended "
                    "face ID and barycentric weights, evaluated on fitted anatomy"
                ),
                "reverse": (
                    "authoritative fitted vertex reconstructed from the stored "
                    "canonical-to-computational face and barycentric coordinate"
                ),
                "vertex_order_preserved": True,
                "face_topology_preserved": True,
            },
            "geometry": {
                "bounds": np.stack(
                    (self.vertices.min(axis=0), self.vertices.max(axis=0)), axis=0
                ).tolist(),
                "signed_volume": self.signed_volume,
                "surface_resolution": self.surface_resolution,
                "geometry_sha256": self.geometry_digest,
            },
            "fidelity": self.diagnostics["fidelity"],
            "self_intersections": self.diagnostics["self_intersections"],
            "frozen_envelope": self.diagnostics["frozen_envelope"],
            "knee_truncation": self.diagnostics["knee_truncation"],
            "digests": {
                "canonical_volume_topology_sha256": (
                    self.canonical_volume_topology_digest
                ),
                "canonical_computational_boundary_sha256": (
                    self.canonical_computational_boundary_digest
                ),
                "fitted_extended_surface_sha256": self.fitted_surface_digest,
            },
            "deferred": [
                "topology-preserving target untangling",
                "outer-boundary deformation",
                "tetrahedral interior deformation",
                "chi_i and Phi_i",
                "anatomical fibers",
                "shoe geometry mapping",
            ],
        }


@dataclass(frozen=True)
class InstanceVolumeProblem:
    """Validated Checkpoint 11-B1 inputs for one fitted shoe instance."""

    shoe_name: str
    canonical_volume: CanonicalAnatomicalVolume
    boundary_target: InstanceBoundaryTarget
    fitted_surface: TriangleMesh
    anatomical_volume_root: Path
    extended_surface_root: Path
    boundary_target_metadata: dict[str, Any]
    fitted_surface_metadata: dict[str, Any]


@dataclass(frozen=True)
class InstanceVolumeContinuation:
    """Last valid baseline state reached by Checkpoint 11-B2."""

    shoe_name: str
    volume_vertices: np.ndarray
    reached_alpha: float
    status: str
    stopping_reason: str
    jacobian_determinants: np.ndarray
    accepted_alphas: np.ndarray
    attempts: tuple[dict[str, Any], ...]
    canonical_volume_topology_digest: str
    boundary_target_geometry_digest: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return portable metadata for the incomplete continuation state."""

        return {
            "schema_version": 1,
            "stage": "instance_volume_continuation",
            "status": self.status,
            "meaning": (
                "Checkpoint 11-B2 warm start only; not a final cleaned instance "
                "volume and not chi_i or Phi_i"
            ),
            "shoe_name": self.shoe_name,
            "reached_alpha": self.reached_alpha,
            "accepted_alphas": self.accepted_alphas.tolist(),
            "stopping_reason": self.stopping_reason,
            "configuration": {
                "initial_step": INSTANCE_CONTINUATION_INITIAL_STEP,
                "minimum_step": INSTANCE_CONTINUATION_MINIMUM_STEP,
                "jacobian_determinant_floor": (
                    INSTANCE_JACOBIAN_DETERMINANT_FLOOR
                ),
                "outer_boundary": "fixed exactly to the canonical envelope",
                "inner_boundary": (
                    "linear interpolation from the canonical computational "
                    "boundary to the Checkpoint 11-A target"
                ),
                "interior": "linear tetrahedral FEM smooth displacement",
            },
            "counts": {
                "volume_vertices": int(len(self.volume_vertices)),
                "attempts": int(len(self.attempts)),
                "accepted_steps": int(len(self.accepted_alphas) - 1),
            },
            "final": {
                "jacobian_determinant": _summary(
                    self.jacobian_determinants
                ),
                **self.diagnostics,
            },
            "history": list(self.attempts),
            "digests": {
                "canonical_volume_topology_sha256": (
                    self.canonical_volume_topology_digest
                ),
                "boundary_target_geometry_sha256": (
                    self.boundary_target_geometry_digest
                ),
            },
            "deferred": [
                "joint boundary and interior optimization",
                "topology-preserving target untangling",
                "bounded-distortion acceptance",
                "final instance-volume artifacts",
                "chi_i and Phi_i",
            ],
        }


@dataclass(frozen=True)
class _InstanceDeformationSystem:
    """Reusable factorization for smooth displacement of the canonical volume."""

    topology_digest: str
    boundary_vertex_indices: np.ndarray
    free_vertex_indices: np.ndarray
    free_matrix: Any
    free_to_boundary: Any
    factorization: Any


@dataclass(frozen=True)
class _ContinuationValidation:
    accepted: bool
    failure: str | None
    minimum_jacobian_determinant: float | None
    below_determinant_floor_count: int
    degenerate_inner_face_count: int | None = None
    envelope_outside_vertex_count: int | None = None
    inner_self_intersection_count: int | None = None
    inner_outer_intersection_count: int | None = None

    def to_dict(self, alpha: float, accepted: bool) -> dict[str, Any]:
        return {
            "alpha": alpha,
            "accepted": accepted,
            "failure": self.failure,
            "minimum_jacobian_determinant": (
                self.minimum_jacobian_determinant
            ),
            "below_determinant_floor_count": (
                self.below_determinant_floor_count
            ),
            "degenerate_inner_face_count": self.degenerate_inner_face_count,
            "envelope_outside_vertex_count": (
                self.envelope_outside_vertex_count
            ),
            "inner_self_intersection_count": (
                self.inner_self_intersection_count
            ),
            "inner_outer_intersection_count": (
                self.inner_outer_intersection_count
            ),
        }


def load_canonical_anatomical_volume(
    anatomical_volume_root: str | Path,
) -> CanonicalAnatomicalVolume:
    """Load and validate the saved Checkpoint 10-B canonical volume."""

    root = Path(anatomical_volume_root).expanduser().resolve(strict=True)
    reference = root / "reference"
    if not reference.is_dir():
        raise NotADirectoryError(reference)
    json_path = reference / "canonical_volume.json"
    npz_path = reference / "canonical_volume.npz"
    for path in (json_path, npz_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if (
        payload.get("schema_version") != 3
        or payload.get("stage") != "canonical_foot_lower_leg_anatomical_volume"
    ):
        raise ValueError("canonical anatomical volume has an unsupported schema")

    required = {
        "volume_vertices",
        "tetrahedra",
        "boundary_faces",
        "boundary_labels",
        "harmonic_r",
        "harmonic_r_gradient",
        "computational_inner_vertex_indices",
        "computational_inner_faces",
        "computational_inner_face_labels",
        "zero_boundary_vertex_indices",
        "knee_cap_face_indices",
        "knee_cap_natural_vertex_indices",
        "computational_to_canonical_face_indices",
        "computational_to_canonical_barycentric",
        "computational_to_canonical_distances",
        "canonical_to_computational_face_indices",
        "canonical_to_computational_barycentric",
        "canonical_to_computational_distances",
        "initial_self_intersection_pairs",
        "repair_zone_canonical_vertex_indices",
        "outer_vertex_indices",
        "tetrahedron_signed_volumes",
        "tetrahedron_mean_ratio_quality",
        "volume_topology_sha256",
        "extended_surface_sha256",
        "outer_envelope_topology_sha256",
    }
    with np.load(npz_path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"canonical volume NPZ is missing arrays: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in required}

    vertices = np.asarray(arrays["volume_vertices"], dtype=np.float64)
    tetrahedra = np.asarray(arrays["tetrahedra"], dtype=np.int64)
    boundary_faces = np.asarray(arrays["boundary_faces"], dtype=np.int64)
    boundary_labels = np.asarray(arrays["boundary_labels"], dtype=np.int16)
    harmonic_r = np.asarray(arrays["harmonic_r"], dtype=np.float64)
    harmonic_gradient = np.asarray(
        arrays["harmonic_r_gradient"], dtype=np.float64
    )
    inner_indices = np.asarray(
        arrays["computational_inner_vertex_indices"], dtype=np.int64
    )
    inner_faces = np.asarray(arrays["computational_inner_faces"], dtype=np.int64)
    inner_labels = np.asarray(
        arrays["computational_inner_face_labels"], dtype=np.int16
    )
    outer_indices = np.asarray(arrays["outer_vertex_indices"], dtype=np.int64)
    counts = payload.get("counts", {})
    if (
        vertices.ndim != 2
        or vertices.shape[1:] != (3,)
        or tetrahedra.ndim != 2
        or tetrahedra.shape[1:] != (4,)
        or boundary_faces.ndim != 2
        or boundary_faces.shape[1:] != (3,)
        or boundary_labels.shape != (len(boundary_faces),)
        or harmonic_r.shape != (len(vertices),)
        or harmonic_gradient.shape != (len(tetrahedra), 3)
        or not np.isfinite(vertices).all()
        or not np.isfinite(harmonic_r).all()
        or not np.isfinite(harmonic_gradient).all()
        or np.any(tetrahedra < 0)
        or np.any(tetrahedra >= len(vertices))
        or np.any(boundary_faces < 0)
        or np.any(boundary_faces >= len(vertices))
        or counts.get("volume_vertices") != len(vertices)
        or counts.get("tetrahedra") != len(tetrahedra)
        or counts.get("boundary_faces") != len(boundary_faces)
    ):
        raise ValueError("canonical volume arrays or recorded counts are invalid")
    if (
        inner_indices.shape != (8_224,)
        or not np.array_equal(inner_indices, np.arange(8_224, dtype=np.int64))
        or inner_faces.shape != (16_444, 3)
        or inner_labels.shape != (16_444,)
        or np.any(inner_faces < 0)
        or np.any(inner_faces >= len(inner_indices))
        or set(np.unique(inner_labels).tolist())
        != {
            BOUNDARY_FOOT_SKIN,
            BOUNDARY_ANKLE_TRANSITION,
            BOUNDARY_LOWER_LEG_SKIN,
            BOUNDARY_KNEE_TRUNCATION,
        }
        or outer_indices.shape != (642,)
        or not np.array_equal(
            outer_indices,
            np.arange(len(inner_indices), len(inner_indices) + 642, dtype=np.int64),
        )
    ):
        raise ValueError("canonical computational or outer boundary is invalid")

    computational_faces = np.asarray(
        arrays["computational_to_canonical_face_indices"], dtype=np.int64
    )
    computational_weights = np.asarray(
        arrays["computational_to_canonical_barycentric"], dtype=np.float64
    )
    canonical_faces = np.asarray(
        arrays["canonical_to_computational_face_indices"], dtype=np.int64
    )
    canonical_weights = np.asarray(
        arrays["canonical_to_computational_barycentric"], dtype=np.float64
    )
    if (
        computational_faces.shape != (len(inner_indices),)
        or computational_weights.shape != (len(inner_indices), 3)
        or np.any(computational_faces < 0)
        or np.any(computational_faces >= EXTENDED_FACE_COUNT + 68)
        or canonical_faces.shape != (EXTENDED_VERTEX_COUNT,)
        or canonical_weights.shape != (EXTENDED_VERTEX_COUNT, 3)
        or np.any(canonical_faces < 0)
        or np.any(canonical_faces >= len(inner_faces))
        or not np.isfinite(computational_weights).all()
        or not np.isfinite(canonical_weights).all()
        or np.any(computational_weights < -1.0e-12)
        or np.any(canonical_weights < -1.0e-12)
        or not np.allclose(
            computational_weights.sum(axis=1), 1.0, atol=1.0e-12, rtol=0.0
        )
        or not np.allclose(
            canonical_weights.sum(axis=1), 1.0, atol=1.0e-12, rtol=0.0
        )
    ):
        raise ValueError("canonical volume surface correspondence is invalid")

    topology_digest = str(arrays["volume_topology_sha256"].item())
    extended_digest = str(arrays["extended_surface_sha256"].item())
    envelope_digest = str(arrays["outer_envelope_topology_sha256"].item())
    digests = payload.get("digests", {})
    if (
        topology_digest != _array_digest(tetrahedra, boundary_faces, boundary_labels)
        or topology_digest != digests.get("volume_topology_sha256")
        or extended_digest != digests.get("extended_surface_sha256")
        or envelope_digest
        != payload.get("outer_envelope", {}).get("topology_sha256")
    ):
        raise ValueError("canonical volume digest validation failed")
    computational_digest = _array_digest(
        vertices[inner_indices], inner_faces, inner_labels
    )
    if computational_digest != digests.get("computational_inner_sha256"):
        raise ValueError("canonical computational-boundary digest is invalid")

    diagnostics = {
        "gmsh_version": payload.get("tetrahedralization", {}).get("version"),
        "gmsh_options": payload.get("tetrahedralization", {}).get("options", {}),
        "computational_inner_sha256": computational_digest,
        "repair": payload.get("computational_boundary", {}),
        "harmonic_r": payload.get("harmonic_r", {}),
        "quality": payload.get("quality", {}),
        "connectivity": payload.get("connectivity", {}),
    }
    return CanonicalAnatomicalVolume(
        volume_vertices=vertices,
        tetrahedra=tetrahedra,
        boundary_faces=boundary_faces,
        boundary_labels=boundary_labels,
        harmonic_r=harmonic_r,
        harmonic_r_gradient=harmonic_gradient,
        computational_inner_vertex_indices=inner_indices,
        computational_inner_faces=inner_faces,
        computational_inner_face_labels=inner_labels,
        zero_boundary_vertex_indices=np.asarray(
            arrays["zero_boundary_vertex_indices"], dtype=np.int64
        ),
        knee_cap_face_indices=np.asarray(
            arrays["knee_cap_face_indices"], dtype=np.int64
        ),
        knee_cap_natural_vertex_indices=np.asarray(
            arrays["knee_cap_natural_vertex_indices"], dtype=np.int64
        ),
        computational_to_canonical_face_indices=computational_faces,
        computational_to_canonical_barycentric=computational_weights,
        computational_to_canonical_distances=np.asarray(
            arrays["computational_to_canonical_distances"], dtype=np.float64
        ),
        canonical_to_computational_face_indices=canonical_faces,
        canonical_to_computational_barycentric=canonical_weights,
        canonical_to_computational_distances=np.asarray(
            arrays["canonical_to_computational_distances"], dtype=np.float64
        ),
        initial_self_intersection_pairs=np.asarray(
            arrays["initial_self_intersection_pairs"], dtype=np.int64
        ),
        repair_zone_canonical_vertex_indices=np.asarray(
            arrays["repair_zone_canonical_vertex_indices"], dtype=np.int64
        ),
        outer_vertex_indices=outer_indices,
        tetrahedron_signed_volumes=np.asarray(
            arrays["tetrahedron_signed_volumes"], dtype=np.float64
        ),
        tetrahedron_mean_ratio_quality=np.asarray(
            arrays["tetrahedron_mean_ratio_quality"], dtype=np.float64
        ),
        topology_digest=topology_digest,
        envelope_topology_digest=envelope_digest,
        extended_surface_digest=extended_digest,
        diagnostics=diagnostics,
    )


def _read_json_object(path: Path) -> dict[str, Any]:
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


def _load_extended_reference(root: str | Path) -> _ExtendedReference:
    source = Path(root).expanduser().resolve(strict=True)
    reference = source / "reference"
    if not reference.is_dir():
        raise NotADirectoryError(reference)
    json_path = reference / "canonical_extended_surface.json"
    npz_path = reference / "canonical_extended_surface.npz"
    ply_path = reference / "neutral_foot_lower_leg.ply"
    for path in (json_path, npz_path, ply_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    payload = _read_json_object(json_path)
    if (
        payload.get("schema_version") != 1
        or payload.get("stage") != "canonical_dense_supr_foot_lower_leg_reference"
    ):
        raise ValueError("extended anatomical reference has an unsupported schema")
    required = {
        "extended_reference_vertices",
        "extended_faces",
        "foot_vertex_indices",
        "foot_face_indices",
        "lower_leg_vertex_indices",
        "lower_leg_face_indices",
        "ankle_transition_face_indices",
        "knee_boundary_vertex_indices",
        "ankle_loop_correspondence",
        "extended_vertex_chart_face_indices",
        "extended_vertex_chart_barycentric",
        "foot_native_reference_vertices",
        "foot_native_faces",
        "foot_dense_face_parent_indices",
        "longitudinal_vertex_labels",
        "longitudinal_face_labels",
        "surface_vertex_labels",
        "surface_face_labels",
        "component_vertex_labels",
        "component_face_labels",
        "landmark_names",
        "landmark_primary_vertex_indices",
        "landmark_reference_positions",
    }
    with np.load(npz_path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"extended anatomical NPZ is missing arrays: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in required}

    vertices = np.asarray(arrays["extended_reference_vertices"], dtype=np.float64)
    faces = np.asarray(arrays["extended_faces"], dtype=np.int64)
    if vertices.shape != (EXTENDED_VERTEX_COUNT, 3) or faces.shape != (
        EXTENDED_FACE_COUNT,
        3,
    ):
        raise ValueError("extended anatomy must use the 6,951/13,832 topology")
    if (
        not np.isfinite(vertices).all()
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        raise ValueError("extended anatomy contains invalid geometry")

    def indices(name: str, expected: np.ndarray | None = None) -> np.ndarray:
        values = np.asarray(arrays[name], dtype=np.int64)
        if values.ndim != 1 or len(np.unique(values)) != len(values):
            raise ValueError(f"{name} must contain unique one-dimensional indices")
        if expected is not None and not np.array_equal(values, expected):
            raise ValueError(f"{name} does not match the fixed shared topology")
        return values

    foot_vertices = indices(
        "foot_vertex_indices", np.arange(DENSE_VERTEX_COUNT, dtype=np.int64)
    )
    foot_faces = indices(
        "foot_face_indices", np.arange(DENSE_FACE_COUNT, dtype=np.int64)
    )
    leg_vertices = indices(
        "lower_leg_vertex_indices",
        np.arange(DENSE_VERTEX_COUNT, EXTENDED_VERTEX_COUNT, dtype=np.int64),
    )
    leg_faces = indices("lower_leg_face_indices")
    transition_faces = indices("ankle_transition_face_indices")
    if (
        len(leg_faces) != 5_472
        or len(transition_faces) != 120
        or not np.array_equal(
            np.sort(np.concatenate((foot_faces, leg_faces, transition_faces))),
            np.arange(EXTENDED_FACE_COUNT, dtype=np.int64),
        )
    ):
        raise ValueError("extended component face ranges are invalid")

    knee_loop = np.asarray(arrays["knee_boundary_vertex_indices"], dtype=np.int64)
    correspondence = np.asarray(arrays["ankle_loop_correspondence"], dtype=np.int64)
    if (
        knee_loop.shape != (68,)
        or correspondence.shape != (60, 2)
        or not np.array_equal(_directed_boundary_loop(faces), knee_loop)
    ):
        raise ValueError("extended ankle or knee correspondence is invalid")

    component_faces = np.asarray(arrays["component_face_labels"], dtype=np.int16)
    component_vertices = np.asarray(arrays["component_vertex_labels"], dtype=np.int16)
    if (
        component_faces.shape != (EXTENDED_FACE_COUNT,)
        or component_vertices.shape != (EXTENDED_VERTEX_COUNT,)
        or np.any(component_faces < 0)
        or np.any(component_faces > 2)
        or np.any(component_vertices < 0)
        or np.any(component_vertices > 2)
        or not np.array_equal(np.flatnonzero(component_faces == 0), foot_faces)
        or not np.array_equal(np.flatnonzero(component_faces == 1), transition_faces)
        or not np.array_equal(np.flatnonzero(component_faces == 2), leg_faces)
    ):
        raise ValueError("extended component labels disagree with the topology")
    for name, size in (
        ("longitudinal_vertex_labels", EXTENDED_VERTEX_COUNT),
        ("surface_vertex_labels", EXTENDED_VERTEX_COUNT),
        ("longitudinal_face_labels", EXTENDED_FACE_COUNT),
        ("surface_face_labels", EXTENDED_FACE_COUNT),
    ):
        values = np.asarray(arrays[name])
        if values.shape != (size,) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} is invalid")

    chart_faces = np.asarray(
        arrays["extended_vertex_chart_face_indices"], dtype=np.int64
    )
    chart_weights = np.asarray(
        arrays["extended_vertex_chart_barycentric"], dtype=np.float64
    )
    if (
        chart_faces.shape != (EXTENDED_VERTEX_COUNT,)
        or chart_weights.shape != (EXTENDED_VERTEX_COUNT, 3)
        or np.any(chart_faces < 0)
        or np.any(chart_faces >= EXTENDED_FACE_COUNT)
        or not np.isfinite(chart_weights).all()
        or np.any(chart_weights < -1.0e-12)
        or not np.allclose(chart_weights.sum(axis=1), 1.0, atol=1.0e-12)
    ):
        raise ValueError("extended surface charts are invalid")
    reconstructed = np.einsum(
        "ni,nij->nj", chart_weights, vertices[faces[chart_faces]]
    )
    if not np.allclose(reconstructed, vertices, atol=1.0e-12, rtol=0.0):
        raise ValueError("extended surface charts do not reconstruct the reference")

    native_vertices = np.asarray(
        arrays["foot_native_reference_vertices"], dtype=np.float64
    )
    native_faces = np.asarray(arrays["foot_native_faces"], dtype=np.int64)
    parents = np.asarray(arrays["foot_dense_face_parent_indices"], dtype=np.int64)
    if (
        native_vertices.shape != (266, 3)
        or native_faces.shape != (515, 3)
        or parents.shape != (DENSE_FACE_COUNT,)
        or not np.array_equal(vertices[:266], native_vertices)
        or not np.array_equal(faces[:DENSE_FACE_COUNT], faces[foot_faces])
        or np.any(parents < 0)
        or np.any(parents >= len(native_faces))
    ):
        raise ValueError("extended reference lost native SUPR foot provenance")

    landmark_names = np.asarray(arrays["landmark_names"])
    landmark_ids = np.asarray(
        arrays["landmark_primary_vertex_indices"], dtype=np.int64
    )
    landmark_points = np.asarray(
        arrays["landmark_reference_positions"], dtype=np.float64
    )
    if (
        landmark_names.ndim != 1
        or landmark_ids.shape != (len(landmark_names),)
        or landmark_points.shape != (len(landmark_names), 3)
        or np.any(landmark_ids < 0)
        or np.any(landmark_ids >= len(vertices))
        or not np.isfinite(landmark_points).all()
    ):
        raise ValueError("extended anatomical landmarks are invalid")

    geometry_digest = _array_digest(vertices, faces)
    topology_digest = _topology_digest(faces)
    digests = payload.get("digests", {})
    counts = payload.get("counts", {})
    if (
        digests.get("canonical_geometry_sha256") != geometry_digest
        or digests.get("topology_sha256") != topology_digest
        or counts.get("vertices") != EXTENDED_VERTEX_COUNT
        or counts.get("faces") != EXTENDED_FACE_COUNT
        or payload.get("boundaries", {}).get("knee_loop") != knee_loop.tolist()
    ):
        raise ValueError("extended anatomical JSON disagrees with its NPZ")

    saved_mesh = load_triangle_mesh(ply_path)
    storage_tolerance = 8.0 * float(
        np.spacing(np.float32(max(1.0, float(np.max(np.abs(vertices))))))
    )
    if (
        not np.array_equal(saved_mesh.faces, faces)
        or not np.allclose(
            saved_mesh.vertices, vertices, atol=storage_tolerance, rtol=0.0
        )
    ):
        raise ValueError("neutral foot-and-lower-leg PLY disagrees with the NPZ")

    return _ExtendedReference(
        root=source,
        vertices=vertices,
        faces=faces,
        foot_vertex_indices=foot_vertices,
        foot_face_indices=foot_faces,
        lower_leg_vertex_indices=leg_vertices,
        lower_leg_face_indices=leg_faces,
        ankle_transition_face_indices=transition_faces,
        knee_loop=knee_loop,
        ankle_correspondence=correspondence,
        component_face_labels=component_faces,
        native_foot_vertices=native_vertices,
        native_foot_faces=native_faces,
        dense_foot_face_parent_indices=parents,
        geometry_digest=geometry_digest,
        topology_digest=topology_digest,
        metadata=payload,
    )


def load_fitted_extended_surface(
    extended_surface_root: str | Path,
    shoe_name: str,
    *,
    extended_reference: _ExtendedReference | None = None,
) -> tuple[TriangleMesh, dict[str, Any]]:
    """Load one fitted surface using the shared Checkpoint 10-A contract."""

    root = Path(extended_surface_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    if (
        not shoe_name
        or Path(shoe_name).name != shoe_name
        or shoe_name in {".", "..", "reference"}
    ):
        raise ValueError(f"invalid shoe directory name: {shoe_name!r}")
    reference = extended_reference or _load_extended_reference(root)
    if reference.root != root:
        raise ValueError("extended reference belongs to another surface root")

    directory = (root / shoe_name).resolve(strict=True)
    if directory.parent != root or not directory.is_dir():
        raise ValueError(f"{shoe_name}: input must be directly inside the surface root")
    json_path = directory / "extended_anatomical_surface.json"
    mesh_path = directory / "foot_lower_leg.ply"
    for path in (json_path, mesh_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = _read_json_object(json_path)
    if (
        metadata.get("schema_version") != 1
        or metadata.get("stage")
        != "extended_canonical_supr_anatomical_surface"
        or metadata.get("shoe_profile") != "normal"
        or metadata.get("shoe_name") != shoe_name
    ):
        raise ValueError(f"{shoe_name}: unsupported extended anatomical surface record")
    topology = metadata.get("topology", {})
    geometry = metadata.get("geometry", {})
    if (
        topology.get("vertex_count") != EXTENDED_VERTEX_COUNT
        or topology.get("face_count") != EXTENDED_FACE_COUNT
        or topology.get("topology_sha256") != reference.topology_digest
        or not topology.get("shared_vertex_and_face_ids")
    ):
        raise ValueError(f"{shoe_name}: extended anatomical topology metadata is invalid")

    fitted = load_triangle_mesh(mesh_path)
    if fitted.vertices.shape != (EXTENDED_VERTEX_COUNT, 3) or fitted.faces.shape != (
        EXTENDED_FACE_COUNT,
        3,
    ):
        raise ValueError(f"{shoe_name}: fitted extended anatomy has invalid dimensions")
    if not np.array_equal(fitted.faces, reference.faces):
        raise ValueError(f"{shoe_name}: fitted extended anatomy changed shared topology")
    fitted_digest = _array_digest(fitted.vertices, fitted.faces)
    if geometry.get("geometry_sha256") != fitted_digest:
        raise ValueError(f"{shoe_name}: fitted surface digest does not match its JSON")
    return fitted, metadata


def build_outer_envelope() -> TriangleMesh:
    """Return the frozen subdivision-3 fourth-order superellipsoid."""

    sphere = trimesh.creation.icosphere(subdivisions=ENVELOPE_SUBDIVISIONS, radius=1.0)
    directions = np.asarray(sphere.vertices, dtype=np.float64)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    denominator = np.sum(
        np.abs(directions / ENVELOPE_RADII[None, :]) ** ENVELOPE_POWER,
        axis=1,
    )
    scale = denominator ** (-1.0 / ENVELOPE_POWER)
    vertices = ENVELOPE_CENTER[None, :] + directions * scale[:, None]
    faces = np.asarray(sphere.faces, dtype=np.int64)
    mesh = TriangleMesh(vertices, faces)
    equation = np.sum(
        np.abs((mesh.vertices - ENVELOPE_CENTER) / ENVELOPE_RADII)
        ** ENVELOPE_POWER,
        axis=1,
    )
    if not np.allclose(equation, 1.0, atol=2.0e-14, rtol=0.0):
        raise RuntimeError("outer-envelope vertices do not lie on the superellipsoid")
    shell = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False, validate=False)
    if not shell.is_watertight or not shell.is_winding_consistent:
        raise RuntimeError("outer envelope is not a consistently wound closed surface")
    if shell.volume < 0.0:
        mesh = TriangleMesh(mesh.vertices, mesh.faces[:, (0, 2, 1)])
    return mesh


def _close_truncation(
    vertices: np.ndarray,
    faces: np.ndarray,
    boundary_loop: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    cap_index = len(vertices)
    cap_center = vertices[boundary_loop].mean(axis=0)
    cap_faces = np.asarray(
        [
            (
                boundary_loop[(index + 1) % len(boundary_loop)],
                boundary_loop[index],
                cap_index,
            )
            for index in range(len(boundary_loop))
        ],
        dtype=np.int64,
    )
    closed_vertices = np.vstack((vertices, cap_center))
    closed_faces = np.vstack((faces, cap_faces))
    shell = trimesh.Trimesh(
        closed_vertices, closed_faces, process=False, validate=False
    )
    if not shell.is_watertight or not shell.is_winding_consistent:
        raise ValueError("centroid fan did not close the anatomical surface")
    if shell.volume <= 0.0:
        raise ValueError("closed anatomical surface does not have outward face winding")
    return closed_vertices, cap_faces, cap_index


def _surface_coordinates(
    points: np.ndarray,
    surface_vertices: np.ndarray,
    surface_faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    scale = max(float(np.max(np.abs(surface_vertices))), 1.0)
    tolerance = 512.0 * np.finfo(np.float64).eps * scale
    closest, distances, face_indices = _closest_points_on_triangles(
        np.asarray(points, dtype=np.float64),
        np.asarray(surface_vertices, dtype=np.float64)[surface_faces],
        np.arange(len(surface_faces), dtype=np.int64),
        tolerance,
    )
    triangles = np.asarray(surface_vertices, dtype=np.float64)[
        np.asarray(surface_faces, dtype=np.int64)[face_indices]
    ]
    barycentric = trimesh.triangles.points_to_barycentric(triangles, closest)
    barycentric[np.abs(barycentric) <= tolerance] = 0.0
    barycentric = np.maximum(barycentric, 0.0)
    barycentric /= barycentric.sum(axis=1)[:, None]
    reconstructed = np.einsum("ni,nij->nj", barycentric, triangles)
    if (
        not np.isfinite(barycentric).all()
        or np.any(barycentric < -tolerance)
        or not np.allclose(barycentric.sum(axis=1), 1.0, atol=1.0e-12)
        or not np.allclose(reconstructed, closest, atol=8.0 * tolerance, rtol=0.0)
    ):
        raise RuntimeError("surface correspondence is not barycentrically reversible")
    return closest, distances, face_indices, barycentric


def _self_intersection_pairs(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    triangles = np.asarray(vertices, dtype=np.float64)[faces]
    indices = np.arange(len(faces), dtype=np.int64)
    scale = max(float(np.max(np.abs(vertices))), 1.0)
    pairs = _find_collision_pairs(
        triangles,
        indices,
        triangles,
        indices,
        512.0 * np.finfo(np.float64).eps * scale,
    )
    pairs = pairs[pairs[:, 0] < pairs[:, 1]]
    if len(pairs) == 0:
        return pairs
    disjoint = np.asarray(
        [
            not np.intersect1d(faces[first], faces[second], assume_unique=False).size
            for first, second in pairs
        ],
        dtype=bool,
    )
    return pairs[disjoint]


def _expanded_face_region(
    faces: np.ndarray,
    seed_face_indices: np.ndarray,
    rings: int,
) -> np.ndarray:
    selected = set(np.asarray(seed_face_indices, dtype=np.int64).tolist())
    vertex_faces: list[list[int]] = [[] for _ in range(int(np.max(faces)) + 1)]
    for face_index, face in enumerate(np.asarray(faces, dtype=np.int64)):
        for vertex_index in face:
            vertex_faces[int(vertex_index)].append(face_index)
    for _ in range(rings):
        vertices = np.unique(faces[np.asarray(sorted(selected), dtype=np.int64)])
        selected.update(
            face_index
            for vertex_index in vertices
            for face_index in vertex_faces[int(vertex_index)]
        )
    return np.asarray(sorted(selected), dtype=np.int64)


def _bridge_faces(first_loop: np.ndarray, second_loop: np.ndarray) -> np.ndarray:
    result: list[tuple[int, int, int]] = []
    for index in range(len(first_loop)):
        following = (index + 1) % len(first_loop)
        result.append(
            (
                int(first_loop[index]),
                int(second_loop[index]),
                int(first_loop[following]),
            )
        )
        result.append(
            (
                int(first_loop[following]),
                int(second_loop[index]),
                int(second_loop[following]),
            )
        )
    return np.asarray(result, dtype=np.int64)


def _repair_native_foot(
    reference: _ExtendedReference,
) -> tuple[TriangleMesh, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    try:
        from pymeshfix import PyTMesh
        import pymeshfix
    except ImportError as error:
        raise RuntimeError(
            "Checkpoint 10-B requires the 'volume' extra with pymeshfix==0.18.1"
        ) from error
    if pymeshfix.__version__ != "0.18.1":
        raise RuntimeError(
            f"Checkpoint 10-B requires pymeshfix 0.18.1, found {pymeshfix.__version__}"
        )

    native_loop = _directed_boundary_loop(reference.native_foot_faces)
    closed_vertices, cap_faces, _ = _close_truncation(
        reference.native_foot_vertices,
        reference.native_foot_faces,
        native_loop,
    )
    closed_faces = np.vstack((reference.native_foot_faces, cap_faces))
    intersections = _self_intersection_pairs(closed_vertices, closed_faces)
    anatomical_pairs = intersections[np.all(intersections < len(reference.native_foot_faces), axis=1)]
    if len(anatomical_pairs) == 0:
        raise RuntimeError("canonical native foot unexpectedly has no repairable intersections")

    repair = PyTMesh()
    repair.set_quiet(True)
    repair.load_array(
        np.ascontiguousarray(closed_vertices, dtype=np.float64),
        np.ascontiguousarray(closed_faces, dtype=np.int32),
    )
    if not repair.strong_intersection_removal(PYMESHFIX_MAX_ITERATIONS):
        raise RuntimeError("PyMeshFix did not complete native-foot intersection removal")
    repaired_vertices, repaired_faces = repair.return_arrays()
    repaired_vertices = np.asarray(repaired_vertices, dtype=np.float64)
    repaired_faces = np.asarray(repaired_faces, dtype=np.int64)
    if (
        repaired_vertices.ndim != 2
        or repaired_vertices.shape[1:] != (3,)
        or repaired_faces.ndim != 2
        or repaired_faces.shape[1:] != (3,)
        or not np.isfinite(repaired_vertices).all()
        or np.any(repaired_faces < 0)
        or np.any(repaired_faces >= len(repaired_vertices))
    ):
        raise RuntimeError("PyMeshFix returned invalid native-foot geometry")

    _, face_distances, source_faces, _ = _surface_coordinates(
        repaired_vertices[repaired_faces].mean(axis=1),
        closed_vertices,
        closed_faces,
    )
    keep = source_faces < len(reference.native_foot_faces)
    open_faces = repaired_faces[keep]
    used = np.unique(open_faces)
    remap = np.full(len(repaired_vertices), -1, dtype=np.int64)
    remap[used] = np.arange(len(used), dtype=np.int64)
    open_mesh = TriangleMesh(repaired_vertices[used], remap[open_faces])
    repaired_loop = _directed_boundary_loop(open_mesh.faces)
    shell = trimesh.Trimesh(
        open_mesh.vertices, open_mesh.faces, process=False, validate=False
    )
    if (
        len(repaired_loop) != 15
        or not shell.is_winding_consistent
        or shell.body_count != 1
    ):
        raise RuntimeError("repaired native foot did not preserve one ankle opening")

    remaining_intersections = _self_intersection_pairs(
        open_mesh.vertices, open_mesh.faces
    )
    if len(remaining_intersections):
        raise RuntimeError("repaired native foot still self-intersects")

    repair_zone_faces = _expanded_face_region(
        reference.native_foot_faces,
        np.unique(anatomical_pairs),
        REPAIR_NEIGHBOUR_RINGS,
    )
    metadata = {
        "engine": "PyMeshFix PyTMesh",
        "version": pymeshfix.__version__,
        "operation": "strong_intersection_removal_only",
        "maximum_iterations": PYMESHFIX_MAX_ITERATIONS,
        "native_input_vertices": int(len(reference.native_foot_vertices)),
        "native_input_faces": int(len(reference.native_foot_faces)),
        "native_repaired_vertices": int(len(open_mesh.vertices)),
        "native_repaired_faces": int(len(open_mesh.faces)),
        "native_intersection_pair_count": int(len(anatomical_pairs)),
        "removed_computational_cap_face_count": int(np.count_nonzero(~keep)),
        "maximum_repaired_face_to_source_distance": float(np.max(face_distances)),
        "repair_zone_native_face_indices": repair_zone_faces.tolist(),
    }
    return open_mesh, repaired_loop, repair_zone_faces, anatomical_pairs, metadata


def _build_computational_boundary(
    reference: _ExtendedReference,
) -> _ComputationalBoundary:
    (
        repaired_native,
        _,
        repair_zone_faces,
        native_intersection_pairs,
        repair_metadata,
    ) = _repair_native_foot(reference)
    subdivision = build_supr_mesh_subdivision(
        repaired_native.faces, len(repaired_native.vertices), 2
    )
    repaired_foot = subdivision.apply_mesh(repaired_native)
    foot_loop = _directed_boundary_loop(repaired_foot.faces)
    if len(foot_loop) != 60:
        raise RuntimeError("subdivided computational foot must have a 60-point ankle")

    leg_start = int(reference.lower_leg_vertex_indices[0])
    if not np.array_equal(
        reference.lower_leg_vertex_indices,
        np.arange(leg_start, EXTENDED_VERTEX_COUNT, dtype=np.int64),
    ):
        raise ValueError("canonical lower-leg vertices are not one contiguous block")
    leg_vertices = reference.vertices[reference.lower_leg_vertex_indices]
    leg_faces = reference.faces[reference.lower_leg_face_indices] - leg_start
    leg_loop = reference.ankle_correspondence[:, 1] - leg_start
    canonical_knee_loop = reference.knee_loop - leg_start
    leg_offset = len(repaired_foot.vertices)

    candidates: list[tuple[float, int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for reversed_direction in (0, 1):
        ordered = leg_loop if reversed_direction == 0 else leg_loop[::-1]
        for shift in range(len(ordered)):
            paired = np.roll(ordered, -shift)
            bridge = _bridge_faces(foot_loop, paired + leg_offset)
            candidate_faces = np.vstack(
                (repaired_foot.faces, leg_faces + leg_offset, bridge)
            )
            try:
                knee_loop = _directed_boundary_loop(candidate_faces)
            except ValueError:
                continue
            expected_knee = canonical_knee_loop + leg_offset
            if len(knee_loop) != 68 or set(knee_loop) != set(expected_knee):
                continue
            residual = float(
                np.sum(
                    (repaired_foot.vertices[foot_loop] - leg_vertices[paired]) ** 2
                )
            )
            candidates.append(
                (residual, reversed_direction, shift, paired, bridge, knee_loop)
            )
    if not candidates:
        raise RuntimeError("no manifold computational ankle bridge was found")
    candidates.sort(key=lambda item: (item[0], item[1], item[2]))
    residual, direction, shift, _, bridge, knee_loop = candidates[0]

    joined_vertices = np.vstack((repaired_foot.vertices, leg_vertices))
    joined_faces = np.vstack(
        (repaired_foot.faces, leg_faces + leg_offset, bridge)
    )
    closed_vertices, cap_faces, _ = _close_truncation(
        joined_vertices, joined_faces, knee_loop
    )
    closed_faces = np.vstack((joined_faces, cap_faces))
    intersections = _self_intersection_pairs(closed_vertices, closed_faces)
    if len(intersections):
        raise RuntimeError("computational inner boundary still self-intersects")
    shell = trimesh.Trimesh(
        closed_vertices, closed_faces, process=False, validate=False
    )
    if (
        not shell.is_watertight
        or not shell.is_winding_consistent
        or shell.body_count != 1
        or shell.volume <= 0.0
    ):
        raise RuntimeError("computational inner boundary is not one closed solid")

    foot_count = len(repaired_foot.faces)
    leg_count = len(leg_faces)
    labels = np.empty(len(closed_faces), dtype=np.int16)
    labels[:foot_count] = BOUNDARY_FOOT_SKIN
    labels[foot_count : foot_count + leg_count] = BOUNDARY_LOWER_LEG_SKIN
    labels[foot_count + leg_count : len(joined_faces)] = BOUNDARY_ANKLE_TRANSITION
    labels[len(joined_faces) :] = BOUNDARY_KNEE_TRUNCATION
    cap_indices = np.arange(len(joined_faces), len(closed_faces), dtype=np.int64)

    dense_zone_faces = np.flatnonzero(
        np.isin(reference.dense_foot_face_parent_indices, repair_zone_faces)
    )
    repair_zone_vertices = np.unique(reference.faces[dense_zone_faces])
    return _ComputationalBoundary(
        vertices=closed_vertices,
        faces=closed_faces,
        expected_face_labels=labels,
        knee_cap_face_indices=cap_indices,
        repair_zone_canonical_vertex_indices=repair_zone_vertices,
        native_intersection_pairs=native_intersection_pairs,
        subdivision_metadata={
            **repair_metadata,
            "subdivision_levels": 2,
            "dense_repaired_foot_vertices": int(len(repaired_foot.vertices)),
            "dense_repaired_foot_faces": int(len(repaired_foot.faces)),
            "computational_ankle_bridge_faces": int(len(bridge)),
            "computational_ankle_bridge_squared_residual": residual,
            "computational_ankle_bridge_reversed": bool(direction),
            "computational_ankle_bridge_cyclic_shift": int(shift),
        },
    )


def _surface_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    triangles = vertices[faces]
    return float(
        np.sum(
            np.einsum(
                "ij,ij->i",
                triangles[:, 0],
                np.cross(triangles[:, 1], triangles[:, 2]),
            )
        )
        / 6.0
    )


def _validate_closed_inner_surface(
    vertices: np.ndarray, faces: np.ndarray, *, context: str
) -> float:
    points = np.asarray(vertices, dtype=np.float64)
    triangles = points[np.asarray(faces, dtype=np.int64)]
    twice_area = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    area_tolerance = 128.0 * np.finfo(np.float64).eps * scale**2
    if not np.isfinite(points).all() or np.any(twice_area <= area_tolerance):
        raise RuntimeError(f"{context} contains a degenerate face")
    shell = trimesh.Trimesh(points, faces, process=False, validate=False)
    signed_volume = _surface_volume(points, faces)
    if (
        not shell.is_watertight
        or not shell.is_winding_consistent
        or shell.body_count != 1
        or signed_volume <= 0.0
    ):
        raise RuntimeError(f"{context} is not one oriented closed surface")
    return signed_volume


def _canonical_face_rows(faces: np.ndarray) -> np.ndarray:
    sorted_faces = np.sort(np.asarray(faces, dtype=np.int64), axis=1)
    order = np.lexsort(
        (sorted_faces[:, 2], sorted_faces[:, 1], sorted_faces[:, 0])
    )
    return sorted_faces[order]


def _gmsh_remesh_inner_boundary(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[TriangleMesh, str]:
    """Return a conforming computational proxy close to the repaired anatomy."""

    try:
        import gmsh
    except ImportError as error:
        raise RuntimeError(
            "Checkpoint 10 requires the 'volume' extra with gmsh==4.15.2"
        ) from error
    if gmsh.__version__ != "4.15.2":
        raise RuntimeError(
            f"Checkpoint 10 requires gmsh 4.15.2, found {gmsh.__version__}"
        )

    points = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    node_tags = np.arange(1, len(points) + 1, dtype=np.int64)
    element_tags = np.arange(1, len(triangles) + 1, dtype=np.int64)

    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0.0)
        gmsh.option.setNumber("General.NumThreads", 1.0)
        gmsh.option.setNumber("Mesh.MaxNumThreads1D", 1.0)
        gmsh.option.setNumber("Mesh.MaxNumThreads2D", 1.0)
        gmsh.option.setNumber("Mesh.Algorithm", 6.0)
        gmsh.option.setNumber("Mesh.ElementOrder", 1.0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0.0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0.0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0.0)
        gmsh.option.setNumber("Mesh.MeshSizeMin", GMSH_INNER_REMESH_MIN_SIZE)
        gmsh.option.setNumber("Mesh.MeshSizeMax", GMSH_INNER_REMESH_MAX_SIZE)
        gmsh.option.setNumber("Mesh.RandomFactor", 1.0e-9)
        gmsh.model.add("canonical_anatomical_inner_proxy")
        gmsh.model.addDiscreteEntity(2, 1)
        gmsh.model.mesh.addNodes(2, 1, node_tags, points.ravel())
        gmsh.model.mesh.addElementsByType(
            1, 2, element_tags, (triangles + 1).ravel()
        )
        gmsh.model.mesh.classifySurfaces(
            math.radians(40.0),
            boundary=True,
            forReparametrization=True,
            curveAngle=math.pi,
        )
        gmsh.model.mesh.createGeometry()
        gmsh.model.mesh.generate(2)

        output_node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        output_node_tags = np.asarray(output_node_tags, dtype=np.int64)
        coordinates = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
        coordinate_by_tag = {
            int(tag): point for tag, point in zip(output_node_tags, coordinates)
        }
        face_tags: list[np.ndarray] = []
        for _, surface_tag in sorted(gmsh.model.getEntities(2)):
            _, element_nodes = gmsh.model.mesh.getElementsByType(2, surface_tag)
            values = np.asarray(element_nodes, dtype=np.int64)
            if values.size:
                face_tags.append(values.reshape(-1, 3))
        if not face_tags:
            raise RuntimeError("Gmsh produced no computational boundary triangles")
        tagged_faces = np.vstack(face_tags)
        used_tags = np.unique(tagged_faces)
        missing = [int(tag) for tag in used_tags if int(tag) not in coordinate_by_tag]
        if missing:
            raise RuntimeError(
                "Gmsh surface remeshing omitted coordinates for used nodes"
            )
        output_vertices = np.asarray(
            [coordinate_by_tag[int(tag)] for tag in used_tags], dtype=np.float64
        )
        index_by_tag = {int(tag): index for index, tag in enumerate(used_tags)}
        output_faces = np.asarray(
            [
                [index_by_tag[int(tag)] for tag in row]
                for row in tagged_faces
            ],
            dtype=np.int64,
        )
    except Exception as error:
        raise RuntimeError(f"Gmsh surface remeshing failed: {error}") from error
    finally:
        gmsh.finalize()

    shell = trimesh.Trimesh(
        output_vertices, output_faces, process=False, validate=False
    )
    if (
        not shell.is_watertight
        or not shell.is_winding_consistent
        or shell.body_count != 1
    ):
        raise RuntimeError("Gmsh computational boundary is not one closed manifold")
    if shell.volume < 0.0:
        output_faces = output_faces[:, (0, 2, 1)]
        shell = trimesh.Trimesh(
            output_vertices, output_faces, process=False, validate=False
        )
    if shell.volume <= 0.0 or len(
        _self_intersection_pairs(output_vertices, output_faces)
    ):
        raise RuntimeError("Gmsh computational boundary is not a valid solid")
    return TriangleMesh(output_vertices, output_faces), gmsh.__version__


def _gmsh_tetrahedralize(
    inner_vertices: np.ndarray,
    inner_faces: np.ndarray,
    outer_mesh: TriangleMesh,
) -> tuple[np.ndarray, np.ndarray, str]:
    try:
        import gmsh
    except ImportError as error:
        raise RuntimeError(
            "Checkpoint 10 requires the 'volume' extra with gmsh==4.15.2"
        ) from error
    if gmsh.__version__ != "4.15.2":
        raise RuntimeError(
            f"Checkpoint 10 requires gmsh 4.15.2, found {gmsh.__version__}"
        )

    inner_count = len(inner_vertices)
    outer_count = len(outer_mesh.vertices)
    inner_node_tags = np.arange(1, inner_count + 1, dtype=np.int64)
    outer_node_tags = np.arange(
        inner_count + 1, inner_count + outer_count + 1, dtype=np.int64
    )
    all_input_vertices = np.vstack((inner_vertices, outer_mesh.vertices))
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0.0)
        for name, value in GMSH_OPTIONS.items():
            gmsh.option.setNumber(name, value)
        gmsh.model.add("canonical_anatomical_volume")
        element_offset = 0
        for surface_tag, node_tags, vertices, faces in (
            (1, inner_node_tags, inner_vertices, inner_faces),
            (2, outer_node_tags, outer_mesh.vertices, outer_mesh.faces),
        ):
            gmsh.model.addDiscreteEntity(2, surface_tag)
            gmsh.model.mesh.addNodes(2, surface_tag, node_tags, vertices.ravel())
            element_tags = np.arange(
                element_offset + 1, element_offset + len(faces) + 1, dtype=np.int64
            )
            gmsh.model.mesh.addElementsByType(
                surface_tag,
                2,
                element_tags,
                (faces + int(node_tags[0])).ravel(),
            )
            element_offset += len(faces)

        # The inner proxy has already been made conforming by the dedicated
        # surface-remeshing pass.  Referencing the two discrete entities
        # directly avoids a second classification/remeshing pass and preserves
        # both supplied shells exactly.
        outer_loop = gmsh.model.geo.addSurfaceLoop([2])
        inner_loop = gmsh.model.geo.addSurfaceLoop([1])
        volume_tag = gmsh.model.geo.addVolume([outer_loop, inner_loop])
        gmsh.model.geo.synchronize()
        gmsh.model.mesh.generate(3)

        node_tags, coordinates, _ = gmsh.model.mesh.getNodes()
        node_tags = np.asarray(node_tags, dtype=np.int64)
        coordinates = np.asarray(coordinates, dtype=np.float64).reshape(-1, 3)
        coordinate_by_tag = {
            int(tag): point for tag, point in zip(node_tags, coordinates)
        }
        input_tags = np.concatenate((inner_node_tags, outer_node_tags))
        missing_input_tags = np.asarray(
            [int(tag) for tag in input_tags if int(tag) not in coordinate_by_tag],
            dtype=np.int64,
        )
        if len(missing_input_tags):
            raise RuntimeError(
                "Gmsh omitted "
                f"{len(missing_input_tags)} supplied boundary vertices "
                f"(first tags: {missing_input_tags[:8].tolist()})"
            )
        recovered_input = np.asarray(
            [coordinate_by_tag[int(tag)] for tag in input_tags], dtype=np.float64
        )
        if not np.array_equal(recovered_input, all_input_vertices):
            raise RuntimeError("Gmsh moved a supplied boundary vertex")

        _, actual_inner_nodes = gmsh.model.mesh.getElementsByType(2, 1)
        _, actual_outer_nodes = gmsh.model.mesh.getElementsByType(2, 2)
        actual_inner = np.asarray(actual_inner_nodes, dtype=np.int64).reshape(-1, 3)
        actual_outer = np.asarray(actual_outer_nodes, dtype=np.int64).reshape(-1, 3)
        expected_inner = inner_faces + 1
        expected_outer = outer_mesh.faces + int(outer_node_tags[0])
        if not np.array_equal(
            _canonical_face_rows(actual_inner), _canonical_face_rows(expected_inner)
        ):
            raise RuntimeError("Gmsh changed the supplied anatomical boundary faces")
        if not np.array_equal(
            _canonical_face_rows(actual_outer), _canonical_face_rows(expected_outer)
        ):
            raise RuntimeError("Gmsh changed the supplied outer-envelope faces")

        _, tetrahedron_nodes = gmsh.model.mesh.getElementsByType(4, volume_tag)
        tetrahedron_tags = np.asarray(tetrahedron_nodes, dtype=np.int64).reshape(-1, 4)
        if len(tetrahedron_tags) == 0:
            raise RuntimeError("Gmsh produced no first-order tetrahedra")
        remaining_tags = sorted(set(node_tags.tolist()).difference(input_tags.tolist()))
        ordered_tags = np.concatenate(
            (input_tags, np.asarray(remaining_tags, dtype=np.int64))
        )
        vertices = np.asarray(
            [coordinate_by_tag[int(tag)] for tag in ordered_tags], dtype=np.float64
        )
        index_by_tag = {int(tag): index for index, tag in enumerate(ordered_tags)}
        tetrahedra = np.asarray(
            [[index_by_tag[int(tag)] for tag in row] for row in tetrahedron_tags],
            dtype=np.int64,
        )
    except Exception as error:
        raise RuntimeError(f"Gmsh tetrahedralization failed: {error}") from error
    finally:
        gmsh.finalize()
    return vertices, tetrahedra, gmsh.__version__


def _signed_tetrahedron_volumes(
    vertices: np.ndarray, tetrahedra: np.ndarray
) -> np.ndarray:
    """Measure signed volumes without changing the supplied cell ordering."""

    points = np.asarray(vertices, dtype=np.float64)
    cells = np.asarray(tetrahedra, dtype=np.int64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError("vertices must have shape (N, 3)")
    if cells.ndim != 2 or cells.shape[1:] != (4,):
        raise ValueError("tetrahedra must have shape (M, 4)")
    if (
        not np.isfinite(points).all()
        or np.any(cells < 0)
        or np.any(cells >= len(points))
    ):
        raise ValueError("tetrahedral mesh contains invalid values")
    tetra_points = points[cells]
    return np.einsum(
        "ij,ij->i",
        tetra_points[:, 1] - tetra_points[:, 0],
        np.cross(
            tetra_points[:, 2] - tetra_points[:, 0],
            tetra_points[:, 3] - tetra_points[:, 0],
        ),
    ) / 6.0


def orient_tetrahedra_positive(
    vertices: np.ndarray, tetrahedra: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive tetrahedra and their signed volumes."""

    points = np.asarray(vertices, dtype=np.float64)
    cells = np.asarray(tetrahedra, dtype=np.int64).copy()
    volumes = _signed_tetrahedron_volumes(points, cells)
    negative = volumes < 0.0
    if np.any(negative):
        swapped = cells[negative, 1].copy()
        cells[negative, 1] = cells[negative, 2]
        cells[negative, 2] = swapped
        volumes[negative] *= -1.0
    scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    tolerance = 128.0 * np.finfo(np.float64).eps * scale**3 / 6.0
    if np.any(volumes <= tolerance):
        raise ValueError("tetrahedral mesh contains a degenerate cell")
    return cells, volumes


def _tetrahedron_faces(tetrahedra: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (
            tetrahedra[:, (0, 2, 1)],
            tetrahedra[:, (0, 1, 3)],
            tetrahedra[:, (0, 3, 2)],
            tetrahedra[:, (1, 2, 3)],
        ),
        axis=0,
    )


def _validate_volume_topology(
    tetrahedra: np.ndarray, boundary_faces: np.ndarray
) -> int:
    cell_faces = np.sort(_tetrahedron_faces(tetrahedra), axis=1)
    unique, inverse, counts = np.unique(
        cell_faces,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    if np.any(counts > 2):
        raise ValueError("tetrahedral mesh contains a non-manifold face")
    measured_boundary = unique[counts == 1]
    if not np.array_equal(
        _canonical_face_rows(measured_boundary),
        _canonical_face_rows(boundary_faces),
    ):
        raise ValueError("tetrahedral boundary does not match supplied boundary faces")

    owners = np.tile(np.arange(len(tetrahedra), dtype=np.int64), 4)
    order = np.lexsort(
        (cell_faces[:, 2], cell_faces[:, 1], cell_faces[:, 0])
    )
    sorted_owners = owners[order]
    starts = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(counts[:-1], dtype=np.int64))
    )[counts == 2]
    adjacency_owners = np.column_stack(
        (sorted_owners[starts], sorted_owners[starts + 1])
    )
    rows = np.concatenate((adjacency_owners[:, 0], adjacency_owners[:, 1]))
    columns = np.concatenate((adjacency_owners[:, 1], adjacency_owners[:, 0]))
    graph = sparse.coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(len(tetrahedra), len(tetrahedra)),
    ).tocsr()
    component_count = int(
        csgraph.connected_components(graph, directed=False, return_labels=False)
    )
    if component_count != 1:
        raise ValueError("tetrahedral mesh contains disconnected cell components")
    return component_count


def _tetrahedron_gradients(
    vertices: np.ndarray, tetrahedra: np.ndarray
) -> np.ndarray:
    matrices = np.ones((len(tetrahedra), 4, 4), dtype=np.float64)
    matrices[:, :, 1:] = vertices[tetrahedra]
    inverse = np.linalg.inv(matrices)
    return np.transpose(inverse[:, 1:, :], (0, 2, 1))


def _tetrahedral_stiffness(
    vertices: np.ndarray, tetrahedra: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, sparse.csr_matrix]:
    """Assemble the shared scalar linear-tetrahedron stiffness matrix."""

    points = np.asarray(vertices, dtype=np.float64)
    cells, volumes = orient_tetrahedra_positive(points, tetrahedra)
    gradients = _tetrahedron_gradients(points, cells)
    local = volumes[:, None, None] * np.einsum(
        "mik,mjk->mij", gradients, gradients
    )
    rows = np.repeat(cells, 4, axis=1).ravel()
    columns = np.tile(cells, (1, 4)).ravel()
    stiffness = sparse.coo_matrix(
        (local.ravel(), (rows, columns)), shape=(len(points), len(points))
    ).tocsr()
    return cells, volumes, gradients, stiffness


def solve_harmonic_r(
    vertices: np.ndarray,
    tetrahedra: np.ndarray,
    zero_vertex_indices: np.ndarray,
    one_vertex_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the linear tetrahedral FEM field with fixed zero/one boundaries."""

    points = np.asarray(vertices, dtype=np.float64)
    cells, _, gradients, stiffness = _tetrahedral_stiffness(points, tetrahedra)
    zero = np.unique(np.asarray(zero_vertex_indices, dtype=np.int64))
    one = np.unique(np.asarray(one_vertex_indices, dtype=np.int64))
    if (
        len(zero) == 0
        or len(one) == 0
        or np.any(zero < 0)
        or np.any(one < 0)
        or np.any(zero >= len(points))
        or np.any(one >= len(points))
        or np.intersect1d(zero, one).size
    ):
        raise ValueError("harmonic boundaries must be valid, nonempty and disjoint")
    boundary = np.concatenate((zero, one))
    boundary_values = np.concatenate(
        (np.zeros(len(zero), dtype=np.float64), np.ones(len(one), dtype=np.float64))
    )
    is_free = np.ones(len(points), dtype=bool)
    is_free[boundary] = False
    free = np.flatnonzero(is_free)
    if len(free) == 0:
        raise ValueError("harmonic volume contains no interior or natural-boundary nodes")
    free_matrix = stiffness[free][:, free]
    right_hand_side = -(stiffness[free][:, boundary] @ boundary_values)
    free_values = np.asarray(spsolve(free_matrix, right_hand_side), dtype=np.float64)
    values = np.empty(len(points), dtype=np.float64)
    values[boundary] = boundary_values
    values[free] = free_values
    residual_vector = free_matrix @ free_values - right_hand_side
    residual = float(
        np.linalg.norm(residual_vector) / max(np.linalg.norm(right_hand_side), 1.0)
    )
    field_gradient = np.einsum("mi,mij->mj", values[cells], gradients)
    tolerance = 5.0e-9
    if (
        not np.isfinite(values).all()
        or not np.isfinite(field_gradient).all()
        or not np.array_equal(values[zero], np.zeros(len(zero)))
        or not np.array_equal(values[one], np.ones(len(one)))
        or float(np.min(values)) < -tolerance
        or float(np.max(values)) > 1.0 + tolerance
        or residual > 1.0e-8
    ):
        raise RuntimeError("harmonic outward-coordinate solve failed validation")
    return values, field_gradient, residual


def map_volume_coordinates(
    tetrahedron_indices: np.ndarray,
    barycentric_weights: np.ndarray,
    volume_vertices: np.ndarray,
    tetrahedra: np.ndarray,
) -> np.ndarray:
    """Map tetrahedron IDs and four barycentric weights to volume points."""

    indices = np.asarray(tetrahedron_indices, dtype=np.int64)
    weights = np.asarray(barycentric_weights, dtype=np.float64)
    vertices = np.asarray(volume_vertices, dtype=np.float64)
    cells = np.asarray(tetrahedra, dtype=np.int64)
    if indices.ndim != 1 or weights.shape != (len(indices), 4):
        raise ValueError("volume coordinates must have shapes (N,) and (N, 4)")
    tolerance = 1.0e-12
    if (
        np.any(indices < 0)
        or np.any(indices >= len(cells))
        or not np.isfinite(weights).all()
        or np.any(weights < -tolerance)
        or np.any(weights > 1.0 + tolerance)
        or not np.allclose(weights.sum(axis=1), 1.0, atol=tolerance, rtol=0.0)
    ):
        raise ValueError("invalid tetrahedral barycentric coordinates")
    return np.einsum("ni,nij->nj", weights, vertices[cells[indices]])


def _summary(values: np.ndarray) -> dict[str, float]:
    data = np.asarray(values, dtype=np.float64)
    return {
        "minimum": float(np.min(data)),
        "p01": float(np.percentile(data, 1.0)),
        "p05": float(np.percentile(data, 5.0)),
        "median": float(np.median(data)),
        "p95": float(np.percentile(data, 95.0)),
        "p99": float(np.percentile(data, 99.0)),
        "maximum": float(np.max(data)),
    }


def _dense_foot_surface_resolution(
    vertices: np.ndarray, faces: np.ndarray
) -> float:
    foot_faces = np.asarray(faces, dtype=np.int64)[:DENSE_FACE_COUNT]
    edges = np.sort(
        np.concatenate(
            (
                foot_faces[:, (0, 1)],
                foot_faces[:, (1, 2)],
                foot_faces[:, (2, 0)],
            ),
            axis=0,
        ),
        axis=1,
    )
    edges = np.unique(edges, axis=0)
    resolution = float(
        np.median(
            np.linalg.norm(
                np.asarray(vertices, dtype=np.float64)[edges[:, 0]]
                - np.asarray(vertices, dtype=np.float64)[edges[:, 1]],
                axis=1,
            )
        )
    )
    if not np.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("fitted dense-foot surface resolution is invalid")
    return resolution


def build_instance_boundary_target(
    canonical_volume: CanonicalAnatomicalVolume,
    extended_reference: _ExtendedReference,
    fitted_extended_surface: TriangleMesh,
) -> InstanceBoundaryTarget:
    """Map the canonical computational boundary onto one fitted anatomy.

    The returned surface is a target for the fold-free Checkpoint 11-B solve.
    Expected foot-only intersections are diagnosed rather than repaired here.
    """

    fitted = fitted_extended_surface
    if fitted.vertices.shape != (EXTENDED_VERTEX_COUNT, 3) or fitted.faces.shape != (
        EXTENDED_FACE_COUNT,
        3,
    ):
        raise ValueError("fitted extended anatomy must use the 6,951/13,832 topology")
    if not np.array_equal(fitted.faces, extended_reference.faces):
        raise ValueError("fitted extended anatomy differs from the canonical topology")
    if canonical_volume.extended_surface_digest != extended_reference.geometry_digest:
        raise ValueError("canonical volume references another extended anatomy")

    closed_vertices, cap_faces, cap_center_index = _close_truncation(
        fitted.vertices, fitted.faces, extended_reference.knee_loop
    )
    closed_faces = np.vstack((fitted.faces, cap_faces))
    source_faces = canonical_volume.computational_to_canonical_face_indices
    source_weights = canonical_volume.computational_to_canonical_barycentric
    if (
        source_faces.shape
        != (len(canonical_volume.computational_inner_vertex_indices),)
        or source_weights.shape != (len(source_faces), 3)
        or np.any(source_faces < 0)
        or np.any(source_faces >= len(closed_faces))
    ):
        raise ValueError("canonical computational-to-anatomy mapping is invalid")
    target_vertices = np.einsum(
        "ni,nij->nj", source_weights, closed_vertices[closed_faces[source_faces]]
    )
    if not np.isfinite(target_vertices).all():
        raise RuntimeError("instance boundary target contains non-finite vertices")

    faces = canonical_volume.computational_inner_faces
    labels = canonical_volume.computational_inner_face_labels
    signed_volume = _validate_closed_inner_surface(
        target_vertices,
        faces,
        context="instance boundary target",
    )

    reverse_faces = canonical_volume.canonical_to_computational_face_indices
    reverse_weights = canonical_volume.canonical_to_computational_barycentric
    reverse_vertices = np.einsum(
        "ni,nij->nj", reverse_weights, target_vertices[faces[reverse_faces]]
    )
    reverse_distances = np.linalg.norm(reverse_vertices - fitted.vertices, axis=1)
    resolution = _dense_foot_surface_resolution(fitted.vertices, fitted.faces)
    repair_zone = np.zeros(EXTENDED_VERTEX_COUNT, dtype=bool)
    repair_zone[canonical_volume.repair_zone_canonical_vertex_indices] = True
    preserved = reverse_distances[~repair_zone]
    repaired = reverse_distances[repair_zone]
    if (
        len(preserved) == 0
        or len(repaired) == 0
        or float(np.percentile(preserved, 99.0)) > 0.5 * resolution
        or float(np.max(preserved)) > resolution
        or float(np.percentile(repaired, 99.0)) > 2.0 * resolution
        or float(np.max(repaired)) > 2.0 * resolution
    ):
        raise RuntimeError("instance boundary target exceeds correspondence fidelity limits")

    intersection_pairs = _self_intersection_pairs(target_vertices, faces)
    intersection_faces = (
        np.unique(intersection_pairs)
        if len(intersection_pairs)
        else np.empty(0, dtype=np.int64)
    )
    if len(intersection_faces) and np.any(
        labels[intersection_faces] != BOUNDARY_FOOT_SKIN
    ):
        affected = sorted(
            {
                BOUNDARY_LABEL_NAMES[int(label)]
                for label in labels[intersection_faces]
            }
        )
        raise RuntimeError(
            "instance boundary target intersects outside foot skin: "
            + ", ".join(affected)
        )

    envelope_equation = np.sum(
        np.abs((target_vertices - ENVELOPE_CENTER) / ENVELOPE_RADII)
        ** ENVELOPE_POWER,
        axis=1,
    )
    status = "ready" if len(intersection_pairs) == 0 else "ready_requires_untangling"
    computational_digest = str(
        canonical_volume.diagnostics["computational_inner_sha256"]
    )
    geometry_digest = _array_digest(target_vertices, faces, labels)
    fitted_digest = _array_digest(fitted.vertices, fitted.faces)
    component_counts = {
        BOUNDARY_LABEL_NAMES[index]: int(
            np.count_nonzero(labels[intersection_faces] == index)
        )
        for index in range(1, len(BOUNDARY_LABEL_NAMES))
    }
    diagnostics = {
        "fidelity": {
            "method": "stored bidirectional face-and-barycentric correspondence",
            "surface_resolution": resolution,
            "limits": {
                "preserved_p99": 0.5 * resolution,
                "preserved_maximum": resolution,
                "repair_zone_p99": 2.0 * resolution,
                "repair_zone_maximum": 2.0 * resolution,
            },
            "all_reverse_distance": _summary(reverse_distances),
            "preserved_reverse_distance": _summary(preserved),
            "repair_zone_reverse_distance": _summary(repaired),
            "target_vertex_reconstruction_maximum": 0.0,
        },
        "self_intersections": {
            "policy": (
                "foot-skin pairs are retained as soft target diagnostics; any "
                "ankle-transition, lower-leg, or knee-cap pair is invalid"
            ),
            "pair_count": int(len(intersection_pairs)),
            "face_count": int(len(intersection_faces)),
            "affected_face_indices": intersection_faces.tolist(),
            "affected_component_face_counts": component_counts,
            "requires_topology_preserving_untangling": bool(
                len(intersection_pairs)
            ),
        },
        "frozen_envelope": {
            "policy": "diagnostic_only_in_checkpoint_11_a",
            "equation": "sum(abs((x-center)/radii)^4)",
            "center": ENVELOPE_CENTER.tolist(),
            "radii": ENVELOPE_RADII.tolist(),
            "inside_when": "value < 1",
            "outside_vertex_count": int(
                np.count_nonzero(envelope_equation >= 1.0 - 1.0e-10)
            ),
            "maximum_equation_value": float(np.max(envelope_equation)),
        },
        "knee_truncation": {
            "method": "same ordered 68-vertex knee-loop centroid fan as reference",
            "authoritative_cap_center_index": int(cap_center_index),
            "authoritative_cap_face_count": int(len(cap_faces)),
            "computational_cap_face_count": int(
                np.count_nonzero(labels == BOUNDARY_KNEE_TRUNCATION)
            ),
            "anatomical_skin": False,
        },
    }
    return InstanceBoundaryTarget(
        vertices=target_vertices,
        faces=faces,
        face_labels=labels,
        intersecting_face_pairs=intersection_pairs,
        intersecting_face_indices=intersection_faces,
        reverse_reconstructed_vertices=reverse_vertices,
        reverse_distances=reverse_distances,
        envelope_equation=envelope_equation,
        surface_resolution=resolution,
        signed_volume=signed_volume,
        geometry_digest=geometry_digest,
        canonical_volume_topology_digest=canonical_volume.topology_digest,
        canonical_computational_boundary_digest=computational_digest,
        fitted_surface_digest=fitted_digest,
        status=status,
        diagnostics=diagnostics,
    )


def load_instance_volume_problem(
    anatomical_volume_root: str | Path,
    extended_surface_root: str | Path,
    shoe_name: str,
    *,
    canonical_volume: CanonicalAnatomicalVolume | None = None,
    extended_reference: _ExtendedReference | None = None,
) -> InstanceVolumeProblem:
    """Load and cross-validate the three Checkpoint 11-B1 inputs."""

    volume_root = Path(anatomical_volume_root).expanduser().resolve(strict=True)
    surface_root = Path(extended_surface_root).expanduser().resolve(strict=True)
    if not volume_root.is_dir() or not surface_root.is_dir():
        raise NotADirectoryError("anatomical volume and surface roots must be directories")
    volume = canonical_volume or load_canonical_anatomical_volume(volume_root)
    reference = extended_reference or _load_extended_reference(surface_root)
    if reference.root != surface_root:
        raise ValueError("extended reference belongs to another surface root")
    if volume.extended_surface_digest != reference.geometry_digest:
        raise ValueError("canonical volume references another extended anatomy")

    fitted, fitted_metadata = load_fitted_extended_surface(
        surface_root,
        shoe_name,
        extended_reference=reference,
    )
    target_directory = (volume_root / shoe_name).resolve(strict=True)
    if target_directory.parent != volume_root or not target_directory.is_dir():
        raise ValueError(f"{shoe_name}: target must be directly inside the volume root")
    json_path = target_directory / "boundary_target.json"
    npz_path = target_directory / "boundary_target.npz"
    for path in (json_path, npz_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    metadata = _read_json_object(json_path)
    if (
        metadata.get("schema_version") != 1
        or metadata.get("stage") != "fitted_anatomical_boundary_target"
        or metadata.get("shoe_name") != shoe_name
        or metadata.get("shoe_profile") != "normal"
        or metadata.get("status") not in {"ready", "ready_requires_untangling"}
    ):
        raise ValueError(f"{shoe_name}: unsupported boundary-target record")
    recorded_inputs = metadata.get("inputs", {})
    input_digests = {
        "canonical_volume_json_sha256": _file_digest(
            volume_root / "reference" / "canonical_volume.json"
        ),
        "canonical_volume_npz_sha256": _file_digest(
            volume_root / "reference" / "canonical_volume.npz"
        ),
        "extended_anatomical_surface_json_sha256": _file_digest(
            surface_root / shoe_name / "extended_anatomical_surface.json"
        ),
        "fitted_surface_file_sha256": _file_digest(
            surface_root / shoe_name / "foot_lower_leg.ply"
        ),
    }
    if any(recorded_inputs.get(name) != digest for name, digest in input_digests.items()):
        raise ValueError(f"{shoe_name}: boundary-target input file digest changed")

    required = {
        "target_inner_vertices",
        "intersecting_face_pairs",
        "intersecting_face_indices",
        "reverse_reconstructed_vertices",
        "reverse_distances",
        "envelope_equation",
    }
    with np.load(npz_path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"{shoe_name}: boundary target is missing arrays: {missing}")
        raw = {name: np.asarray(archive[name]) for name in required}
    if not np.issubdtype(raw["intersecting_face_pairs"].dtype, np.integer) or not np.issubdtype(
        raw["intersecting_face_indices"].dtype, np.integer
    ):
        raise ValueError(f"{shoe_name}: intersection arrays must contain integers")

    inner_count = len(volume.computational_inner_vertex_indices)
    faces = volume.computational_inner_faces
    labels = volume.computational_inner_face_labels
    target_vertices = np.asarray(raw["target_inner_vertices"], dtype=np.float64)
    pairs = np.asarray(raw["intersecting_face_pairs"], dtype=np.int64)
    intersection_faces = np.asarray(
        raw["intersecting_face_indices"], dtype=np.int64
    )
    reverse_vertices = np.asarray(
        raw["reverse_reconstructed_vertices"], dtype=np.float64
    )
    reverse_distances = np.asarray(raw["reverse_distances"], dtype=np.float64)
    envelope_equation = np.asarray(raw["envelope_equation"], dtype=np.float64)
    if (
        target_vertices.shape != (inner_count, 3)
        or pairs.ndim != 2
        or pairs.shape[1:] != (2,)
        or intersection_faces.ndim != 1
        or reverse_vertices.shape != (EXTENDED_VERTEX_COUNT, 3)
        or reverse_distances.shape != (EXTENDED_VERTEX_COUNT,)
        or envelope_equation.shape != (inner_count,)
        or not np.isfinite(target_vertices).all()
        or not np.isfinite(reverse_vertices).all()
        or not np.isfinite(reverse_distances).all()
        or not np.isfinite(envelope_equation).all()
        or np.any(reverse_distances < 0.0)
        or np.any(pairs < 0)
        or np.any(pairs >= len(faces))
        or np.any(intersection_faces < 0)
        or np.any(intersection_faces >= len(faces))
    ):
        raise ValueError(f"{shoe_name}: boundary-target arrays are invalid")
    expected_intersection_faces = (
        np.unique(pairs)
        if len(pairs)
        else np.empty(0, dtype=np.int64)
    )
    if (
        not np.array_equal(pairs, np.unique(pairs, axis=0))
        or (len(pairs) and np.any(pairs[:, 0] >= pairs[:, 1]))
        or not np.array_equal(intersection_faces, expected_intersection_faces)
        or (
            len(intersection_faces)
            and np.any(labels[intersection_faces] != BOUNDARY_FOOT_SKIN)
        )
    ):
        raise ValueError(f"{shoe_name}: recorded target intersections are invalid")
    if any(
        np.intersect1d(faces[first], faces[second]).size
        for first, second in pairs
    ):
        raise ValueError(f"{shoe_name}: adjacent faces were recorded as intersections")

    status = "ready" if len(pairs) == 0 else "ready_requires_untangling"
    counts = metadata.get("counts", {})
    correspondence = metadata.get("correspondence", {})
    recorded_intersections = metadata.get("self_intersections", {})
    if (
        metadata.get("status") != status
        or counts.get("target_vertices") != inner_count
        or counts.get("target_faces") != len(faces)
        or counts.get("intersecting_face_pairs") != len(pairs)
        or counts.get("intersecting_faces") != len(intersection_faces)
        or recorded_intersections.get("pair_count") != len(pairs)
        or recorded_intersections.get("face_count") != len(intersection_faces)
        or recorded_intersections.get("affected_face_indices")
        != intersection_faces.tolist()
        or bool(
            recorded_intersections.get(
                "requires_topology_preserving_untangling"
            )
        )
        != bool(len(pairs))
        or not correspondence.get("vertex_order_preserved")
        or not correspondence.get("face_topology_preserved")
    ):
        raise ValueError(f"{shoe_name}: boundary-target metadata counts are invalid")

    digests = metadata.get("digests", {})
    computational_digest = str(volume.diagnostics["computational_inner_sha256"])
    fitted_digest = _array_digest(fitted.vertices, fitted.faces)
    geometry_digest = _array_digest(target_vertices, faces, labels)
    geometry = metadata.get("geometry", {})
    if (
        digests.get("canonical_volume_topology_sha256") != volume.topology_digest
        or digests.get("canonical_computational_boundary_sha256")
        != computational_digest
        or digests.get("fitted_extended_surface_sha256") != fitted_digest
        or geometry.get("geometry_sha256") != geometry_digest
    ):
        raise ValueError(f"{shoe_name}: boundary-target digest validation failed")

    expected_reverse = np.einsum(
        "ni,nij->nj",
        volume.canonical_to_computational_barycentric,
        target_vertices[
            faces[volume.canonical_to_computational_face_indices]
        ],
    )
    expected_reverse_distances = np.linalg.norm(
        expected_reverse - fitted.vertices, axis=1
    )
    if (
        not np.allclose(reverse_vertices, expected_reverse, atol=1.0e-12, rtol=0.0)
        or not np.allclose(
            reverse_distances,
            expected_reverse_distances,
            atol=1.0e-12,
            rtol=0.0,
        )
    ):
        raise ValueError(f"{shoe_name}: saved reverse correspondence is inconsistent")
    repair_zone = np.zeros(EXTENDED_VERTEX_COUNT, dtype=bool)
    repair_zone[volume.repair_zone_canonical_vertex_indices] = True
    preserved_distances = expected_reverse_distances[~repair_zone]
    repaired_distances = expected_reverse_distances[repair_zone]
    resolution = _dense_foot_surface_resolution(fitted.vertices, fitted.faces)
    if (
        float(np.percentile(preserved_distances, 99.0)) > 0.5 * resolution
        or float(np.max(preserved_distances)) > resolution
        or float(np.percentile(repaired_distances, 99.0)) > 2.0 * resolution
        or float(np.max(repaired_distances)) > 2.0 * resolution
    ):
        raise ValueError(f"{shoe_name}: boundary target exceeds fidelity limits")

    expected_envelope = np.sum(
        np.abs((target_vertices - ENVELOPE_CENTER) / ENVELOPE_RADII)
        ** ENVELOPE_POWER,
        axis=1,
    )
    outside_count = int(np.count_nonzero(expected_envelope >= 1.0 - 1.0e-10))
    recorded_envelope = metadata.get("frozen_envelope", {})
    if (
        not np.allclose(
            envelope_equation, expected_envelope, atol=1.0e-12, rtol=0.0
        )
        or counts.get("envelope_outside_vertices") != outside_count
        or recorded_envelope.get("outside_vertex_count") != outside_count
        or not np.isclose(
            float(recorded_envelope.get("maximum_equation_value", np.nan)),
            float(np.max(expected_envelope)),
            atol=1.0e-12,
            rtol=0.0,
        )
        or outside_count
    ):
        raise ValueError(f"{shoe_name}: boundary target is outside the frozen envelope")

    signed_volume = _validate_closed_inner_surface(
        target_vertices,
        faces,
        context=f"{shoe_name} boundary target",
    )
    recorded_bounds = np.asarray(geometry.get("bounds"), dtype=np.float64)
    if (
        recorded_bounds.shape != (2, 3)
        or not np.allclose(
            recorded_bounds,
            np.stack((target_vertices.min(axis=0), target_vertices.max(axis=0))),
            atol=1.0e-12,
            rtol=0.0,
        )
        or not np.isclose(
            float(geometry.get("signed_volume", np.nan)),
            signed_volume,
            atol=1.0e-12,
            rtol=0.0,
        )
        or not np.isclose(
            float(geometry.get("surface_resolution", np.nan)),
            resolution,
            atol=1.0e-12,
            rtol=0.0,
        )
    ):
        raise ValueError(f"{shoe_name}: boundary-target geometry metadata is invalid")

    diagnostic_names = (
        "fidelity",
        "self_intersections",
        "frozen_envelope",
        "knee_truncation",
    )
    if any(not isinstance(metadata.get(name), dict) for name in diagnostic_names):
        raise ValueError(f"{shoe_name}: boundary-target diagnostics are invalid")
    target = InstanceBoundaryTarget(
        vertices=target_vertices,
        faces=faces,
        face_labels=labels,
        intersecting_face_pairs=pairs,
        intersecting_face_indices=intersection_faces,
        reverse_reconstructed_vertices=reverse_vertices,
        reverse_distances=reverse_distances,
        envelope_equation=envelope_equation,
        surface_resolution=resolution,
        signed_volume=signed_volume,
        geometry_digest=geometry_digest,
        canonical_volume_topology_digest=volume.topology_digest,
        canonical_computational_boundary_digest=computational_digest,
        fitted_surface_digest=fitted_digest,
        status=status,
        diagnostics={name: metadata[name] for name in diagnostic_names},
    )
    return InstanceVolumeProblem(
        shoe_name=shoe_name,
        canonical_volume=volume,
        boundary_target=target,
        fitted_surface=fitted,
        anatomical_volume_root=volume_root,
        extended_surface_root=surface_root,
        boundary_target_metadata=metadata,
        fitted_surface_metadata=fitted_metadata,
    )


def _build_instance_deformation_system(
    canonical_volume: CanonicalAnatomicalVolume,
) -> _InstanceDeformationSystem:
    """Factor the canonical Dirichlet system once for one or more shoes."""

    volume = canonical_volume
    cells, volumes, _, stiffness = _tetrahedral_stiffness(
        volume.volume_vertices, volume.tetrahedra
    )
    if (
        not np.array_equal(cells, volume.tetrahedra)
        or not np.allclose(
            volumes,
            volume.tetrahedron_signed_volumes,
            atol=1.0e-15,
            rtol=1.0e-12,
        )
    ):
        raise ValueError("canonical tetrahedra are not stored with positive orientation")
    boundary = np.concatenate(
        (volume.computational_inner_vertex_indices, volume.outer_vertex_indices)
    )
    if len(np.unique(boundary)) != len(boundary):
        raise ValueError("instance deformation boundaries overlap")
    is_free = np.ones(len(volume.volume_vertices), dtype=bool)
    is_free[boundary] = False
    free = np.flatnonzero(is_free)
    if len(free) == 0:
        raise ValueError("instance deformation contains no interior vertices")
    free_matrix = stiffness[free][:, free].tocsc()
    return _InstanceDeformationSystem(
        topology_digest=volume.topology_digest,
        boundary_vertex_indices=boundary,
        free_vertex_indices=free,
        free_matrix=free_matrix,
        free_to_boundary=stiffness[free][:, boundary].tocsr(),
        factorization=splu(free_matrix),
    )


def _solve_smooth_instance_displacement(
    problem: InstanceVolumeProblem,
    system: _InstanceDeformationSystem,
) -> tuple[np.ndarray, float]:
    volume = problem.canonical_volume
    if system.topology_digest != volume.topology_digest:
        raise ValueError("deformation system belongs to another canonical volume")
    expected_boundary = np.concatenate(
        (volume.computational_inner_vertex_indices, volume.outer_vertex_indices)
    )
    if not np.array_equal(system.boundary_vertex_indices, expected_boundary):
        raise ValueError("deformation system uses different boundary vertex IDs")

    inner = volume.computational_inner_vertex_indices
    outer = volume.outer_vertex_indices
    boundary_displacement = np.vstack(
        (
            problem.boundary_target.vertices - volume.volume_vertices[inner],
            np.zeros((len(outer), 3), dtype=np.float64),
        )
    )
    right_hand_side = -(system.free_to_boundary @ boundary_displacement)
    free_displacement = np.asarray(
        system.factorization.solve(np.asarray(right_hand_side)), dtype=np.float64
    )
    displacement = np.zeros_like(volume.volume_vertices)
    displacement[system.boundary_vertex_indices] = boundary_displacement
    displacement[system.free_vertex_indices] = free_displacement
    if not np.isfinite(displacement).all():
        raise RuntimeError(f"{problem.shoe_name}: smooth displacement is non-finite")

    residual_vector = system.free_matrix @ free_displacement - right_hand_side
    residual = float(
        np.linalg.norm(residual_vector) / max(np.linalg.norm(right_hand_side), 1.0)
    )
    if residual > 1.0e-8:
        raise RuntimeError(
            f"{problem.shoe_name}: smooth displacement residual is too large"
        )
    return displacement, residual


def _validate_continuation_candidate(
    canonical_volume: CanonicalAnatomicalVolume,
    vertices: np.ndarray,
) -> tuple[_ContinuationValidation, np.ndarray]:
    volume = canonical_volume
    points = np.asarray(vertices, dtype=np.float64)
    if points.shape != volume.volume_vertices.shape or not np.isfinite(points).all():
        return (
            _ContinuationValidation(False, "non_finite_vertices", None, 0),
            np.full(len(volume.tetrahedra), -math.inf, dtype=np.float64),
        )
    if not np.array_equal(
        points[volume.outer_vertex_indices],
        volume.volume_vertices[volume.outer_vertex_indices],
    ):
        return (
            _ContinuationValidation(False, "outer_boundary_moved", None, 0),
            np.full(len(volume.tetrahedra), -math.inf, dtype=np.float64),
        )

    signed_volumes = _signed_tetrahedron_volumes(points, volume.tetrahedra)
    determinants = signed_volumes / volume.tetrahedron_signed_volumes
    below = int(
        np.count_nonzero(
            ~np.isfinite(determinants)
            | (determinants < INSTANCE_JACOBIAN_DETERMINANT_FLOOR)
        )
    )
    finite_determinants = determinants[np.isfinite(determinants)]
    minimum = (
        float(np.min(finite_determinants)) if len(finite_determinants) else None
    )
    if below:
        return (
            _ContinuationValidation(
                False,
                "jacobian_determinant",
                minimum,
                below,
            ),
            determinants,
        )

    inner_vertices = points[volume.computational_inner_vertex_indices]
    inner_faces = volume.computational_inner_faces
    triangles = inner_vertices[inner_faces]
    twice_area = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    area_tolerance = 128.0 * np.finfo(np.float64).eps * scale**2
    degenerate = int(np.count_nonzero(twice_area <= area_tolerance))
    if degenerate:
        return (
            _ContinuationValidation(
                False,
                "degenerate_inner_face",
                minimum,
                below,
                degenerate_inner_face_count=degenerate,
            ),
            determinants,
        )

    envelope_equation = np.sum(
        np.abs((inner_vertices - ENVELOPE_CENTER) / ENVELOPE_RADII)
        ** ENVELOPE_POWER,
        axis=1,
    )
    outside = int(np.count_nonzero(envelope_equation >= 1.0 - 1.0e-10))
    if outside:
        return (
            _ContinuationValidation(
                False,
                "outside_frozen_envelope",
                minimum,
                below,
                degenerate_inner_face_count=0,
                envelope_outside_vertex_count=outside,
            ),
            determinants,
        )

    inner_pairs = _self_intersection_pairs(inner_vertices, inner_faces)
    if len(inner_pairs):
        return (
            _ContinuationValidation(
                False,
                "inner_self_intersection",
                minimum,
                below,
                degenerate_inner_face_count=0,
                envelope_outside_vertex_count=0,
                inner_self_intersection_count=int(len(inner_pairs)),
            ),
            determinants,
        )

    outer_faces = volume.boundary_faces[
        volume.boundary_labels == BOUNDARY_OUTER_ENVELOPE
    ]
    tolerance = 512.0 * np.finfo(np.float64).eps * max(
        float(np.max(np.abs(points))), 1.0
    )
    inner_outer_pairs = _find_collision_pairs(
        triangles,
        np.arange(len(inner_faces), dtype=np.int64),
        points[outer_faces],
        np.arange(len(outer_faces), dtype=np.int64),
        tolerance,
    )
    if len(inner_outer_pairs):
        return (
            _ContinuationValidation(
                False,
                "inner_outer_intersection",
                minimum,
                below,
                degenerate_inner_face_count=0,
                envelope_outside_vertex_count=0,
                inner_self_intersection_count=0,
                inner_outer_intersection_count=int(len(inner_outer_pairs)),
            ),
            determinants,
        )
    return (
        _ContinuationValidation(
            True,
            None,
            minimum,
            0,
            degenerate_inner_face_count=0,
            envelope_outside_vertex_count=0,
            inner_self_intersection_count=0,
            inner_outer_intersection_count=0,
        ),
        determinants,
    )


def continue_instance_volume(
    problem: InstanceVolumeProblem,
    *,
    deformation_system: _InstanceDeformationSystem | None = None,
) -> InstanceVolumeContinuation:
    """Advance the smooth Checkpoint 11-B2 baseline to its last valid alpha."""

    volume = problem.canonical_volume
    system = deformation_system or _build_instance_deformation_system(volume)
    displacement, residual = _solve_smooth_instance_displacement(problem, system)
    inner = volume.computational_inner_vertex_indices
    outer = volume.outer_vertex_indices
    target = problem.boundary_target.vertices

    alpha = 0.0
    step = INSTANCE_CONTINUATION_INITIAL_STEP
    last_vertices = volume.volume_vertices.copy()
    last_determinants = np.ones(len(volume.tetrahedra), dtype=np.float64)
    accepted_alphas = [0.0]
    attempts: list[dict[str, Any]] = []
    last_failure = "none"

    if np.array_equal(target, volume.volume_vertices[inner]):
        alpha = 1.0
        accepted_alphas.append(alpha)
        attempts.append(
            _ContinuationValidation(
                True,
                None,
                1.0,
                0,
                degenerate_inner_face_count=0,
                envelope_outside_vertex_count=0,
                inner_self_intersection_count=0,
                inner_outer_intersection_count=0,
            ).to_dict(alpha, True)
        )
    else:
        while alpha < 1.0:
            trial_alpha = min(1.0, alpha + step)
            candidate = volume.volume_vertices + trial_alpha * displacement
            candidate[inner] = (
                (1.0 - trial_alpha) * volume.volume_vertices[inner]
                + trial_alpha * target
            )
            candidate[outer] = volume.volume_vertices[outer]
            validation, determinants = _validate_continuation_candidate(
                volume, candidate
            )
            attempts.append(validation.to_dict(trial_alpha, validation.accepted))
            if validation.accepted:
                alpha = trial_alpha
                last_vertices = candidate
                last_determinants = determinants
                accepted_alphas.append(alpha)
                if alpha == 1.0:
                    break
                continue

            last_failure = str(validation.failure)
            step *= 0.5
            if step < INSTANCE_CONTINUATION_MINIMUM_STEP:
                break

    reached_target = alpha == 1.0
    status = "baseline_reached_target" if reached_target else "needs_11_b3"
    stopping_reason = (
        "target_reached"
        if reached_target
        else f"minimum_step_after_{last_failure}"
    )
    target_distances = np.linalg.norm(last_vertices[inner] - target, axis=1)
    displacement_magnitudes = np.linalg.norm(
        last_vertices - volume.volume_vertices, axis=1
    )
    return InstanceVolumeContinuation(
        shoe_name=problem.shoe_name,
        volume_vertices=last_vertices,
        reached_alpha=alpha,
        status=status,
        stopping_reason=stopping_reason,
        jacobian_determinants=last_determinants,
        accepted_alphas=np.asarray(accepted_alphas, dtype=np.float64),
        attempts=tuple(attempts),
        canonical_volume_topology_digest=volume.topology_digest,
        boundary_target_geometry_digest=problem.boundary_target.geometry_digest,
        diagnostics={
            "smooth_linear_system_relative_residual": residual,
            "target_distance": _summary(target_distances),
            "vertex_displacement_magnitude": _summary(displacement_magnitudes),
            "inner_self_intersection_count": 0,
            "inner_outer_intersection_count": 0,
            "outer_boundary_exact": bool(
                np.array_equal(
                    last_vertices[outer], volume.volume_vertices[outer]
                )
            ),
        },
    )


def build_canonical_anatomical_volume(
    extended_anatomical_surface_root: str | Path,
) -> CanonicalAnatomicalVolume:
    """Build and validate the fixed foot-and-lower-leg canonical volume ``A``."""

    reference = _load_extended_reference(extended_anatomical_surface_root)
    computational = _build_computational_boundary(reference)
    outer = build_outer_envelope()
    canonical_vertices, canonical_cap_faces, _ = _close_truncation(
        reference.vertices, reference.faces, reference.knee_loop
    )
    canonical_faces = np.vstack((reference.faces, canonical_cap_faces))
    initial_intersections = _self_intersection_pairs(
        canonical_vertices, canonical_faces
    )
    if len(initial_intersections) == 0:
        raise RuntimeError(
            "authoritative anatomy unexpectedly has no self-intersection to repair"
        )
    if np.any(initial_intersections >= DENSE_FACE_COUNT):
        raise RuntimeError(
            "self-intersections outside the canonical foot cannot be repaired safely"
        )

    source_labels = np.empty(len(canonical_faces), dtype=np.int16)
    source_labels[reference.foot_face_indices] = BOUNDARY_FOOT_SKIN
    source_labels[reference.ankle_transition_face_indices] = (
        BOUNDARY_ANKLE_TRANSITION
    )
    source_labels[reference.lower_leg_face_indices] = BOUNDARY_LOWER_LEG_SKIN
    source_labels[len(reference.faces) :] = BOUNDARY_KNEE_TRUNCATION

    (
        _,
        proxy_distances,
        _,
        _,
    ) = _surface_coordinates(
        computational.vertices,
        canonical_vertices,
        canonical_faces,
    )
    (
        _,
        proxy_canonical_distances,
        _,
        _,
    ) = _surface_coordinates(
        reference.vertices,
        computational.vertices,
        computational.faces,
    )
    _, _, face_sources, _ = _surface_coordinates(
        computational.vertices[computational.faces].mean(axis=1),
        canonical_vertices,
        canonical_faces,
    )
    proxy_labels = source_labels[face_sources]
    if not np.array_equal(
        proxy_labels, computational.expected_face_labels
    ):
        raise RuntimeError(
            "nearest canonical faces disagree with computational boundary components"
        )

    foot_edges = np.sort(
        np.concatenate(
            (
                reference.faces[reference.foot_face_indices][:, (0, 1)],
                reference.faces[reference.foot_face_indices][:, (1, 2)],
                reference.faces[reference.foot_face_indices][:, (2, 0)],
            ),
            axis=0,
        ),
        axis=1,
    )
    foot_edges = np.unique(foot_edges, axis=0)
    surface_resolution = float(
        np.median(
            np.linalg.norm(
                reference.vertices[foot_edges[:, 0]]
                - reference.vertices[foot_edges[:, 1]],
                axis=1,
            )
        )
    )
    if not np.isfinite(surface_resolution) or surface_resolution <= 0.0:
        raise RuntimeError("canonical dense-foot resolution is invalid")
    repair_zone = np.zeros(len(reference.vertices), dtype=bool)
    repair_zone[computational.repair_zone_canonical_vertex_indices] = True
    proxy_preserved_distances = proxy_canonical_distances[~repair_zone]
    proxy_repaired_region_distances = proxy_canonical_distances[repair_zone]
    if (
        float(np.percentile(proxy_distances, 99.0))
        > 0.5 * surface_resolution
        or float(np.max(proxy_distances)) > surface_resolution
        or float(np.percentile(proxy_preserved_distances, 99.0))
        > 0.5 * surface_resolution
        or float(np.max(proxy_preserved_distances)) > surface_resolution
        or float(np.percentile(proxy_repaired_region_distances, 99.0))
        > 2.0 * surface_resolution
        or float(np.max(proxy_repaired_region_distances)) > 2.0 * surface_resolution
    ):
        raise RuntimeError("computational-boundary repair exceeds fidelity limits")
    storage_tolerance = 8.0 * float(
        np.spacing(
            np.float32(max(1.0, float(np.max(np.abs(reference.vertices)))))
        )
    )
    meaningful = proxy_canonical_distances > storage_tolerance
    if np.any(meaningful & ~repair_zone):
        raise RuntimeError("computational-boundary changes escaped the repair region")

    remeshed, surface_gmsh_version = _gmsh_remesh_inner_boundary(
        computational.vertices, computational.faces
    )
    (
        _,
        computational_distances,
        computational_source_faces,
        computational_barycentric,
    ) = _surface_coordinates(
        remeshed.vertices,
        canonical_vertices,
        canonical_faces,
    )
    (
        _,
        canonical_distances,
        canonical_target_faces,
        canonical_barycentric,
    ) = _surface_coordinates(
        reference.vertices,
        remeshed.vertices,
        remeshed.faces,
    )
    _, _, remeshed_face_sources, _ = _surface_coordinates(
        remeshed.vertices[remeshed.faces].mean(axis=1),
        canonical_vertices,
        canonical_faces,
    )
    computational_labels = source_labels[remeshed_face_sources]
    preserved_distances = canonical_distances[~repair_zone]
    repaired_region_distances = canonical_distances[repair_zone]
    if (
        float(np.percentile(computational_distances, 99.0))
        > 0.5 * surface_resolution
        or float(np.max(computational_distances)) > surface_resolution
        or float(np.percentile(preserved_distances, 99.0))
        > 0.5 * surface_resolution
        or float(np.max(preserved_distances)) > surface_resolution
        or float(np.percentile(repaired_region_distances, 99.0))
        > 2.0 * surface_resolution
        or float(np.max(repaired_region_distances)) > 2.0 * surface_resolution
    ):
        raise RuntimeError("Gmsh-remeshed boundary exceeds fidelity limits")
    if set(np.unique(computational_labels).tolist()) != {
        BOUNDARY_FOOT_SKIN,
        BOUNDARY_ANKLE_TRANSITION,
        BOUNDARY_LOWER_LEG_SKIN,
        BOUNDARY_KNEE_TRUNCATION,
    }:
        raise RuntimeError("Gmsh-remeshed boundary lost an anatomical component")

    normalized = np.abs(
        (remeshed.vertices - ENVELOPE_CENTER) / ENVELOPE_RADII
    )
    envelope_equation = np.sum(normalized**ENVELOPE_POWER, axis=1)
    if np.any(envelope_equation >= 1.0 - 1.0e-10):
        raise ValueError(
            "canonical foot and lower leg are not strictly inside the frozen envelope"
        )

    vertices, tetrahedra, gmsh_version = _gmsh_tetrahedralize(
        remeshed.vertices, remeshed.faces, outer
    )
    if gmsh_version != surface_gmsh_version:
        raise RuntimeError("Gmsh surface and volume versions disagree")
    tetrahedra, signed_volumes = orient_tetrahedra_positive(vertices, tetrahedra)
    inner_indices = np.arange(len(remeshed.vertices), dtype=np.int64)
    anatomical_face_mask = np.isin(
        computational_labels,
        (BOUNDARY_FOOT_SKIN, BOUNDARY_ANKLE_TRANSITION, BOUNDARY_LOWER_LEG_SKIN),
    )
    anatomical_indices = np.unique(
        remeshed.faces[anatomical_face_mask]
    )
    natural_indices = np.setdiff1d(inner_indices, anatomical_indices)
    if len(natural_indices) == 0:
        raise RuntimeError("knee truncation must retain natural boundary vertices")
    outer_indices = np.arange(
        len(remeshed.vertices),
        len(remeshed.vertices) + len(outer.vertices),
        dtype=np.int64,
    )
    outer_faces = outer.faces + int(outer_indices[0])
    boundary_faces = np.vstack((remeshed.faces, outer_faces))
    boundary_labels = np.concatenate(
        (
            computational_labels,
            np.full(
                len(outer.faces), BOUNDARY_OUTER_ENVELOPE, dtype=np.int16
            ),
        )
    )
    component_count = _validate_volume_topology(tetrahedra, boundary_faces)

    expected_volume = abs(_surface_volume(outer.vertices, outer.faces)) - abs(
        _surface_volume(remeshed.vertices, remeshed.faces)
    )
    measured_volume = float(np.sum(signed_volumes))
    if not np.isclose(measured_volume, expected_volume, atol=1.0e-9, rtol=1.0e-8):
        raise RuntimeError("tetrahedral cells do not fill exactly the intended shell")

    tetra_points = vertices[tetrahedra]
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edge_squared_sum = np.zeros(len(tetrahedra), dtype=np.float64)
    for first, second in edge_pairs:
        edge_squared_sum += np.sum(
            (tetra_points[:, first] - tetra_points[:, second]) ** 2, axis=1
        )
    quality = 12.0 * (3.0 * signed_volumes) ** (2.0 / 3.0) / edge_squared_sum
    if not np.isfinite(quality).all() or np.any(quality <= 0.0):
        raise RuntimeError("tetrahedron quality calculation produced invalid values")

    harmonic_r, harmonic_gradient, residual = solve_harmonic_r(
        vertices, tetrahedra, anatomical_indices, outer_indices
    )
    gradient_magnitude = np.linalg.norm(harmonic_gradient, axis=1)
    topology_digest = _array_digest(tetrahedra, boundary_faces, boundary_labels)
    envelope_digest = _array_digest(outer.vertices, outer.faces)
    computational_digest = _array_digest(
        remeshed.vertices, remeshed.faces, computational_labels
    )
    diagnostics = {
        "gmsh_version": gmsh_version,
        "gmsh_options": {key: value for key, value in GMSH_OPTIONS.items()},
        "computational_inner_sha256": computational_digest,
        "repair": {
            "authoritative_surface_changed": False,
            "method": (
                "repair native foot intersections, attach the unchanged canonical "
                "dense lower leg, then conformingly remesh only the computational copy"
            ),
            **computational.subdivision_metadata,
            "authoritative_intersection_pair_count": int(
                len(initial_intersections)
            ),
            "authoritative_intersecting_face_indices": np.unique(
                initial_intersections
            ).tolist(),
            "computational_intersection_pair_count": 0,
            "pre_remesh_vertices": int(len(computational.vertices)),
            "pre_remesh_faces": int(len(computational.faces)),
            "remeshed_vertices": int(len(remeshed.vertices)),
            "remeshed_faces": int(len(remeshed.faces)),
            "surface_remeshing": {
                "engine": "Gmsh Python API",
                "version": surface_gmsh_version,
                "minimum_element_size": GMSH_INNER_REMESH_MIN_SIZE,
                "maximum_element_size": GMSH_INNER_REMESH_MAX_SIZE,
                "scope": "disposable computational inner boundary only",
            },
            "repair_neighbour_rings": REPAIR_NEIGHBOUR_RINGS,
            "repair_zone_canonical_vertex_count": int(np.count_nonzero(repair_zone)),
            "surface_resolution": surface_resolution,
            "fidelity_limits": {
                "proxy_to_canonical_p99": 0.5 * surface_resolution,
                "proxy_to_canonical_maximum": surface_resolution,
                "preserved_canonical_to_proxy_p99": 0.5 * surface_resolution,
                "preserved_canonical_to_proxy_maximum": surface_resolution,
                "repair_zone_canonical_to_proxy_p99": 2.0 * surface_resolution,
                "repair_zone_canonical_to_proxy_maximum": 2.0 * surface_resolution,
            },
            "proxy_to_canonical_distance": _summary(computational_distances),
            "canonical_to_proxy_distance": _summary(canonical_distances),
            "pre_remesh_proxy_to_canonical_distance": _summary(proxy_distances),
            "pre_remesh_canonical_to_proxy_distance": _summary(
                proxy_canonical_distances
            ),
            "preserved_canonical_to_proxy_distance": _summary(
                preserved_distances
            ),
            "repair_zone_canonical_to_proxy_distance": _summary(
                repaired_region_distances
            ),
            "correspondence": (
                "two-way closest face ID plus barycentric weights; deterministic "
                "smallest-face-ID tie breaking"
            ),
        },
        "harmonic_r": {
            "linear_system_relative_residual": residual,
            "value_range": [float(np.min(harmonic_r)), float(np.max(harmonic_r))],
            "gradient_magnitude": _summary(gradient_magnitude),
            "zero_boundary_vertex_count": int(len(anatomical_indices)),
            "one_boundary_vertex_count": int(len(outer_indices)),
            "natural_knee_cap_values": harmonic_r[natural_indices].tolist(),
        },
        "quality": {
            "signed_volume": _summary(signed_volumes),
            "mean_ratio": _summary(quality),
            "total_tetrahedron_volume": measured_volume,
            "expected_shell_volume": expected_volume,
        },
        "connectivity": {
            "tetrahedron_components": component_count,
            "boundary_is_manifold": True,
            "computational_inner_is_closed": True,
            "computational_inner_components": 1,
        },
    }
    return CanonicalAnatomicalVolume(
        volume_vertices=vertices,
        tetrahedra=tetrahedra,
        boundary_faces=boundary_faces,
        boundary_labels=boundary_labels,
        harmonic_r=harmonic_r,
        harmonic_r_gradient=harmonic_gradient,
        computational_inner_vertex_indices=inner_indices,
        computational_inner_faces=remeshed.faces,
        computational_inner_face_labels=computational_labels,
        zero_boundary_vertex_indices=anatomical_indices,
        knee_cap_face_indices=np.flatnonzero(
            computational_labels == BOUNDARY_KNEE_TRUNCATION
        ),
        knee_cap_natural_vertex_indices=natural_indices,
        computational_to_canonical_face_indices=computational_source_faces,
        computational_to_canonical_barycentric=computational_barycentric,
        computational_to_canonical_distances=computational_distances,
        canonical_to_computational_face_indices=canonical_target_faces,
        canonical_to_computational_barycentric=canonical_barycentric,
        canonical_to_computational_distances=canonical_distances,
        initial_self_intersection_pairs=initial_intersections,
        repair_zone_canonical_vertex_indices=(
            computational.repair_zone_canonical_vertex_indices
        ),
        outer_vertex_indices=outer_indices,
        tetrahedron_signed_volumes=signed_volumes,
        tetrahedron_mean_ratio_quality=quality,
        topology_digest=topology_digest,
        envelope_topology_digest=envelope_digest,
        extended_surface_digest=reference.geometry_digest,
        diagnostics=diagnostics,
    )
