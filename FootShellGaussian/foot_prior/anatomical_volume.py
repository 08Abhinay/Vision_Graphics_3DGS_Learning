"""Canonical tetrahedral volume around the neutral SUPR foot and lower leg."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy import sparse
from scipy.sparse import csgraph
from scipy.sparse.linalg import spsolve
import trimesh

from .anatomy import (
    DENSE_ANKLE_VERTEX_COUNT as DENSE_ANKLE_COUNT,
    DENSE_FOOT_FACE_COUNT as DENSE_FACE_COUNT,
    DENSE_FOOT_VERTEX_COUNT as DENSE_VERTEX_COUNT,
    DenseCanonicalSuprReference as _CheckpointNineReference,
    ExtendedCanonicalSuprAnatomy as _ExtendedAnatomicalSurface,
    array_digest as _array_digest,
    build_extended_canonical_supr_anatomy as _build_extended_anatomical_surface,
    directed_boundary_loop as _directed_boundary_loop,
    load_dense_canonical_supr_reference as _load_checkpoint_nine_reference,
    topology_digest as _topology_digest,
)
from .mesh import TriangleMesh


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
    "Mesh.MeshSizeMin": 0.025,
    "Mesh.MeshSizeMax": 0.075,
    "Mesh.RandomFactor": 1.0e-9,
}


@dataclass(frozen=True)
class CanonicalAnatomicalVolume:
    """Shared foot-and-lower-leg domain and its harmonic outward coordinate."""

    volume_vertices: np.ndarray
    tetrahedra: np.ndarray
    boundary_faces: np.ndarray
    boundary_labels: np.ndarray
    harmonic_r: np.ndarray
    harmonic_r_gradient: np.ndarray
    dense_foot_to_volume_indices: np.ndarray
    lower_leg_to_volume_indices: np.ndarray
    foot_boundary_face_indices: np.ndarray
    ankle_transition_face_indices: np.ndarray
    lower_leg_boundary_face_indices: np.ndarray
    knee_cap_indices: np.ndarray
    knee_cap_vertex_index: int
    dense_knee_loop_indices: np.ndarray
    ankle_loop_correspondence: np.ndarray
    lower_leg_source_vertex_indices: np.ndarray
    lower_leg_source_face_indices: np.ndarray
    body_to_reference: np.ndarray
    outer_vertex_indices: np.ndarray
    tetrahedron_signed_volumes: np.ndarray
    tetrahedron_mean_ratio_quality: np.ndarray
    topology_digest: str
    envelope_topology_digest: str
    extended_surface_digest: str
    diagnostics: dict[str, Any]

    @property
    def inner_anatomical_mesh(self) -> TriangleMesh:
        count = len(self.dense_foot_to_volume_indices) + len(
            self.lower_leg_to_volume_indices
        )
        if not np.array_equal(
            self.dense_foot_to_volume_indices,
            np.arange(len(self.dense_foot_to_volume_indices), dtype=np.int64),
        ):
            raise RuntimeError("dense anatomical vertices are not stored first")
        anatomical = np.isin(
            self.boundary_labels,
            (BOUNDARY_FOOT_SKIN, BOUNDARY_ANKLE_TRANSITION, BOUNDARY_LOWER_LEG_SKIN),
        )
        faces = self.boundary_faces[anatomical]
        return TriangleMesh(self.volume_vertices[:count], faces)

    @property
    def outer_envelope_mesh(self) -> TriangleMesh:
        faces = self.boundary_faces[
            self.boundary_labels == BOUNDARY_OUTER_ENVELOPE
        ]
        return TriangleMesh(self.volume_vertices, faces)

    def to_dict(self) -> dict[str, Any]:
        """Return the human-readable metadata for this reference volume."""

        return {
            "schema_version": 2,
            "stage": "canonical_foot_lower_leg_anatomical_volume",
            "reference_domain": "A",
            "coordinate_convention": {
                "surface": (
                    "Checkpoint 8 native SUPR face ID plus three barycentric weights"
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
                "dense_foot_vertices": int(len(self.dense_foot_to_volume_indices)),
                "dense_foot_faces": int(len(self.foot_boundary_face_indices)),
                "lower_leg_vertices": int(len(self.lower_leg_to_volume_indices)),
                "lower_leg_faces": int(len(self.lower_leg_boundary_face_indices)),
                "ankle_transition_faces": int(
                    len(self.ankle_transition_face_indices)
                ),
                "knee_cap_faces": int(len(self.knee_cap_indices)),
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
            "lower_leg": self.diagnostics["lower_leg"],
            "ankle_attachment": self.diagnostics["ankle_attachment"],
            "knee_truncation": {
                "method": "ordered_dense_knee_loop_centroid_fan",
                "center_volume_vertex_index": int(self.knee_cap_vertex_index),
                "boundary_face_indices": self.knee_cap_indices.tolist(),
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
                    "conforming mesh with the supplied dense anatomical boundary "
                    "preserved exactly"
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
                "checkpoint9_dense_surface_sha256": self.diagnostics[
                    "checkpoint9_dense_surface_sha256"
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


def _canonical_face_rows(faces: np.ndarray) -> np.ndarray:
    sorted_faces = np.sort(np.asarray(faces, dtype=np.int64), axis=1)
    order = np.lexsort(
        (sorted_faces[:, 2], sorted_faces[:, 1], sorted_faces[:, 0])
    )
    return sorted_faces[order]


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
    inner_tag_set = set(inner_node_tags.tolist())
    outer_tag_set = set(outer_node_tags.tolist())

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

        gmsh.model.mesh.classifySurfaces(
            math.radians(40.0),
            boundary=True,
            forReparametrization=True,
            curveAngle=math.pi,
        )
        gmsh.model.mesh.createGeometry()
        grouped_surfaces: dict[str, list[int]] = {"inner": [], "outer": []}
        for _, surface_tag in gmsh.model.getEntities(2):
            tags, _, _ = gmsh.model.mesh.getNodes(
                2, surface_tag, includeBoundary=True
            )
            tag_set = set(np.asarray(tags, dtype=np.int64).tolist())
            if tag_set and tag_set.issubset(inner_tag_set):
                grouped_surfaces["inner"].append(surface_tag)
            elif tag_set and tag_set.issubset(outer_tag_set):
                grouped_surfaces["outer"].append(surface_tag)
            else:
                raise RuntimeError("Gmsh mixed the inner and outer surface nodes")
        if not grouped_surfaces["inner"] or not grouped_surfaces["outer"]:
            raise RuntimeError("Gmsh did not preserve both boundary shells")
        outer_loop = gmsh.model.geo.addSurfaceLoop(
            sorted(grouped_surfaces["outer"])
        )
        inner_loop = gmsh.model.geo.addSurfaceLoop(
            sorted(grouped_surfaces["inner"])
        )
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
        if any(int(tag) not in coordinate_by_tag for tag in input_tags):
            raise RuntimeError("Gmsh omitted a supplied boundary vertex")
        recovered_input = np.asarray(
            [coordinate_by_tag[int(tag)] for tag in input_tags], dtype=np.float64
        )
        if not np.array_equal(recovered_input, all_input_vertices):
            raise RuntimeError("Gmsh moved a supplied boundary vertex")

        actual_surface_faces: dict[str, list[np.ndarray]] = {
            "inner": [],
            "outer": [],
        }
        for name, surface_tags in grouped_surfaces.items():
            for surface_tag in surface_tags:
                _, element_nodes = gmsh.model.mesh.getElementsByType(2, surface_tag)
                values = np.asarray(element_nodes, dtype=np.int64)
                if values.size:
                    actual_surface_faces[name].append(values.reshape(-1, 3))
        actual_inner = np.vstack(actual_surface_faces["inner"])
        actual_outer = np.vstack(actual_surface_faces["outer"])
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


def orient_tetrahedra_positive(
    vertices: np.ndarray, tetrahedra: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return positive tetrahedra and their signed volumes."""

    points = np.asarray(vertices, dtype=np.float64)
    cells = np.asarray(tetrahedra, dtype=np.int64).copy()
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
    six_volume = np.einsum(
        "ij,ij->i",
        tetra_points[:, 1] - tetra_points[:, 0],
        np.cross(
            tetra_points[:, 2] - tetra_points[:, 0],
            tetra_points[:, 3] - tetra_points[:, 0],
        ),
    )
    negative = six_volume < 0.0
    if np.any(negative):
        swapped = cells[negative, 1].copy()
        cells[negative, 1] = cells[negative, 2]
        cells[negative, 2] = swapped
        six_volume[negative] *= -1.0
    scale = max(float(np.ptp(points, axis=0).max()), 1.0)
    tolerance = 128.0 * np.finfo(np.float64).eps * scale**3
    if np.any(six_volume <= tolerance):
        raise ValueError("tetrahedral mesh contains a degenerate cell")
    return cells, six_volume / 6.0


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


def solve_harmonic_r(
    vertices: np.ndarray,
    tetrahedra: np.ndarray,
    zero_vertex_indices: np.ndarray,
    one_vertex_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Solve the linear tetrahedral FEM field with fixed zero/one boundaries."""

    points = np.asarray(vertices, dtype=np.float64)
    cells, volumes = orient_tetrahedra_positive(points, tetrahedra)
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
    gradients = _tetrahedron_gradients(points, cells)
    local = volumes[:, None, None] * np.einsum(
        "mik,mjk->mij", gradients, gradients
    )
    rows = np.repeat(cells, 4, axis=1).ravel()
    columns = np.tile(cells, (1, 4)).ravel()
    stiffness = sparse.coo_matrix(
        (local.ravel(), (rows, columns)), shape=(len(points), len(points))
    ).tocsr()

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
        "maximum": float(np.max(data)),
    }


def build_canonical_anatomical_volume(
    anatomical_surface_root: str | Path,
    full_body_supr_model: str | Path,
) -> CanonicalAnatomicalVolume:
    """Build and validate the fixed foot-and-lower-leg canonical volume ``A``."""

    reference = _load_checkpoint_nine_reference(anatomical_surface_root)
    extended = _build_extended_anatomical_surface(
        reference, full_body_supr_model
    )
    outer = build_outer_envelope()
    inner_vertices, cap_faces, cap_vertex_index = _close_truncation(
        extended.vertices, extended.faces, extended.knee_loop
    )
    inner_faces = np.vstack((extended.faces, cap_faces))
    normalized = np.abs((inner_vertices - ENVELOPE_CENTER) / ENVELOPE_RADII)
    envelope_equation = np.sum(normalized**ENVELOPE_POWER, axis=1)
    if np.any(envelope_equation >= 1.0 - 1.0e-10):
        raise ValueError(
            "canonical foot and lower leg are not strictly inside the frozen envelope"
        )

    vertices, tetrahedra, gmsh_version = _gmsh_tetrahedralize(
        inner_vertices, inner_faces, outer
    )
    tetrahedra, signed_volumes = orient_tetrahedra_positive(vertices, tetrahedra)
    dense_foot_indices = extended.dense_foot_indices
    lower_leg_indices = extended.lower_leg_indices
    anatomical_indices = np.arange(len(extended.vertices), dtype=np.int64)
    outer_indices = np.arange(
        len(inner_vertices), len(inner_vertices) + len(outer.vertices), dtype=np.int64
    )
    outer_faces = outer.faces + int(outer_indices[0])
    boundary_faces = np.vstack((extended.faces, cap_faces, outer_faces))
    boundary_labels = np.empty(len(boundary_faces), dtype=np.int16)
    boundary_labels[extended.foot_face_indices] = BOUNDARY_FOOT_SKIN
    boundary_labels[extended.lower_leg_face_indices] = BOUNDARY_LOWER_LEG_SKIN
    boundary_labels[extended.bridge_face_indices] = BOUNDARY_ANKLE_TRANSITION
    cap_boundary_indices = np.arange(
        len(extended.faces), len(extended.faces) + len(cap_faces), dtype=np.int64
    )
    boundary_labels[cap_boundary_indices] = BOUNDARY_KNEE_TRUNCATION
    outer_boundary_indices = np.arange(
        len(extended.faces) + len(cap_faces), len(boundary_faces), dtype=np.int64
    )
    boundary_labels[outer_boundary_indices] = BOUNDARY_OUTER_ENVELOPE
    component_count = _validate_volume_topology(tetrahedra, boundary_faces)

    expected_volume = abs(_surface_volume(outer.vertices, outer.faces)) - abs(
        _surface_volume(inner_vertices, inner_faces)
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
    native_pairs = extended.ankle_correspondence[
        (extended.ankle_correspondence[:, 0] < 266)
        & (
            extended.ankle_correspondence[:, 1] - DENSE_VERTEX_COUNT
            < len(extended.lower_leg.mesh.vertices)
        )
    ]
    diagnostics = {
        "gmsh_version": gmsh_version,
        "gmsh_options": {key: value for key, value in GMSH_OPTIONS.items()},
        "checkpoint9_dense_surface_sha256": reference.dense_surface_digest,
        "lower_leg": extended.lower_leg_metadata,
        "ankle_attachment": {
            "method": (
                "minimum-residual consistently-wound cyclic dense-loop bridge"
            ),
            "dense_correspondence": extended.ankle_correspondence.tolist(),
            "native_correspondence": native_pairs.tolist(),
            "dense_pair_count": int(len(extended.ankle_correspondence)),
            "native_pair_count": int(len(native_pairs)),
            "bridge_face_count": int(len(extended.bridge_face_indices)),
            **extended.attachment_diagnostics,
        },
        "harmonic_r": {
            "linear_system_relative_residual": residual,
            "value_range": [float(np.min(harmonic_r)), float(np.max(harmonic_r))],
            "gradient_magnitude": _summary(gradient_magnitude),
            "zero_boundary_vertex_count": int(len(anatomical_indices)),
            "one_boundary_vertex_count": int(len(outer_indices)),
            "natural_knee_cap_center_value": float(harmonic_r[cap_vertex_index]),
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
        },
    }
    return CanonicalAnatomicalVolume(
        volume_vertices=vertices,
        tetrahedra=tetrahedra,
        boundary_faces=boundary_faces,
        boundary_labels=boundary_labels,
        harmonic_r=harmonic_r,
        harmonic_r_gradient=harmonic_gradient,
        dense_foot_to_volume_indices=dense_foot_indices,
        lower_leg_to_volume_indices=lower_leg_indices,
        foot_boundary_face_indices=extended.foot_face_indices,
        ankle_transition_face_indices=extended.bridge_face_indices,
        lower_leg_boundary_face_indices=extended.lower_leg_face_indices,
        knee_cap_indices=cap_boundary_indices,
        knee_cap_vertex_index=cap_vertex_index,
        dense_knee_loop_indices=extended.knee_loop,
        ankle_loop_correspondence=extended.ankle_correspondence,
        lower_leg_source_vertex_indices=extended.lower_leg.source_vertex_indices,
        lower_leg_source_face_indices=extended.lower_leg.source_face_indices,
        body_to_reference=extended.lower_leg.body_to_reference,
        outer_vertex_indices=outer_indices,
        tetrahedron_signed_volumes=signed_volumes,
        tetrahedron_mean_ratio_quality=quality,
        topology_digest=topology_digest,
        envelope_topology_digest=envelope_digest,
        extended_surface_digest=extended.digest,
        diagnostics=diagnostics,
    )
