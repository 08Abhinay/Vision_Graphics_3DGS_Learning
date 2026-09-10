"""Canonical anatomical labels and exact SUPR surface correspondence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .alignment import (
    CONTACT_REGION_RANGES,
    identify_supr_contact_regions,
    make_supr_to_shoe_axis_remap,
    neutral_length_scale,
    transform_points,
)
from .mesh import TriangleMesh, load_triangle_mesh
from .supr_foot import (
    SuprMeshSubdivision,
    build_supr_mesh_subdivision,
    load_neutral_supr_foot,
)
from .supr_lower_leg import (
    RIGHT_ANKLE_JOINT_INDEX,
    RIGHT_FOOT_JOINT_INDEX,
    RIGHT_KNEE_JOINT_INDEX,
    SuprLowerLeg,
    _ordered_boundary_loops,
    build_canonical_right_lower_leg,
)


JOINT_NAMES = (
    "root",
    "ankle",
    "midfoot",
    "big_toe_base",
    "big_toe_tip",
    "toe_2_base",
    "toe_2_tip",
    "toe_3_base",
    "toe_3_tip",
    "toe_4_base",
    "toe_4_tip",
    "little_toe_base",
    "little_toe_tip",
)
LONGITUDINAL_REGION_NAMES = ("ankle", "heel", "arch", "forefoot", "toes")
SURFACE_REGION_NAMES = (
    "plantar",
    "top",
    "medial",
    "lateral",
    "posterior",
    "anterior",
)
LONGITUDINAL_COLORS = np.asarray(
    (
        (145, 80, 200, 255),
        (220, 70, 70, 255),
        (240, 155, 55, 255),
        (55, 175, 95, 255),
        (55, 120, 225, 255),
    ),
    dtype=np.uint8,
)
SURFACE_COLORS = np.asarray(
    (
        (45, 185, 95, 255),
        (55, 125, 225, 255),
        (240, 145, 50, 255),
        (155, 80, 205, 255),
        (220, 70, 70, 255),
        (235, 205, 55, 255),
    ),
    dtype=np.uint8,
)

DENSE_FOOT_VERTEX_COUNT = 4_151
DENSE_FOOT_FACE_COUNT = 8_240
DENSE_ANKLE_VERTEX_COUNT = 60
EXTENDED_VERTEX_COUNT = 6_951
EXTENDED_FACE_COUNT = 13_832
EXTENDED_LONGITUDINAL_REGION_NAMES = (
    *LONGITUDINAL_REGION_NAMES,
    "lower_shaft",
    "calf",
    "upper_shaft",
)
EXTENDED_LONGITUDINAL_COLORS = np.vstack(
    (
        LONGITUDINAL_COLORS,
        np.asarray(
            (
                (55, 175, 205, 255),
                (45, 135, 180, 255),
                (35, 95, 150, 255),
            ),
            dtype=np.uint8,
        ),
    )
)
COMPONENT_REGION_NAMES = ("foot_skin", "ankle_transition", "lower_leg_skin")
COMPONENT_COLORS = np.asarray(
    (
        (45, 105, 220, 255),
        (240, 145, 50, 255),
        (55, 175, 205, 255),
    ),
    dtype=np.uint8,
)


@dataclass(frozen=True)
class CanonicalSuprAnatomy:
    """Neutral right SUPR foot expressed in the canonical reference frame."""

    raw_mesh: TriangleMesh
    reference_mesh: TriangleMesh
    raw_to_reference: np.ndarray
    reference_to_raw: np.ndarray
    raw_joints: np.ndarray
    reference_joints: np.ndarray
    ankle_boundary_vertex_indices: np.ndarray
    longitudinal_vertex_labels: np.ndarray
    longitudinal_face_labels: np.ndarray
    surface_vertex_labels: np.ndarray
    surface_face_labels: np.ndarray
    landmarks: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        """Return the compact JSON portion of the canonical reference."""

        return {
            "coordinate_convention": {
                "x": "heel_to_toe",
                "y": "positive_down_toward_plantar_surface",
                "z": "right_foot_width; medial_negative_lateral_positive",
                "origin": (
                    "rear-most neutral foot X=0; projected-area plantar center "
                    "Z=0; lowest plantar contact Y=0"
                ),
            },
            "transforms": {
                "raw_supr_to_reference": self.raw_to_reference.tolist(),
                "reference_to_raw_supr": self.reference_to_raw.tolist(),
            },
            "joints": {
                "names": list(JOINT_NAMES),
                "raw_supr": self.raw_joints.tolist(),
                "reference": self.reference_joints.tolist(),
            },
            "ankle_boundary": {
                "ordered_vertex_indices": self.ankle_boundary_vertex_indices.tolist(),
                "centroid_reference": self.reference_mesh.vertices[
                    self.ankle_boundary_vertex_indices
                ].mean(axis=0).tolist(),
            },
            "landmarks": self.landmarks,
            "regions": {
                "interpretation": (
                    "longitudinal anatomy and surface orientation are independent; "
                    "a point carries one label from each map"
                ),
                "longitudinal": _label_record(
                    LONGITUDINAL_REGION_NAMES,
                    LONGITUDINAL_COLORS,
                    self.longitudinal_vertex_labels,
                    self.longitudinal_face_labels,
                ),
                "surface": _label_record(
                    SURFACE_REGION_NAMES,
                    SURFACE_COLORS,
                    self.surface_vertex_labels,
                    self.surface_face_labels,
                ),
            },
        }


@dataclass(frozen=True)
class DenseCanonicalSuprAnatomy:
    """Shared subdiv2 reference and its exact coarse-surface charts."""

    mesh: TriangleMesh
    subdivision: SuprMeshSubdivision
    vertex_chart_face_indices: np.ndarray
    vertex_chart_barycentric: np.ndarray
    ankle_boundary_vertex_indices: np.ndarray
    longitudinal_vertex_labels: np.ndarray
    longitudinal_face_labels: np.ndarray
    surface_vertex_labels: np.ndarray
    surface_face_labels: np.ndarray

    @property
    def longitudinal_colors(self) -> np.ndarray:
        return LONGITUDINAL_COLORS[self.longitudinal_vertex_labels]

    @property
    def surface_colors(self) -> np.ndarray:
        return SURFACE_COLORS[self.surface_vertex_labels]


@dataclass(frozen=True)
class DenseCanonicalSuprReference:
    """Validated Checkpoint 9 reference loaded without rebuilding SUPR."""

    vertices: np.ndarray
    faces: np.ndarray
    ankle_loop: np.ndarray
    native_vertices: np.ndarray
    native_faces: np.ndarray
    native_ankle_loop: np.ndarray
    joint_names: tuple[str, ...]
    reference_joints: np.ndarray
    raw_supr_to_reference: np.ndarray
    reference_to_raw_supr: np.ndarray
    dense_vertex_source_indices: np.ndarray
    dense_vertex_source_weights: np.ndarray
    dense_face_parent_indices: np.ndarray
    vertex_chart_face_indices: np.ndarray
    vertex_chart_barycentric: np.ndarray
    longitudinal_vertex_labels: np.ndarray
    longitudinal_face_labels: np.ndarray
    surface_vertex_labels: np.ndarray
    surface_face_labels: np.ndarray
    foot_landmarks: dict[str, dict[str, Any]]
    dense_surface_digest: str
    topology_digest: str


@dataclass(frozen=True)
class ExtendedCanonicalSuprAnatomy:
    """Canonical dense foot and lower leg with shared anatomical maps."""

    vertices: np.ndarray
    faces: np.ndarray
    foot_face_indices: np.ndarray
    lower_leg_face_indices: np.ndarray
    bridge_face_indices: np.ndarray
    dense_foot_indices: np.ndarray
    lower_leg_indices: np.ndarray
    knee_loop: np.ndarray
    ankle_correspondence: np.ndarray
    lower_leg: SuprLowerLeg
    lower_leg_subdivision: SuprMeshSubdivision
    lower_leg_metadata: dict[str, Any]
    attachment_diagnostics: dict[str, Any]
    vertex_chart_face_indices: np.ndarray
    vertex_chart_barycentric: np.ndarray
    foot_native_chart_face_indices: np.ndarray
    foot_native_chart_barycentric: np.ndarray
    foot_native_vertices: np.ndarray
    foot_native_faces: np.ndarray
    raw_supr_to_reference: np.ndarray
    reference_to_raw_supr: np.ndarray
    foot_dense_vertex_source_indices: np.ndarray
    foot_dense_vertex_source_weights: np.ndarray
    foot_dense_face_parent_indices: np.ndarray
    longitudinal_vertex_labels: np.ndarray
    longitudinal_face_labels: np.ndarray
    surface_vertex_labels: np.ndarray
    surface_face_labels: np.ndarray
    component_vertex_labels: np.ndarray
    component_face_labels: np.ndarray
    landmarks: dict[str, dict[str, Any]]
    foot_joint_names: tuple[str, ...]
    foot_reference_joints: np.ndarray
    lower_leg_joint_names: tuple[str, ...]
    lower_leg_joint_source_indices: np.ndarray
    lower_leg_reference_joints: np.ndarray
    anatomical_frame: np.ndarray
    digest: str
    topology_digest: str

    @property
    def mesh(self) -> TriangleMesh:
        return TriangleMesh(self.vertices, self.faces)

    @property
    def longitudinal_colors(self) -> np.ndarray:
        return EXTENDED_LONGITUDINAL_COLORS[self.longitudinal_vertex_labels]

    @property
    def surface_colors(self) -> np.ndarray:
        return SURFACE_COLORS[self.surface_vertex_labels]

    @property
    def component_colors(self) -> np.ndarray:
        colors = COMPONENT_COLORS[self.component_vertex_labels].copy()
        transition_vertices = np.unique(self.faces[self.bridge_face_indices])
        colors[transition_vertices] = COMPONENT_COLORS[
            COMPONENT_REGION_NAMES.index("ankle_transition")
        ]
        return colors


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


def _load_reference_data(model_path: str | Path) -> tuple[TriangleMesh, np.ndarray]:
    source = Path(model_path).expanduser().resolve(strict=True)
    mesh = load_neutral_supr_foot(source)
    container = np.load(source, allow_pickle=True)
    payload = container.item()
    if not isinstance(payload, Mapping) or "J" not in payload:
        raise ValueError("SUPR model is missing neutral joint positions")
    joints = np.asarray(payload["J"], dtype=np.float64)
    if joints.shape != (len(JOINT_NAMES), 3) or not np.isfinite(joints).all():
        raise ValueError("SUPR neutral joints must have shape (13, 3)")
    return mesh, joints


def _translation_matrix(offset: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = np.asarray(offset, dtype=np.float64)
    return matrix


def _canonical_transform(mesh: TriangleMesh) -> np.ndarray:
    regions = identify_supr_contact_regions(mesh)
    scale = neutral_length_scale(mesh)
    scale_matrix = np.diag((scale, scale, scale, 1.0))
    linear = scale_matrix @ make_supr_to_shoe_axis_remap()
    vertices = transform_points(mesh.vertices, linear)
    triangles = vertices[mesh.faces[regions.plantar_face_indices]]
    planar = triangles[:, :, (0, 2)]
    first = planar[:, 1] - planar[:, 0]
    second = planar[:, 2] - planar[:, 0]
    areas = 0.5 * np.abs(
        first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    )
    if not np.isfinite(areas).all() or float(np.sum(areas)) <= 0.0:
        raise ValueError("neutral SUPR plantar projection has zero area")
    plantar_center_z = float(
        np.sum(areas * triangles.mean(axis=1)[:, 2]) / np.sum(areas)
    )
    plantar_y = vertices[regions.plantar_vertex_indices, 1]
    offset = np.asarray(
        [-np.min(vertices[:, 0]), -np.max(plantar_y), -plantar_center_z],
        dtype=np.float64,
    )
    return _translation_matrix(offset) @ linear


def _weighted_normals(mesh: TriangleMesh) -> tuple[np.ndarray, np.ndarray]:
    triangles = mesh.vertices[mesh.faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(crosses, axis=1)
    if np.any(lengths <= np.finfo(np.float64).eps):
        raise ValueError("SUPR mesh contains a degenerate face")
    face_normals = crosses / lengths[:, None]
    vertex_normals = np.zeros_like(mesh.vertices)
    for corner in range(3):
        np.add.at(vertex_normals, mesh.faces[:, corner], crosses)
    vertex_lengths = np.linalg.norm(vertex_normals, axis=1)
    if np.any(vertex_lengths <= np.finfo(np.float64).eps):
        raise ValueError("SUPR mesh contains a vertex with no surface normal")
    return vertex_normals / vertex_lengths[:, None], face_normals


def _longitudinal_labels(
    points: np.ndarray,
    reference_joints: np.ndarray,
) -> np.ndarray:
    length = float(np.ptp(points[:, 0]))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("canonical SUPR surface has zero length")
    u = (points[:, 0] - np.min(points[:, 0])) / length
    labels = np.full(len(points), -1, dtype=np.int16)
    ankle = (
        (points[:, 1] < reference_joints[1, 1])
        & (points[:, 0] <= reference_joints[2, 0])
    )
    labels[ankle] = LONGITUDINAL_REGION_NAMES.index("ankle")
    remaining = ~ankle
    ranges = CONTACT_REGION_RANGES
    labels[remaining & (u <= ranges["heel"][1])] = LONGITUDINAL_REGION_NAMES.index("heel")
    labels[remaining & (u > ranges["arch"][0]) & (u < ranges["arch"][1])] = LONGITUDINAL_REGION_NAMES.index("arch")
    labels[remaining & (u >= ranges["forefoot"][0]) & (u <= ranges["forefoot"][1])] = LONGITUDINAL_REGION_NAMES.index("forefoot")
    labels[remaining & (u > ranges["toes"][0])] = LONGITUDINAL_REGION_NAMES.index("toes")
    if np.any(labels < 0):
        raise RuntimeError("canonical longitudinal regions are not exhaustive")
    return labels


def _surface_labels(normals: np.ndarray) -> np.ndarray:
    values = np.asarray(normals, dtype=np.float64)
    dominant = np.argmax(np.abs(values), axis=1)
    labels = np.empty(len(values), dtype=np.int16)
    labels[(dominant == 1) & (values[:, 1] >= 0.0)] = SURFACE_REGION_NAMES.index("plantar")
    labels[(dominant == 1) & (values[:, 1] < 0.0)] = SURFACE_REGION_NAMES.index("top")
    labels[(dominant == 2) & (values[:, 2] < 0.0)] = SURFACE_REGION_NAMES.index("medial")
    labels[(dominant == 2) & (values[:, 2] >= 0.0)] = SURFACE_REGION_NAMES.index("lateral")
    labels[(dominant == 0) & (values[:, 0] < 0.0)] = SURFACE_REGION_NAMES.index("posterior")
    labels[(dominant == 0) & (values[:, 0] >= 0.0)] = SURFACE_REGION_NAMES.index("anterior")
    return labels


def _region_labels(
    mesh: TriangleMesh,
    reference_joints: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertex_normals, face_normals = _weighted_normals(mesh)
    face_centroids = mesh.vertices[mesh.faces].mean(axis=1)
    return (
        _longitudinal_labels(mesh.vertices, reference_joints),
        _longitudinal_labels(face_centroids, reference_joints),
        _surface_labels(vertex_normals),
        _surface_labels(face_normals),
    )


def _ordered_boundary_loop(faces: np.ndarray) -> np.ndarray:
    edges = np.sort(
        np.concatenate(
            (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0
        ),
        axis=1,
    )
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("SUPR surface contains a non-manifold edge")
    boundary = unique[counts == 1]
    if len(boundary) == 0:
        raise ValueError("SUPR surface has no ankle boundary")
    adjacency: dict[int, list[int]] = {}
    for first, second in boundary:
        adjacency.setdefault(int(first), []).append(int(second))
        adjacency.setdefault(int(second), []).append(int(first))
    if any(len(neighbours) != 2 for neighbours in adjacency.values()):
        raise ValueError("SUPR ankle boundary is not one or more closed loops")
    start = min(adjacency)
    previous = -1
    current = start
    ordered: list[int] = []
    while True:
        ordered.append(current)
        neighbours = sorted(adjacency[current])
        following = neighbours[0] if neighbours[0] != previous else neighbours[1]
        previous, current = current, following
        if current == start:
            break
        if current in ordered or len(ordered) > len(adjacency):
            raise ValueError("SUPR ankle boundary does not form one simple loop")
    if len(ordered) != len(adjacency):
        raise ValueError("SUPR surface contains multiple boundary loops")
    return np.asarray(ordered, dtype=np.int64)


def _extreme_landmark(
    mesh: TriangleMesh,
    candidates: np.ndarray,
    axis: int,
    maximum: bool,
) -> dict[str, Any]:
    indices = np.asarray(candidates, dtype=np.int64)
    values = mesh.vertices[indices, axis]
    extreme = float(np.max(values) if maximum else np.min(values))
    tolerance = 64.0 * np.finfo(np.float64).eps * max(1.0, abs(extreme))
    tied = indices[np.abs(values - extreme) <= tolerance]
    return {
        "vertex_indices": tied.tolist(),
        "primary_vertex_index": int(np.min(tied)),
        "point_reference": mesh.vertices[tied].mean(axis=0).tolist(),
    }


def _landmarks(
    mesh: TriangleMesh,
    longitudinal_labels: np.ndarray,
    ankle_boundary: np.ndarray,
    plantar_vertex_indices: np.ndarray,
) -> dict[str, dict[str, Any]]:
    all_vertices = np.arange(len(mesh.vertices), dtype=np.int64)
    plantar = np.asarray(plantar_vertex_indices, dtype=np.int64)
    heel_label = LONGITUDINAL_REGION_NAMES.index("heel")
    forefoot_label = LONGITUDINAL_REGION_NAMES.index("forefoot")
    plantar_heel = np.intersect1d(
        plantar,
        np.flatnonzero(longitudinal_labels == heel_label),
        assume_unique=True,
    )
    forefoot = np.flatnonzero(longitudinal_labels == forefoot_label)
    return {
        "rear_heel": _extreme_landmark(mesh, all_vertices, 0, False),
        "longest_toe": _extreme_landmark(mesh, all_vertices, 0, True),
        "plantar_heel_contact": _extreme_landmark(mesh, plantar_heel, 1, True),
        "medial_forefoot": _extreme_landmark(mesh, forefoot, 2, False),
        "lateral_forefoot": _extreme_landmark(mesh, forefoot, 2, True),
        "ankle_boundary_center": {
            "vertex_indices": ankle_boundary.tolist(),
            "primary_vertex_index": int(np.min(ankle_boundary)),
            "point_reference": mesh.vertices[ankle_boundary].mean(axis=0).tolist(),
        },
    }


def build_canonical_supr_anatomy(
    model_path: str | Path,
) -> CanonicalSuprAnatomy:
    """Build the canonical neutral right-foot surface and anatomical labels."""

    raw_mesh, raw_joints = _load_reference_data(model_path)
    raw_regions = identify_supr_contact_regions(raw_mesh)
    forward = _canonical_transform(raw_mesh)
    inverse = np.linalg.inv(forward)
    if not np.isfinite(inverse).all() or not np.allclose(
        inverse @ forward, np.eye(4), atol=1e-12, rtol=0.0
    ):
        raise ValueError("canonical SUPR transform is not reversibly finite")
    reference_mesh = TriangleMesh(
        transform_points(raw_mesh.vertices, forward), raw_mesh.faces
    )
    reference_joints = transform_points(raw_joints, forward)
    labels = _region_labels(reference_mesh, reference_joints)
    boundary = _ordered_boundary_loop(reference_mesh.faces)
    return CanonicalSuprAnatomy(
        raw_mesh=raw_mesh,
        reference_mesh=reference_mesh,
        raw_to_reference=forward,
        reference_to_raw=inverse,
        raw_joints=raw_joints,
        reference_joints=reference_joints,
        ankle_boundary_vertex_indices=boundary,
        longitudinal_vertex_labels=labels[0],
        longitudinal_face_labels=labels[1],
        surface_vertex_labels=labels[2],
        surface_face_labels=labels[3],
        landmarks=_landmarks(
            reference_mesh,
            labels[0],
            boundary,
            raw_regions.plantar_vertex_indices,
        ),
    )


def _dense_vertex_charts(
    subdivision: SuprMeshSubdivision,
) -> tuple[np.ndarray, np.ndarray]:
    faces = subdivision.source_faces
    incident: list[set[int]] = [set() for _ in range(subdivision.source_vertex_count)]
    for face_index, face in enumerate(faces):
        for vertex in face:
            incident[int(vertex)].add(face_index)
    chart_faces = np.empty(subdivision.vertex_count, dtype=np.int64)
    barycentric = np.zeros((subdivision.vertex_count, 3), dtype=np.float64)
    for dense_index, (sources, weights) in enumerate(
        zip(subdivision.vertex_source_indices, subdivision.vertex_source_weights)
    ):
        used = sources >= 0
        source_ids = sources[used]
        candidates = set.intersection(*(incident[int(index)] for index in source_ids))
        if not candidates:
            raise RuntimeError("dense SUPR vertex has no containing source face")
        face_index = min(candidates)
        chart_faces[dense_index] = face_index
        for source, weight in zip(source_ids, weights[used]):
            corner = int(np.flatnonzero(faces[face_index] == source)[0])
            barycentric[dense_index, corner] = weight
    return chart_faces, barycentric


def build_dense_canonical_supr_anatomy(
    anatomy: CanonicalSuprAnatomy,
    subdivision: SuprMeshSubdivision,
) -> DenseCanonicalSuprAnatomy:
    """Apply one shared subdivision and label its canonical dense vertices."""

    dense_mesh = subdivision.apply_mesh(anatomy.reference_mesh)
    labels = _region_labels(dense_mesh, anatomy.reference_joints)
    chart_faces, barycentric = _dense_vertex_charts(subdivision)
    reconstructed = map_surface_coordinates(
        chart_faces,
        barycentric,
        anatomy.reference_mesh.vertices,
        anatomy.reference_mesh.faces,
    )
    if not np.allclose(reconstructed, dense_mesh.vertices, atol=1e-12, rtol=0.0):
        raise RuntimeError("dense SUPR chart does not reconstruct the reference")
    return DenseCanonicalSuprAnatomy(
        mesh=dense_mesh,
        subdivision=subdivision,
        vertex_chart_face_indices=chart_faces,
        vertex_chart_barycentric=barycentric,
        ankle_boundary_vertex_indices=_ordered_boundary_loop(dense_mesh.faces),
        longitudinal_vertex_labels=labels[0],
        longitudinal_face_labels=labels[1],
        surface_vertex_labels=labels[2],
        surface_face_labels=labels[3],
    )


def map_surface_coordinates(
    face_indices: np.ndarray,
    barycentric_weights: np.ndarray,
    target_vertices: np.ndarray,
    target_faces: np.ndarray,
) -> np.ndarray:
    """Map exact canonical face/barycentric coordinates to a SUPR surface."""

    indices = np.asarray(face_indices, dtype=np.int64)
    weights = np.asarray(barycentric_weights, dtype=np.float64)
    vertices = np.asarray(target_vertices, dtype=np.float64)
    faces = np.asarray(target_faces, dtype=np.int64)
    if indices.ndim != 1 or weights.shape != (len(indices), 3):
        raise ValueError("surface coordinates must have shapes (N,) and (N, 3)")
    if vertices.ndim != 2 or vertices.shape[1:] != (3,):
        raise ValueError("target_vertices must have shape (V, 3)")
    if faces.ndim != 2 or faces.shape[1:] != (3,):
        raise ValueError("target_faces must have shape (F, 3)")
    if (
        np.any(indices < 0)
        or np.any(indices >= len(faces))
        or np.any(faces < 0)
        or np.any(faces >= len(vertices))
    ):
        raise ValueError("surface coordinates reference invalid mesh indices")
    tolerance = 1e-12
    if (
        not np.isfinite(weights).all()
        or np.any(weights < -tolerance)
        or np.any(weights > 1.0 + tolerance)
        or not np.allclose(np.sum(weights, axis=1), 1.0, atol=tolerance, rtol=0.0)
    ):
        raise ValueError("barycentric weights must be finite, nonnegative, and sum to one")
    return np.einsum("ni,nij->nj", weights, vertices[faces[indices]])


def array_digest(*arrays: np.ndarray) -> str:
    """Return the deterministic array digest used by anatomical stages."""

    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(values.dtype.str.encode("ascii"))
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def topology_digest(faces: np.ndarray) -> str:
    """Return a topology-only digest independent of vertex positions."""

    return hashlib.sha256(
        np.ascontiguousarray(faces, dtype="<i8").tobytes()
    ).hexdigest()


def directed_boundary_loop(faces: np.ndarray) -> np.ndarray:
    """Return the single consistently wound boundary loop of a surface."""

    values = np.asarray(faces, dtype=np.int64)
    directed = np.concatenate(
        (values[:, (0, 1)], values[:, (1, 2)], values[:, (2, 0)]), axis=0
    )
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(
        undirected, axis=0, return_inverse=True, return_counts=True
    )
    if np.any(counts > 2):
        raise ValueError("dense anatomical surface contains a non-manifold edge")
    boundary = directed[counts[inverse] == 1]
    if len(boundary) == 0:
        raise ValueError("dense anatomical surface has no boundary")
    following: dict[int, int] = {}
    incoming: dict[int, int] = {}
    for first, second in boundary:
        first_int, second_int = int(first), int(second)
        if first_int in following or second_int in incoming:
            raise ValueError(
                "dense anatomical boundary is not one consistently wound loop"
            )
        following[first_int] = second_int
        incoming[second_int] = first_int
    if set(following) != set(incoming):
        raise ValueError("dense anatomical boundary is not closed")
    start = min(following)
    result: list[int] = []
    current = start
    while current not in result:
        result.append(current)
        current = following[current]
    if current != start or len(result) != len(following):
        raise ValueError("dense anatomical surface contains multiple boundary loops")
    return np.asarray(result, dtype=np.int64)


def load_dense_canonical_supr_reference(
    anatomical_surface_root: str | Path,
) -> DenseCanonicalSuprReference:
    """Load and validate the existing Checkpoint 9 canonical reference."""

    root = Path(anatomical_surface_root).expanduser().resolve(strict=True)
    reference = root / "reference"
    if not reference.is_dir():
        raise NotADirectoryError(reference)
    json_path = reference / "canonical_surface.json"
    npz_path = reference / "canonical_surface.npz"
    ply_path = reference / "neutral_dense.ply"
    for path in (json_path, npz_path, ply_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    try:
        metadata = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("Checkpoint 9 canonical_surface.json is invalid") from error
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != 1
        or metadata.get("stage")
        != "canonical_dense_supr_anatomical_reference"
    ):
        raise ValueError("Checkpoint 9 reference metadata has an unsupported schema")

    required = {
        "dense_reference_vertices",
        "dense_faces",
        "dense_ankle_boundary_vertex_indices",
        "reference_vertices",
        "dense_vertex_source_indices",
        "dense_vertex_source_weights",
        "dense_face_parent_indices",
        "dense_vertex_chart_face_indices",
        "dense_vertex_chart_barycentric",
        "dense_longitudinal_vertex_labels",
        "dense_longitudinal_face_labels",
        "dense_surface_vertex_labels",
        "dense_surface_face_labels",
        "native_faces",
        "ankle_boundary_vertex_indices",
        "joint_names",
        "reference_joints",
        "raw_supr_to_reference",
        "reference_to_raw_supr",
    }
    with np.load(npz_path, allow_pickle=False) as archive:
        missing = sorted(required.difference(archive.files))
        if missing:
            raise ValueError(f"Checkpoint 9 NPZ is missing arrays: {missing}")
        arrays = {name: np.asarray(archive[name]) for name in required}

    vertices = np.asarray(arrays["dense_reference_vertices"], dtype=np.float64)
    faces = np.asarray(arrays["dense_faces"], dtype=np.int64)
    ankle_loop = np.asarray(
        arrays["dense_ankle_boundary_vertex_indices"], dtype=np.int64
    )
    native_vertices = np.asarray(arrays["reference_vertices"], dtype=np.float64)
    native_faces = np.asarray(arrays["native_faces"], dtype=np.int64)
    native_ankle_loop = np.asarray(
        arrays["ankle_boundary_vertex_indices"], dtype=np.int64
    )
    if vertices.shape != (DENSE_FOOT_VERTEX_COUNT, 3) or faces.shape != (
        DENSE_FOOT_FACE_COUNT,
        3,
    ):
        raise ValueError("Checkpoint 9 dense topology must be 4,151/8,240")
    if (
        native_vertices.shape != (266, 3)
        or native_faces.shape != (515, 3)
        or not np.array_equal(vertices[:266], native_vertices)
    ):
        raise ValueError("Checkpoint 9 native SUPR correspondence is invalid")
    if (
        np.any(faces < 0)
        or np.any(faces >= len(vertices))
        or not np.isfinite(vertices).all()
    ):
        raise ValueError("Checkpoint 9 dense mesh contains invalid geometry")
    if ankle_loop.shape != (DENSE_ANKLE_VERTEX_COUNT,) or not np.array_equal(
        directed_boundary_loop(faces), ankle_loop
    ):
        raise ValueError("Checkpoint 9 dense ankle boundary is invalid")
    if (
        native_ankle_loop.shape != (15,)
        or len(np.unique(native_ankle_loop)) != 15
        or np.any(native_ankle_loop < 0)
        or np.any(native_ankle_loop >= len(native_vertices))
    ):
        raise ValueError("Checkpoint 9 native ankle boundary is invalid")

    source_indices = np.asarray(arrays["dense_vertex_source_indices"], dtype=np.int64)
    source_weights = np.asarray(arrays["dense_vertex_source_weights"], dtype=np.float64)
    face_parents = np.asarray(arrays["dense_face_parent_indices"], dtype=np.int64)
    chart_faces = np.asarray(arrays["dense_vertex_chart_face_indices"], dtype=np.int64)
    chart_weights = np.asarray(
        arrays["dense_vertex_chart_barycentric"], dtype=np.float64
    )
    if (
        source_indices.shape != (DENSE_FOOT_VERTEX_COUNT, 3)
        or source_weights.shape != (DENSE_FOOT_VERTEX_COUNT, 3)
        or face_parents.shape != (DENSE_FOOT_FACE_COUNT,)
        or chart_faces.shape != (DENSE_FOOT_VERTEX_COUNT,)
        or chart_weights.shape != (DENSE_FOOT_VERTEX_COUNT, 3)
        or not np.array_equal(source_indices[:266, 0], np.arange(266))
        or not np.all(source_indices[:266, 1:] == -1)
        or not np.array_equal(source_weights[:266, 0], np.ones(266))
        or not np.all(source_weights[:266, 1:] == 0.0)
        or np.any(face_parents < 0)
        or np.any(face_parents >= len(native_faces))
        or np.any(chart_faces < 0)
        or np.any(chart_faces >= len(native_faces))
        or not np.isfinite(source_weights).all()
        or not np.isfinite(chart_weights).all()
        or np.any(source_weights < -1.0e-12)
        or np.any(chart_weights < -1.0e-12)
        or not np.allclose(source_weights.sum(axis=1), 1.0, atol=1.0e-12)
        or not np.allclose(chart_weights.sum(axis=1), 1.0, atol=1.0e-12)
    ):
        raise ValueError("Checkpoint 9 subdivision provenance is invalid")
    reconstructed = map_surface_coordinates(
        chart_faces, chart_weights, native_vertices, native_faces
    )
    if not np.allclose(reconstructed, vertices, atol=1.0e-12, rtol=0.0):
        raise ValueError("Checkpoint 9 surface charts do not reconstruct the foot")

    joint_names = tuple(str(value) for value in arrays["joint_names"].tolist())
    reference_joints = np.asarray(arrays["reference_joints"], dtype=np.float64)
    forward = np.asarray(arrays["raw_supr_to_reference"], dtype=np.float64)
    inverse = np.asarray(arrays["reference_to_raw_supr"], dtype=np.float64)
    if (
        joint_names != JOINT_NAMES
        or reference_joints.shape != (len(JOINT_NAMES), 3)
        or forward.shape != (4, 4)
        or inverse.shape != (4, 4)
        or not np.isfinite(reference_joints).all()
        or not np.isfinite(forward).all()
        or not np.isfinite(inverse).all()
        or not np.allclose(forward @ inverse, np.eye(4), atol=1.0e-12, rtol=0.0)
        or not np.allclose(inverse @ forward, np.eye(4), atol=1.0e-12, rtol=0.0)
    ):
        raise ValueError("Checkpoint 9 joints or canonical transform are invalid")

    longitudinal_vertices = np.asarray(
        arrays["dense_longitudinal_vertex_labels"], dtype=np.int16
    )
    longitudinal_faces = np.asarray(
        arrays["dense_longitudinal_face_labels"], dtype=np.int16
    )
    surface_vertices = np.asarray(
        arrays["dense_surface_vertex_labels"], dtype=np.int16
    )
    surface_faces = np.asarray(
        arrays["dense_surface_face_labels"], dtype=np.int16
    )
    if (
        longitudinal_vertices.shape != (DENSE_FOOT_VERTEX_COUNT,)
        or longitudinal_faces.shape != (DENSE_FOOT_FACE_COUNT,)
        or surface_vertices.shape != (DENSE_FOOT_VERTEX_COUNT,)
        or surface_faces.shape != (DENSE_FOOT_FACE_COUNT,)
    ):
        raise ValueError("Checkpoint 9 anatomical labels are invalid")

    dense_digest = array_digest(vertices, faces)
    dense_topology = topology_digest(faces)
    digests = metadata.get("digests", {})
    if (
        digests.get("canonical_dense_surface_sha256") != dense_digest
        or digests.get("dense_topology_sha256") != dense_topology
    ):
        raise ValueError("Checkpoint 9 reference digest does not match")
    saved_mesh = load_triangle_mesh(ply_path)
    if not np.array_equal(saved_mesh.faces, faces) or not np.allclose(
        saved_mesh.vertices, vertices, atol=5.0e-8, rtol=0.0
    ):
        raise ValueError("Checkpoint 9 neutral_dense.ply disagrees with its NPZ")
    landmarks = metadata.get("landmarks")
    if not isinstance(landmarks, dict):
        raise ValueError("Checkpoint 9 foot landmarks are missing")
    return DenseCanonicalSuprReference(
        vertices=vertices,
        faces=faces,
        ankle_loop=ankle_loop,
        native_vertices=native_vertices,
        native_faces=native_faces,
        native_ankle_loop=native_ankle_loop,
        joint_names=joint_names,
        reference_joints=reference_joints,
        raw_supr_to_reference=forward,
        reference_to_raw_supr=inverse,
        dense_vertex_source_indices=source_indices,
        dense_vertex_source_weights=source_weights,
        dense_face_parent_indices=face_parents,
        vertex_chart_face_indices=chart_faces,
        vertex_chart_barycentric=chart_weights,
        longitudinal_vertex_labels=longitudinal_vertices,
        longitudinal_face_labels=longitudinal_faces,
        surface_vertex_labels=surface_vertices,
        surface_face_labels=surface_faces,
        foot_landmarks=landmarks,
        dense_surface_digest=dense_digest,
        topology_digest=dense_topology,
    )


def _bridge_faces(foot_loop: np.ndarray, leg_loop: np.ndarray) -> np.ndarray:
    faces: list[tuple[int, int, int]] = []
    for index in range(len(foot_loop)):
        following = (index + 1) % len(foot_loop)
        faces.append(
            (int(foot_loop[index]), int(leg_loop[index]), int(foot_loop[following]))
        )
        faces.append(
            (
                int(foot_loop[following]),
                int(leg_loop[index]),
                int(leg_loop[following]),
            )
        )
    return np.asarray(faces, dtype=np.int64)


def _vertex_charts_for_topology(faces: np.ndarray, vertex_count: int) -> tuple[np.ndarray, np.ndarray]:
    chart_faces = np.full(vertex_count, -1, dtype=np.int64)
    barycentric = np.zeros((vertex_count, 3), dtype=np.float64)
    for face_index, face in enumerate(faces):
        for corner, vertex in enumerate(face):
            vertex_index = int(vertex)
            if chart_faces[vertex_index] < 0:
                chart_faces[vertex_index] = face_index
                barycentric[vertex_index, corner] = 1.0
    if np.any(chart_faces < 0):
        raise ValueError("extended surface contains an unused vertex")
    return chart_faces, barycentric


def _anatomical_frame(lower_leg: SuprLowerLeg) -> np.ndarray:
    joints = lower_leg.source_joints_reference
    ankle = joints[RIGHT_ANKLE_JOINT_INDEX]
    up = joints[RIGHT_KNEE_JOINT_INDEX] - ankle
    up /= np.linalg.norm(up)
    forward = joints[RIGHT_FOOT_JOINT_INDEX] - ankle
    forward -= up * float(np.dot(forward, up))
    forward /= np.linalg.norm(forward)
    lateral = np.cross(forward, up)
    lateral /= np.linalg.norm(lateral)
    forward = np.cross(up, lateral)
    forward /= np.linalg.norm(forward)
    return np.column_stack((forward, up, lateral))


def _frame_surface_labels(normals: np.ndarray, frame: np.ndarray) -> np.ndarray:
    local = np.asarray(normals, dtype=np.float64) @ np.asarray(frame, dtype=np.float64)
    dominant = np.argmax(np.abs(local), axis=1)
    labels = np.empty(len(local), dtype=np.int16)
    labels[(dominant == 0) & (local[:, 0] >= 0.0)] = SURFACE_REGION_NAMES.index("anterior")
    labels[(dominant == 0) & (local[:, 0] < 0.0)] = SURFACE_REGION_NAMES.index("posterior")
    labels[(dominant == 1) & (local[:, 1] >= 0.0)] = SURFACE_REGION_NAMES.index("top")
    labels[(dominant == 1) & (local[:, 1] < 0.0)] = SURFACE_REGION_NAMES.index("plantar")
    labels[(dominant == 2) & (local[:, 2] >= 0.0)] = SURFACE_REGION_NAMES.index("lateral")
    labels[(dominant == 2) & (local[:, 2] < 0.0)] = SURFACE_REGION_NAMES.index("medial")
    return labels


def _lower_leg_longitudinal_labels(
    points: np.ndarray,
    ankle_center: np.ndarray,
    knee_center: np.ndarray,
) -> np.ndarray:
    axis = np.asarray(knee_center) - np.asarray(ankle_center)
    length_squared = float(np.dot(axis, axis))
    if length_squared <= np.finfo(np.float64).eps:
        raise ValueError("lower-leg ankle and knee centers coincide")
    fraction = (np.asarray(points) - ankle_center) @ axis / length_squared
    labels = np.full(len(points), EXTENDED_LONGITUDINAL_REGION_NAMES.index("calf"), dtype=np.int16)
    labels[fraction < 1.0 / 3.0] = EXTENDED_LONGITUDINAL_REGION_NAMES.index("lower_shaft")
    labels[fraction >= 2.0 / 3.0] = EXTENDED_LONGITUDINAL_REGION_NAMES.index("upper_shaft")
    return labels


def _projected_landmark(
    vertices: np.ndarray,
    candidates: np.ndarray,
    direction: np.ndarray,
    maximum: bool,
) -> dict[str, Any]:
    indices = np.asarray(candidates, dtype=np.int64)
    values = np.asarray(vertices)[indices] @ np.asarray(direction, dtype=np.float64)
    selected = int(indices[np.argmax(values) if maximum else np.argmin(values)])
    return {
        "vertex_indices": [selected],
        "primary_vertex_index": selected,
        "point_reference": np.asarray(vertices)[selected].tolist(),
    }


def build_extended_canonical_supr_anatomy(
    reference: DenseCanonicalSuprReference,
    full_body_supr_model: str | Path,
) -> ExtendedCanonicalSuprAnatomy:
    """Join and label the canonical dense foot and neutral right lower leg."""

    lower_leg = build_canonical_right_lower_leg(
        full_body_supr_model,
        reference.raw_supr_to_reference,
        reference.reference_joints,
        reference.native_vertices[reference.native_ankle_loop],
    )
    subdivision = build_supr_mesh_subdivision(
        lower_leg.mesh.faces, len(lower_leg.mesh.vertices), 2
    )
    dense_leg = subdivision.apply_mesh(lower_leg.mesh)
    loops = _ordered_boundary_loops(dense_leg.faces)
    if sorted(len(loop) for loop in loops) != [60, 68]:
        raise ValueError("subdivided lower leg must have 60/68 ankle and knee loops")
    distal = next(loop for loop in loops if len(loop) == 60)
    proximal = next(loop for loop in loops if len(loop) == 68)

    foot_loop = reference.ankle_loop
    leg_offset = len(reference.vertices)
    candidates: list[tuple[float, int, int, np.ndarray, np.ndarray, np.ndarray]] = []
    for reversed_direction in (0, 1):
        ordered = distal if reversed_direction == 0 else distal[::-1]
        for shift in range(len(ordered)):
            paired_local = np.roll(ordered, -shift)
            native_pairs = (foot_loop < 266) & (
                paired_local < len(lower_leg.mesh.vertices)
            )
            if int(np.count_nonzero(native_pairs)) != 15:
                continue
            paired_global = paired_local + leg_offset
            bridge = _bridge_faces(foot_loop, paired_global)
            faces = np.vstack(
                (reference.faces, dense_leg.faces + leg_offset, bridge)
            )
            try:
                remaining_loop = directed_boundary_loop(faces)
            except ValueError:
                continue
            if len(remaining_loop) != 68 or set(remaining_loop) != set(
                (proximal + leg_offset).tolist()
            ):
                continue
            residual = float(
                np.sum(
                    (reference.vertices[foot_loop] - dense_leg.vertices[paired_local])
                    ** 2
                )
            )
            candidates.append(
                (
                    residual,
                    reversed_direction,
                    shift,
                    paired_local,
                    bridge,
                    remaining_loop,
                )
            )
    if not candidates:
        raise ValueError("no consistently wound ankle-loop correspondence exists")
    candidates.sort(key=lambda value: (value[0], value[1], value[2]))
    residual, direction, shift, paired_local, bridge, knee_loop = candidates[0]
    vertices = np.vstack((reference.vertices, dense_leg.vertices))
    faces = np.vstack((reference.faces, dense_leg.faces + leg_offset, bridge))
    if vertices.shape != (EXTENDED_VERTEX_COUNT, 3) or faces.shape != (
        EXTENDED_FACE_COUNT,
        3,
    ):
        raise ValueError("extended SUPR topology must be 6,951/13,832")
    if not np.array_equal(vertices[:DENSE_FOOT_VERTEX_COUNT], reference.vertices):
        raise RuntimeError("extended anatomy changed a Checkpoint 9 foot vertex")
    if not np.array_equal(faces[:DENSE_FOOT_FACE_COUNT], reference.faces):
        raise RuntimeError("extended anatomy changed a Checkpoint 9 foot face")
    bridge_triangles = vertices[bridge]
    bridge_areas = np.linalg.norm(
        np.cross(
            bridge_triangles[:, 1] - bridge_triangles[:, 0],
            bridge_triangles[:, 2] - bridge_triangles[:, 0],
        ),
        axis=1,
    )
    if np.any(bridge_areas <= 128.0 * np.finfo(np.float64).eps):
        raise ValueError("ankle bridge contains a degenerate triangle")

    foot_faces = np.arange(DENSE_FOOT_FACE_COUNT, dtype=np.int64)
    leg_faces = np.arange(
        DENSE_FOOT_FACE_COUNT,
        DENSE_FOOT_FACE_COUNT + len(dense_leg.faces),
        dtype=np.int64,
    )
    bridge_faces = np.arange(
        DENSE_FOOT_FACE_COUNT + len(dense_leg.faces), len(faces), dtype=np.int64
    )
    correspondence = np.column_stack((foot_loop, paired_local + leg_offset))
    if int(
        np.count_nonzero(
            (correspondence[:, 0] < 266)
            & (correspondence[:, 1] - leg_offset < len(lower_leg.mesh.vertices))
        )
    ) != 15:
        raise RuntimeError("ankle attachment lost native loop correspondence")

    frame = _anatomical_frame(lower_leg)
    ankle_center = dense_leg.vertices[paired_local].mean(axis=0)
    knee_center = vertices[knee_loop].mean(axis=0)
    leg_vertex_longitudinal = _lower_leg_longitudinal_labels(
        dense_leg.vertices, ankle_center, knee_center
    )
    leg_face_longitudinal = _lower_leg_longitudinal_labels(
        dense_leg.vertices[dense_leg.faces].mean(axis=1), ankle_center, knee_center
    )
    longitudinal_vertices = np.concatenate(
        (reference.longitudinal_vertex_labels, leg_vertex_longitudinal)
    )
    longitudinal_faces = np.concatenate(
        (
            reference.longitudinal_face_labels,
            leg_face_longitudinal,
            np.full(len(bridge), LONGITUDINAL_REGION_NAMES.index("ankle"), dtype=np.int16),
        )
    )
    leg_vertex_normals, leg_face_normals = _weighted_normals(dense_leg)
    bridge_triangles = vertices[bridge]
    bridge_crosses = np.cross(
        bridge_triangles[:, 1] - bridge_triangles[:, 0],
        bridge_triangles[:, 2] - bridge_triangles[:, 0],
    )
    bridge_face_normals = bridge_crosses / np.linalg.norm(
        bridge_crosses, axis=1
    )[:, None]
    surface_vertices = np.concatenate(
        (reference.surface_vertex_labels, _frame_surface_labels(leg_vertex_normals, frame))
    )
    surface_faces = np.concatenate(
        (
            reference.surface_face_labels,
            _frame_surface_labels(leg_face_normals, frame),
            _frame_surface_labels(bridge_face_normals, frame),
        )
    )
    component_vertices = np.concatenate(
        (
            np.full(DENSE_FOOT_VERTEX_COUNT, COMPONENT_REGION_NAMES.index("foot_skin"), dtype=np.int16),
            np.full(len(dense_leg.vertices), COMPONENT_REGION_NAMES.index("lower_leg_skin"), dtype=np.int16),
        )
    )
    component_faces = np.empty(len(faces), dtype=np.int16)
    component_faces[foot_faces] = COMPONENT_REGION_NAMES.index("foot_skin")
    component_faces[leg_faces] = COMPONENT_REGION_NAMES.index("lower_leg_skin")
    component_faces[bridge_faces] = COMPONENT_REGION_NAMES.index("ankle_transition")

    chart_faces, chart_weights = _vertex_charts_for_topology(faces, len(vertices))
    reconstructed = map_surface_coordinates(chart_faces, chart_weights, vertices, faces)
    if not np.array_equal(reconstructed, vertices):
        raise RuntimeError("extended surface charts do not reconstruct its vertices")

    axis_values = (dense_leg.vertices - ankle_center) @ frame[:, 1]
    span = float((knee_center - ankle_center) @ frame[:, 1])
    middle_local = np.flatnonzero(
        (axis_values >= 0.4 * span) & (axis_values <= 0.6 * span)
    )
    if len(middle_local) == 0:
        raise RuntimeError("lower leg has no mid-shaft landmark candidates")
    middle_global = middle_local + leg_offset
    landmarks = dict(reference.foot_landmarks)
    landmarks.update(
        {
            "foot_ankle_boundary_center": {
                "vertex_indices": foot_loop.tolist(),
                "primary_vertex_index": int(np.min(foot_loop)),
                "point_reference": reference.vertices[foot_loop].mean(axis=0).tolist(),
            },
            "lower_leg_ankle_boundary_center": {
                "vertex_indices": (paired_local + leg_offset).tolist(),
                "primary_vertex_index": int(np.min(paired_local + leg_offset)),
                "point_reference": ankle_center.tolist(),
            },
            "knee_boundary_center": {
                "vertex_indices": knee_loop.tolist(),
                "primary_vertex_index": int(np.min(knee_loop)),
                "point_reference": knee_center.tolist(),
            },
            "anterior_mid_shaft": _projected_landmark(vertices, middle_global, frame[:, 0], True),
            "posterior_mid_shaft": _projected_landmark(vertices, middle_global, frame[:, 0], False),
            "medial_mid_shaft": _projected_landmark(vertices, middle_global, frame[:, 2], False),
            "lateral_mid_shaft": _projected_landmark(vertices, middle_global, frame[:, 2], True),
        }
    )
    lower_leg_metadata = lower_leg.to_dict()
    lower_leg_metadata["subdivision"] = subdivision.to_dict()
    lower_leg_metadata["dense_vertex_count"] = int(len(dense_leg.vertices))
    lower_leg_metadata["dense_face_count"] = int(len(dense_leg.faces))
    lower_leg_metadata["dense_ankle_boundary_count"] = 60
    lower_leg_metadata["dense_knee_boundary_count"] = 68
    joint_indices = np.asarray(
        (RIGHT_KNEE_JOINT_INDEX, RIGHT_ANKLE_JOINT_INDEX, RIGHT_FOOT_JOINT_INDEX),
        dtype=np.int64,
    )
    return ExtendedCanonicalSuprAnatomy(
        vertices=vertices,
        faces=faces,
        foot_face_indices=foot_faces,
        lower_leg_face_indices=leg_faces,
        bridge_face_indices=bridge_faces,
        dense_foot_indices=np.arange(DENSE_FOOT_VERTEX_COUNT, dtype=np.int64),
        lower_leg_indices=np.arange(leg_offset, len(vertices), dtype=np.int64),
        knee_loop=knee_loop,
        ankle_correspondence=correspondence,
        lower_leg=lower_leg,
        lower_leg_subdivision=subdivision,
        lower_leg_metadata=lower_leg_metadata,
        attachment_diagnostics={
            "paired_squared_residual": residual,
            "lower_leg_loop_direction": "forward" if direction == 0 else "reversed",
            "cyclic_offset": int(shift),
        },
        vertex_chart_face_indices=chart_faces,
        vertex_chart_barycentric=chart_weights,
        foot_native_chart_face_indices=reference.vertex_chart_face_indices,
        foot_native_chart_barycentric=reference.vertex_chart_barycentric,
        foot_native_vertices=reference.native_vertices,
        foot_native_faces=reference.native_faces,
        raw_supr_to_reference=reference.raw_supr_to_reference,
        reference_to_raw_supr=reference.reference_to_raw_supr,
        foot_dense_vertex_source_indices=reference.dense_vertex_source_indices,
        foot_dense_vertex_source_weights=reference.dense_vertex_source_weights,
        foot_dense_face_parent_indices=reference.dense_face_parent_indices,
        longitudinal_vertex_labels=longitudinal_vertices,
        longitudinal_face_labels=longitudinal_faces,
        surface_vertex_labels=surface_vertices,
        surface_face_labels=surface_faces,
        component_vertex_labels=component_vertices,
        component_face_labels=component_faces,
        landmarks=landmarks,
        foot_joint_names=reference.joint_names,
        foot_reference_joints=reference.reference_joints,
        lower_leg_joint_names=("right_knee", "right_ankle", "right_foot"),
        lower_leg_joint_source_indices=joint_indices,
        lower_leg_reference_joints=lower_leg.source_joints_reference[joint_indices],
        anatomical_frame=frame,
        digest=array_digest(vertices, faces),
        topology_digest=topology_digest(faces),
    )
