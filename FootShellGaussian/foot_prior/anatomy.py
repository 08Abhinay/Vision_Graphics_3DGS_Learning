"""Canonical anatomical labels and exact SUPR surface correspondence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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
from .mesh import TriangleMesh
from .supr_foot import SuprMeshSubdivision, load_neutral_supr_foot


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
