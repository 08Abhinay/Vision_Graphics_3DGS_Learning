"""Surface-based cavity clearance and collision analysis for fitted feet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

from .alignment import CONTACT_REGION_RANGES
from .mesh import TriangleMesh, sample_triangle_mesh_y


@dataclass(frozen=True)
class SignedCavityClearance:
    """Local winding-independent upper and side containment measurements."""

    vertex_upper_clearances: np.ndarray
    vertex_upper_faces: np.ndarray
    vertex_upper_points: np.ndarray
    vertex_side_clearances: np.ndarray
    vertex_side_faces: np.ndarray
    vertex_side_points: np.ndarray
    vertex_combined_clearances: np.ndarray
    face_upper_clearances: np.ndarray
    face_upper_faces: np.ndarray
    face_upper_points: np.ndarray
    face_side_clearances: np.ndarray
    face_side_faces: np.ndarray
    face_side_points: np.ndarray
    face_combined_clearances: np.ndarray
    signed_exempt_vertex_indices: np.ndarray
    signed_exempt_face_indices: np.ndarray
    outside_vertex_indices: np.ndarray
    outside_face_indices: np.ndarray
    outside_area: float
    outside_area_fraction: float
    protrusion_energy: float
    protrusion_statistics: dict[str, Any]
    regional_summaries: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        def optional(values: np.ndarray) -> list[float | None]:
            return [float(value) if np.isfinite(value) else None for value in values]

        def points(values: np.ndarray) -> list[list[float] | None]:
            return [
                point.tolist() if np.isfinite(point).all() else None
                for point in values
            ]

        return {
            "definition": {
                "positive": "foot sample is inside its local inner boundary",
                "zero": "foot sample touches its local inner boundary",
                "negative": "foot sample lies beyond its local inner boundary",
                "open_space": "no boundary is invented when no shoe surface is found",
                "signed_exemptions": (
                    "listed SUPR samples are omitted only from signed upper/side "
                    "scoring; exact obstacle intersections remain active"
                ),
            },
            "signed_score_exemptions": {
                "vertex_indices": self.signed_exempt_vertex_indices.tolist(),
                "face_indices": self.signed_exempt_face_indices.tolist(),
            },
            "outside_vertex_indices": self.outside_vertex_indices.tolist(),
            "outside_face_indices": self.outside_face_indices.tolist(),
            "outside_area": self.outside_area,
            "outside_area_fraction": self.outside_area_fraction,
            "protrusion_energy": self.protrusion_energy,
            "protrusion_statistics": self.protrusion_statistics,
            "regional_summaries": self.regional_summaries,
            "vertices": {
                "upper": {
                    "clearances": optional(self.vertex_upper_clearances),
                    "boundary_shoe_face_indices": self.vertex_upper_faces.tolist(),
                    "boundary_points": points(self.vertex_upper_points),
                    "open_sample_count": int(np.count_nonzero(~np.isfinite(self.vertex_upper_clearances))),
                },
                "side": {
                    "clearances": optional(self.vertex_side_clearances),
                    "boundary_shoe_face_indices": self.vertex_side_faces.tolist(),
                    "boundary_points": points(self.vertex_side_points),
                    "open_sample_count": int(np.count_nonzero(~np.isfinite(self.vertex_side_clearances))),
                },
                "combined_clearances": optional(self.vertex_combined_clearances),
            },
            "face_centroids": {
                "upper": {
                    "clearances": optional(self.face_upper_clearances),
                    "boundary_shoe_face_indices": self.face_upper_faces.tolist(),
                    "boundary_points": points(self.face_upper_points),
                    "open_sample_count": int(np.count_nonzero(~np.isfinite(self.face_upper_clearances))),
                },
                "side": {
                    "clearances": optional(self.face_side_clearances),
                    "boundary_shoe_face_indices": self.face_side_faces.tolist(),
                    "boundary_points": points(self.face_side_points),
                    "open_sample_count": int(np.count_nonzero(~np.isfinite(self.face_side_clearances))),
                },
                "combined_clearances": optional(self.face_combined_clearances),
            },
        }


@dataclass(frozen=True)
class CavityAnalysis:
    """Deterministic contact, collision, and clearance measurements."""

    numerical_tolerance: float
    footbed_source_face_indices: np.ndarray
    obstacle_face_count: int
    ignored_degenerate_obstacle_face_indices: np.ndarray
    ignored_degenerate_foot_face_indices: np.ndarray
    support_contact: dict[str, Any]
    collision_pairs: np.ndarray
    vertex_closest_points: np.ndarray
    vertex_vectors_to_obstacle: np.ndarray
    vertex_clearances: np.ndarray
    vertex_nearest_shoe_faces: np.ndarray
    face_closest_points: np.ndarray
    face_vectors_to_obstacle: np.ndarray
    face_clearances: np.ndarray
    face_nearest_shoe_faces: np.ndarray
    clearance_summaries: dict[str, Any]
    signed_clearance: SignedCavityClearance

    @property
    def status(self) -> str:
        support_penetration = (
            self.support_contact["plantar_vertices"]["penetrating_sample_count"]
            + self.support_contact["plantar_face_centroids"][
                "penetrating_sample_count"
            ]
        )
        forbidden_touch = bool(
            np.any(self.vertex_clearances <= self.numerical_tolerance)
            or np.any(self.face_clearances <= self.numerical_tolerance)
        )
        if len(self.collision_pairs) or support_penetration or forbidden_touch:
            return "collisions_detected"
        if len(self.signed_clearance.outside_face_indices):
            return "protrusion_detected"
        return "clear"

    @property
    def colliding_foot_face_indices(self) -> np.ndarray:
        if len(self.collision_pairs) == 0:
            return np.empty(0, dtype=np.int64)
        return np.unique(self.collision_pairs[:, 0])

    @property
    def colliding_shoe_face_indices(self) -> np.ndarray:
        if len(self.collision_pairs) == 0:
            return np.empty(0, dtype=np.int64)
        return np.unique(self.collision_pairs[:, 1])

    def foot_vertex_colors(
        self,
        foot_mesh: TriangleMesh,
        near_distance: float,
    ) -> np.ndarray:
        """Color foot vertices by exact collision and signed containment."""

        threshold = float(near_distance)
        if not np.isfinite(threshold) or threshold <= 0.0:
            raise ValueError("near_distance must be finite and positive")
        if len(foot_mesh.vertices) != len(self.vertex_clearances):
            raise ValueError("foot mesh does not match cavity-analysis vertices")

        blue = np.asarray([45, 105, 220, 255], dtype=np.uint8)
        yellow = np.asarray([245, 190, 35, 255], dtype=np.uint8)
        red = np.asarray([220, 45, 45, 255], dtype=np.uint8)
        magenta = np.asarray([205, 45, 190, 255], dtype=np.uint8)
        colors = np.tile(blue, (len(foot_mesh.vertices), 1))
        colors[self.vertex_clearances <= threshold] = yellow

        outside_vertices = set(self.signed_clearance.outside_vertex_indices.tolist())
        if len(self.signed_clearance.outside_face_indices):
            outside_vertices.update(
                np.unique(foot_mesh.faces[self.signed_clearance.outside_face_indices]).tolist()
            )
        if outside_vertices:
            colors[np.asarray(sorted(outside_vertices), dtype=np.int64)] = magenta

        red_vertices = set(
            np.flatnonzero(
                self.vertex_clearances <= self.numerical_tolerance
            ).tolist()
        )
        if len(self.colliding_foot_face_indices):
            red_vertices.update(
                np.unique(
                    foot_mesh.faces[self.colliding_foot_face_indices]
                ).tolist()
            )
        penetrating = self.support_contact["plantar_vertices"][
            "penetrating_sample_indices"
        ]
        red_vertices.update(int(index) for index in penetrating)
        if red_vertices:
            colors[np.asarray(sorted(red_vertices), dtype=np.int64)] = red
        return colors

    def to_dict(self) -> dict[str, Any]:
        """Return a complete JSON-compatible diagnostic record."""

        return {
            "status": self.status,
            "contact_policy": {
                "allowed": "plantar contact with the detected footbed",
                "forbidden": (
                    "contact or intersection with every non-footbed shoe face"
                ),
                "opening": (
                    "the ankle may pass through empty opening space but may not "
                    "intersect the collar"
                ),
                "distance_type": (
                    "unsigned nearest distance plus local signed upper/side "
                    "clearance; no closed-volume inside/outside claim"
                ),
            },
            "numerical_tolerance": self.numerical_tolerance,
            "shoe_face_partition": {
                "footbed_source_face_indices_source": (
                    "shoe_preparation.json:footbed_selection."
                    "original_face_indices"
                ),
                "footbed_face_count": int(len(self.footbed_source_face_indices)),
                "obstacle_face_count": self.obstacle_face_count,
                "ignored_degenerate_obstacle_face_indices": (
                    self.ignored_degenerate_obstacle_face_indices.tolist()
                ),
            },
            "support_contact": self.support_contact,
            "obstacle_collisions": {
                "intersecting_pair_count": int(len(self.collision_pairs)),
                "intersecting_face_pairs": self.collision_pairs.tolist(),
                "foot_face_indices": self.colliding_foot_face_indices.tolist(),
                "shoe_face_indices": self.colliding_shoe_face_indices.tolist(),
                "ignored_degenerate_foot_face_indices": (
                    self.ignored_degenerate_foot_face_indices.tolist()
                ),
                "touching_vertex_indices": np.flatnonzero(
                    self.vertex_clearances <= self.numerical_tolerance
                ).tolist(),
                "touching_face_centroid_indices": np.flatnonzero(
                    self.face_clearances <= self.numerical_tolerance
                ).tolist(),
            },
            "clearance": {
                "summaries": self.clearance_summaries,
                "vertices": {
                    "distances": self.vertex_clearances.tolist(),
                    "nearest_shoe_face_indices": (
                        self.vertex_nearest_shoe_faces.tolist()
                    ),
                    "closest_points": self.vertex_closest_points.tolist(),
                    "vectors_to_obstacle": (
                        self.vertex_vectors_to_obstacle.tolist()
                    ),
                },
                "face_centroids": {
                    "distances": self.face_clearances.tolist(),
                    "nearest_shoe_face_indices": (
                        self.face_nearest_shoe_faces.tolist()
                    ),
                    "closest_points": self.face_closest_points.tolist(),
                    "vectors_to_obstacle": (
                        self.face_vectors_to_obstacle.tolist()
                    ),
                },
            },
            "signed_clearance": self.signed_clearance.to_dict(),
        }


def _validate_indices(
    values: np.ndarray,
    upper_bound: int,
    label: str,
    *,
    allow_empty: bool = False,
) -> np.ndarray:
    indices = np.asarray(values)
    if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise ValueError(f"{label} must be a one-dimensional integer array")
    indices = np.asarray(indices, dtype=np.int64)
    if not allow_empty and len(indices) == 0:
        raise ValueError(f"{label} must not be empty")
    if len(indices) and (np.any(indices < 0) or np.any(indices >= upper_bound)):
        raise ValueError(f"{label} contain indices outside the source mesh")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"{label} must not contain duplicates")
    return np.sort(indices)


def _coordinate_tolerance(*meshes: TriangleMesh) -> float:
    maximum = max(float(np.max(np.abs(mesh.vertices))) for mesh in meshes)
    magnitude = np.float32(max(1.0, maximum))
    return float(np.spacing(magnitude))


def _canonical_triangle_rows(triangles: np.ndarray) -> np.ndarray:
    canonical = np.empty_like(triangles, dtype=np.float64)
    for index, triangle in enumerate(np.asarray(triangles, dtype=np.float64)):
        order = np.lexsort((triangle[:, 2], triangle[:, 1], triangle[:, 0]))
        canonical[index] = triangle[order]
    flattened = canonical.reshape(len(canonical), -1)
    order = np.lexsort(
        tuple(flattened[:, column] for column in range(8, -1, -1))
    )
    return canonical[order]


def _validate_footbed_faces(
    shoe_mesh: TriangleMesh,
    footbed_mesh: TriangleMesh,
    face_indices: np.ndarray,
    tolerance: float,
) -> None:
    if len(face_indices) != len(footbed_mesh.faces):
        raise ValueError(
            "footbed source-face count does not match the saved footbed mesh"
        )
    shoe_triangles = _canonical_triangle_rows(
        shoe_mesh.vertices[shoe_mesh.faces[face_indices]]
    )
    footbed_triangles = _canonical_triangle_rows(
        footbed_mesh.vertices[footbed_mesh.faces]
    )
    if not np.allclose(
        shoe_triangles,
        footbed_triangles,
        atol=8.0 * tolerance,
        rtol=0.0,
    ):
        raise ValueError(
            "saved footbed geometry does not match its normalized shoe faces"
        )


def _triangle_double_areas(triangles: np.ndarray) -> np.ndarray:
    return np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )


def _axis_separates(
    first: np.ndarray,
    second: np.ndarray,
    axes: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    norms = np.linalg.norm(axes, axis=1)
    usable = norms > np.finfo(np.float64).eps
    unit = np.divide(
        axes,
        norms[:, None],
        out=np.zeros_like(axes),
        where=usable[:, None],
    )
    first_projection = np.einsum("vj,nj->nv", first, unit)
    second_projection = np.einsum("nvj,nj->nv", second, unit)
    separated = (
        first_projection.max(axis=1)
        < second_projection.min(axis=1) - tolerance
    ) | (
        second_projection.max(axis=1)
        < first_projection.min(axis=1) - tolerance
    )
    return usable & separated


def _triangle_intersections(
    triangle: np.ndarray,
    candidates: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Return triangle/triangle contact using a zero-thickness SAT test."""

    first_edges = np.asarray(
        [
            triangle[1] - triangle[0],
            triangle[2] - triangle[1],
            triangle[0] - triangle[2],
        ]
    )
    second_edges = np.stack(
        (
            candidates[:, 1] - candidates[:, 0],
            candidates[:, 2] - candidates[:, 1],
            candidates[:, 0] - candidates[:, 2],
        ),
        axis=1,
    )
    first_normal = np.cross(first_edges[0], first_edges[1])
    second_normals = np.cross(second_edges[:, 0], second_edges[:, 1])
    count = len(candidates)
    separated = np.zeros(count, dtype=bool)

    axes: list[np.ndarray] = [
        np.tile(first_normal, (count, 1)),
        second_normals,
    ]
    for first_edge in first_edges:
        for second_index in range(3):
            axes.append(np.cross(first_edge, second_edges[:, second_index]))

    # These in-plane edge normals make the test complete for coplanar pairs.
    for first_edge in first_edges:
        axes.append(np.tile(np.cross(first_normal, first_edge), (count, 1)))
    for second_index in range(3):
        axes.append(np.cross(second_normals, second_edges[:, second_index]))

    for axis in axes:
        separated |= _axis_separates(
            triangle, candidates, axis, tolerance
        )
        if np.all(separated):
            break
    return ~separated


def _find_collision_pairs(
    foot_triangles: np.ndarray,
    foot_face_indices: np.ndarray,
    obstacle_triangles: np.ndarray,
    obstacle_face_indices: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    obstacle_minimum = obstacle_triangles.min(axis=1)
    obstacle_maximum = obstacle_triangles.max(axis=1)
    pairs: list[np.ndarray] = []
    for triangle, foot_index in zip(foot_triangles, foot_face_indices):
        minimum = triangle.min(axis=0) - tolerance
        maximum = triangle.max(axis=0) + tolerance
        overlaps = np.all(obstacle_maximum >= minimum, axis=1) & np.all(
            obstacle_minimum <= maximum, axis=1
        )
        candidate_indices = np.flatnonzero(overlaps)
        if len(candidate_indices) == 0:
            continue
        hits = _triangle_intersections(
            triangle, obstacle_triangles[candidate_indices], tolerance
        )
        hit_faces = obstacle_face_indices[candidate_indices[hits]]
        if len(hit_faces):
            pairs.append(
                np.column_stack(
                    (
                        np.full(len(hit_faces), foot_index, dtype=np.int64),
                        hit_faces,
                    )
                )
            )
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.concatenate(pairs, axis=0), axis=0)


def _closest_points_on_triangles(
    points: np.ndarray,
    triangles: np.ndarray,
    source_face_indices: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find exact closest points using triangle AABBs as a safe accelerator."""

    triangle_minimum = triangles.min(axis=1)
    triangle_maximum = triangles.max(axis=1)
    closest_points = np.empty_like(points, dtype=np.float64)
    distances = np.empty(len(points), dtype=np.float64)
    nearest_faces = np.empty(len(points), dtype=np.int64)
    tolerance_squared = tolerance * tolerance

    for point_index, point in enumerate(points):
        outside = np.maximum(
            np.maximum(triangle_minimum - point, point - triangle_maximum),
            0.0,
        )
        lower_bounds_squared = np.einsum("ij,ij->i", outside, outside)
        first_index = int(np.argmin(lower_bounds_squared))
        first_closest = trimesh.triangles.closest_point(
            triangles[first_index : first_index + 1], point[None, :]
        )[0]
        first_distance_squared = float(np.sum((first_closest - point) ** 2))
        candidates = np.flatnonzero(
            lower_bounds_squared <= first_distance_squared + tolerance_squared
        )
        repeated = np.repeat(point[None, :], len(candidates), axis=0)
        candidate_closest = trimesh.triangles.closest_point(
            triangles[candidates], repeated
        )
        distance_squared = np.einsum(
            "ij,ij->i", candidate_closest - repeated, candidate_closest - repeated
        )
        minimum = float(distance_squared.min())
        ties = np.flatnonzero(
            distance_squared <= minimum + tolerance_squared
        )
        tied_source_faces = source_face_indices[candidates[ties]]
        chosen_tie = int(ties[np.argmin(tied_source_faces)])
        chosen = int(candidates[chosen_tie])
        closest_points[point_index] = candidate_closest[chosen_tie]
        distances[point_index] = float(np.sqrt(distance_squared[chosen_tie]))
        nearest_faces[point_index] = source_face_indices[chosen]
    return closest_points, distances, nearest_faces


def _support_record(
    mesh: TriangleMesh,
    points: np.ndarray,
    sample_indices: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    heights, valid = sample_triangle_mesh_y(mesh, points[:, (0, 2)])
    gaps = np.full(len(points), np.nan, dtype=np.float64)
    gaps[valid] = heights[valid] - points[valid, 1]
    covered = gaps[valid]
    penetrating = valid & (gaps < -tolerance)
    contacting = valid & (np.abs(gaps) <= tolerance)
    record: dict[str, Any] = {
        "sample_count": int(len(points)),
        "covered_sample_count": int(np.count_nonzero(valid)),
        "coverage": float(np.mean(valid)),
        "contacting_sample_count": int(np.count_nonzero(contacting)),
        "contacting_sample_indices": sample_indices[contacting].tolist(),
        "penetrating_sample_count": int(np.count_nonzero(penetrating)),
        "penetrating_sample_indices": sample_indices[penetrating].tolist(),
        "gaps": [
            None if not finite else float(value)
            for value, finite in zip(gaps, valid)
        ],
    }
    if len(covered):
        record.update(
            {
                "minimum_gap": float(covered.min()),
                "fifth_percentile_gap": float(np.percentile(covered, 5.0)),
                "median_gap": float(np.median(covered)),
                "maximum_gap": float(covered.max()),
            }
        )
    else:
        record.update(
            {
                "minimum_gap": None,
                "fifth_percentile_gap": None,
                "median_gap": None,
                "maximum_gap": None,
            }
        )
    return record


def _surface_normals(mesh: TriangleMesh) -> tuple[np.ndarray, np.ndarray]:
    triangles = mesh.vertices[mesh.faces]
    weighted_face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(weighted_face_normals, axis=1)
    face_normals = np.divide(
        weighted_face_normals,
        lengths[:, None],
        out=np.zeros_like(weighted_face_normals),
        where=lengths[:, None] > np.finfo(np.float64).eps,
    )
    vertex_normals = np.zeros_like(mesh.vertices)
    for corner in range(3):
        np.add.at(vertex_normals, mesh.faces[:, corner], weighted_face_normals)
    vertex_lengths = np.linalg.norm(vertex_normals, axis=1)
    vertex_normals = np.divide(
        vertex_normals,
        vertex_lengths[:, None],
        out=np.zeros_like(vertex_normals),
        where=vertex_lengths[:, None] > np.finfo(np.float64).eps,
    )
    return vertex_normals, face_normals


def _clearance_summary(distances: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(distances, dtype=np.float64)
    if len(values) == 0:
        return {
            "sample_count": 0,
            "minimum": None,
            "fifth_percentile": None,
            "median": None,
        }
    return {
        "sample_count": int(len(values)),
        "minimum": float(values.min()),
        "fifth_percentile": float(np.percentile(values, 5.0)),
        "median": float(np.median(values)),
    }


def _clearance_summaries(
    foot_mesh: TriangleMesh,
    vertex_clearances: np.ndarray,
    face_clearances: np.ndarray,
    colliding_foot_faces: np.ndarray,
) -> dict[str, Any]:
    face_centroids = foot_mesh.vertices[foot_mesh.faces].mean(axis=1)
    points = np.concatenate((foot_mesh.vertices, face_centroids), axis=0)
    distances = np.concatenate((vertex_clearances, face_clearances))
    vertex_normals, face_normals = _surface_normals(foot_mesh)
    normals = np.concatenate((vertex_normals, face_normals), axis=0)
    x_minimum = float(foot_mesh.bounds[0, 0])
    x_length = float(foot_mesh.extents[0])
    if x_length <= np.finfo(np.float64).eps:
        raise ValueError("fitted foot must have positive heel-to-toe length")
    fractions = (points[:, 0] - x_minimum) / x_length

    longitudinal: dict[str, Any] = {}
    colliding = set(int(index) for index in colliding_foot_faces)
    face_offset = len(foot_mesh.vertices)
    for name, (lower, upper) in CONTACT_REGION_RANGES.items():
        if name == "heel":
            mask = fractions <= upper
        else:
            mask = (fractions > lower) & (fractions <= upper)
        face_mask = mask[face_offset:]
        longitudinal[name] = {
            **_clearance_summary(distances[mask]),
            "intersecting_foot_face_count": int(
                sum(
                    bool(face_mask[index])
                    for index in colliding
                    if index < len(face_mask)
                )
            ),
        }

    dominant_axis = np.argmax(np.abs(normals), axis=1)
    surface_masks = {
        "top": (dominant_axis == 1) & (normals[:, 1] < 0.0),
        "medial": (dominant_axis == 2) & (normals[:, 2] < 0.0),
        "lateral": (dominant_axis == 2) & (normals[:, 2] > 0.0),
    }
    return {
        "overall": _clearance_summary(distances),
        "longitudinal": longitudinal,
        "surface": {
            name: _clearance_summary(distances[mask])
            for name, mask in surface_masks.items()
        },
        "surface_convention": {
            "top": "outward foot normal points toward -Y",
            "medial": "right-foot inward side; outward normal points toward -Z",
            "lateral": "right-foot outward side; outward normal points toward +Z",
        },
    }


def _combine_signed_clearances(*values: np.ndarray) -> np.ndarray:
    stacked = np.stack(values, axis=1)
    finite = np.isfinite(stacked)
    result = np.full(len(stacked), np.nan, dtype=np.float64)
    valid = np.any(finite, axis=1)
    result[valid] = np.min(
        np.where(finite[valid], stacked[valid], np.inf), axis=1
    )
    return result


def _signed_summary(values: np.ndarray, tolerance: float) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if len(finite) == 0:
        return {
            "bounded_sample_count": 0,
            "outside_sample_count": 0,
            "minimum_signed_clearance": None,
            "maximum_protrusion_depth": None,
            "median_protrusion_depth": None,
            "ninety_fifth_percentile_protrusion_depth": None,
        }
    protrusions = -finite[finite < -tolerance]
    return {
        "bounded_sample_count": int(len(finite)),
        "outside_sample_count": int(len(protrusions)),
        "minimum_signed_clearance": float(np.min(finite)),
        "maximum_protrusion_depth": (
            float(np.max(protrusions)) if len(protrusions) else 0.0
        ),
        "median_protrusion_depth": (
            float(np.median(protrusions)) if len(protrusions) else 0.0
        ),
        "ninety_fifth_percentile_protrusion_depth": (
            float(np.percentile(protrusions, 95.0)) if len(protrusions) else 0.0
        ),
    }


def _signed_regional_summaries(
    foot_mesh: TriangleMesh,
    vertex_signed: np.ndarray,
    face_signed: np.ndarray,
    tolerance: float,
) -> dict[str, Any]:
    face_centroids = foot_mesh.vertices[foot_mesh.faces].mean(axis=1)
    points = np.concatenate((foot_mesh.vertices, face_centroids), axis=0)
    clearances = np.concatenate((vertex_signed, face_signed))
    vertex_normals, face_normals = _surface_normals(foot_mesh)
    normals = np.concatenate((vertex_normals, face_normals), axis=0)
    x_minimum = float(foot_mesh.bounds[0, 0])
    x_length = float(foot_mesh.extents[0])
    fractions = (points[:, 0] - x_minimum) / x_length

    longitudinal: dict[str, Any] = {}
    for name, (lower, upper) in CONTACT_REGION_RANGES.items():
        mask = fractions <= upper if name == "heel" else (
            (fractions > lower) & (fractions <= upper)
        )
        longitudinal[name] = _signed_summary(clearances[mask], tolerance)

    dominant_axis = np.argmax(np.abs(normals), axis=1)
    surface_masks = {
        "top": (dominant_axis == 1) & (normals[:, 1] < 0.0),
        "medial": (dominant_axis == 2) & (normals[:, 2] < 0.0),
        "lateral": (dominant_axis == 2) & (normals[:, 2] > 0.0),
    }
    return {
        "overall": _signed_summary(clearances, tolerance),
        "longitudinal": longitudinal,
        "surface": {
            name: _signed_summary(clearances[mask], tolerance)
            for name, mask in surface_masks.items()
        },
    }


def _ray_triangle_parameters(
    origin: np.ndarray,
    direction: np.ndarray,
    triangles: np.ndarray,
    tolerance: float,
) -> np.ndarray:
    """Return forward ray parameters for triangle hits, NaN otherwise."""

    edge1 = triangles[:, 1] - triangles[:, 0]
    edge2 = triangles[:, 2] - triangles[:, 0]
    repeated_direction = np.broadcast_to(direction, edge2.shape)
    cross_direction = np.cross(repeated_direction, edge2)
    determinant = np.einsum("ij,ij->i", edge1, cross_direction)
    usable = np.abs(determinant) > np.finfo(np.float64).eps
    inverse = np.divide(
        1.0,
        determinant,
        out=np.zeros_like(determinant),
        where=usable,
    )
    offset = origin - triangles[:, 0]
    u = inverse * np.einsum("ij,ij->i", offset, cross_direction)
    cross_offset = np.cross(offset, edge1)
    v = inverse * np.einsum("ij,ij->i", repeated_direction, cross_offset)
    distance = inverse * np.einsum("ij,ij->i", edge2, cross_offset)
    barycentric_tolerance = max(1e-12, tolerance)
    hits = (
        usable
        & (u >= -barycentric_tolerance)
        & (v >= -barycentric_tolerance)
        & (u + v <= 1.0 + barycentric_tolerance)
        & (distance >= -tolerance)
    )
    return np.where(hits, np.maximum(distance, 0.0), np.nan)


def _projection_bins(
    triangle_minimum: np.ndarray,
    triangle_maximum: np.ndarray,
    axes: tuple[int, int],
    origin: np.ndarray,
    cell_size: float,
    shape: np.ndarray,
) -> tuple[dict[tuple[int, int], np.ndarray], np.ndarray]:
    bins: dict[tuple[int, int], list[int]] = {}
    broad: list[int] = []
    first_all = np.floor(
        (triangle_minimum[:, axes] - origin[list(axes)]) / cell_size
    ).astype(np.int64)
    last_all = np.floor(
        (triangle_maximum[:, axes] - origin[list(axes)]) / cell_size
    ).astype(np.int64)
    first_all = np.clip(first_all, 0, shape[list(axes)] - 1)
    last_all = np.clip(last_all, 0, shape[list(axes)] - 1)
    for index, (first, last) in enumerate(zip(first_all, last_all)):
        if int(np.prod(last - first + 1)) > 256:
            broad.append(index)
            continue
        for first_axis in range(int(first[0]), int(last[0]) + 1):
            for second_axis in range(int(first[1]), int(last[1]) + 1):
                bins.setdefault((first_axis, second_axis), []).append(index)
    return (
        {key: np.asarray(value, dtype=np.int64) for key, value in bins.items()},
        np.asarray(broad, dtype=np.int64),
    )


_SPATIAL_AXIS_CELLS = 32
_SPATIAL_MAX_CELLS_PER_TRIANGLE = 512


@dataclass(frozen=True)
class CavityEvaluator:
    """Prepared normal-shoe obstacles for repeated exact foot evaluation."""

    shoe_mesh: TriangleMesh
    footbed_mesh: TriangleMesh
    numerical_tolerance: float
    footbed_source_face_indices: np.ndarray
    obstacle_face_indices: np.ndarray
    obstacle_triangles: np.ndarray
    ignored_degenerate_obstacle_face_indices: np.ndarray
    spatial_origin: np.ndarray
    spatial_cell_size: float
    spatial_shape: np.ndarray
    spatial_bins: dict[tuple[int, int, int], np.ndarray]
    broad_obstacle_indices: np.ndarray
    normalized_centerline_xz: np.ndarray
    upper_projection_bins: dict[tuple[int, int], np.ndarray]
    upper_broad_indices: np.ndarray
    side_projection_bins: dict[tuple[int, int], np.ndarray]
    side_broad_indices: np.ndarray

    @classmethod
    def build(
        cls,
        shoe_mesh: TriangleMesh,
        footbed_mesh: TriangleMesh,
        footbed_source_face_indices: np.ndarray,
        reference_foot: TriangleMesh,
        normalized_centerline_xz: np.ndarray,
    ) -> "CavityEvaluator":
        """Validate and cache the fixed shoe geometry for repeated candidates."""

        centerline = np.asarray(normalized_centerline_xz, dtype=np.float64)
        if (
            centerline.ndim != 2
            or centerline.shape[1] != 2
            or len(centerline) < 2
            or not np.isfinite(centerline).all()
            or np.any(np.diff(centerline[:, 0]) <= 0.0)
        ):
            raise ValueError(
                "normalized_centerline_xz must be finite, strictly ordered, "
                "and have shape (N, 2)"
            )
        tolerance = _coordinate_tolerance(
            shoe_mesh, footbed_mesh, reference_foot
        )
        footbed_faces = _validate_indices(
            footbed_source_face_indices,
            len(shoe_mesh.faces),
            "footbed_source_face_indices",
        )
        _validate_footbed_faces(
            shoe_mesh, footbed_mesh, footbed_faces, tolerance
        )
        obstacle_faces = np.setdiff1d(
            np.arange(len(shoe_mesh.faces), dtype=np.int64),
            footbed_faces,
            assume_unique=True,
        )
        if len(obstacle_faces) == 0:
            raise ValueError("shoe contains no non-footbed obstacle faces")
        all_triangles = shoe_mesh.vertices[shoe_mesh.faces[obstacle_faces]]
        nondegenerate = (
            _triangle_double_areas(all_triangles) > tolerance * tolerance
        )
        ignored = obstacle_faces[~nondegenerate]
        obstacle_faces = obstacle_faces[nondegenerate]
        triangles = all_triangles[nondegenerate]
        if len(obstacle_faces) == 0:
            raise ValueError("shoe contains no nondegenerate obstacle faces")

        minimum = triangles.min(axis=(0, 1)) - tolerance
        maximum = triangles.max(axis=(0, 1)) + tolerance
        maximum_extent = float(np.max(maximum - minimum))
        if not np.isfinite(maximum_extent) or maximum_extent <= 0.0:
            raise ValueError("shoe obstacle bounds must have positive extent")
        cell_size = maximum_extent / _SPATIAL_AXIS_CELLS
        shape = np.maximum(
            1, np.ceil((maximum - minimum) / cell_size).astype(np.int64)
        )

        bins: dict[tuple[int, int, int], list[int]] = {}
        broad: list[int] = []
        triangle_minimum = triangles.min(axis=1) - tolerance
        triangle_maximum = triangles.max(axis=1) + tolerance
        for triangle_index, (lower, upper) in enumerate(
            zip(triangle_minimum, triangle_maximum)
        ):
            first = np.clip(
                np.floor((lower - minimum) / cell_size).astype(np.int64),
                0,
                shape - 1,
            )
            last = np.clip(
                np.floor((upper - minimum) / cell_size).astype(np.int64),
                0,
                shape - 1,
            )
            counts = last - first + 1
            if int(np.prod(counts)) > _SPATIAL_MAX_CELLS_PER_TRIANGLE:
                broad.append(triangle_index)
                continue
            for x_index in range(int(first[0]), int(last[0]) + 1):
                for y_index in range(int(first[1]), int(last[1]) + 1):
                    for z_index in range(int(first[2]), int(last[2]) + 1):
                        bins.setdefault((x_index, y_index, z_index), []).append(
                            triangle_index
                        )
        frozen_bins = {
            key: np.asarray(values, dtype=np.int64)
            for key, values in bins.items()
        }
        upper_bins, upper_broad = _projection_bins(
            triangle_minimum,
            triangle_maximum,
            (0, 2),
            minimum,
            cell_size,
            shape,
        )
        side_bins, side_broad = _projection_bins(
            triangle_minimum,
            triangle_maximum,
            (0, 1),
            minimum,
            cell_size,
            shape,
        )
        return cls(
            shoe_mesh=shoe_mesh,
            footbed_mesh=footbed_mesh,
            numerical_tolerance=tolerance,
            footbed_source_face_indices=footbed_faces,
            obstacle_face_indices=obstacle_faces,
            obstacle_triangles=triangles,
            ignored_degenerate_obstacle_face_indices=ignored,
            spatial_origin=minimum,
            spatial_cell_size=cell_size,
            spatial_shape=shape,
            spatial_bins=frozen_bins,
            broad_obstacle_indices=np.asarray(broad, dtype=np.int64),
            normalized_centerline_xz=centerline,
            upper_projection_bins=upper_bins,
            upper_broad_indices=upper_broad,
            side_projection_bins=side_bins,
            side_broad_indices=side_broad,
        )

    def _projected_candidates(
        self,
        point: np.ndarray,
        axes: tuple[int, int],
        bins: dict[tuple[int, int], np.ndarray],
        broad: np.ndarray,
    ) -> np.ndarray:
        coordinates = np.floor(
            (point[list(axes)] - self.spatial_origin[list(axes)])
            / self.spatial_cell_size
        ).astype(np.int64)
        if np.any(coordinates < 0) or np.any(
            coordinates >= self.spatial_shape[list(axes)]
        ):
            return np.empty(0, dtype=np.int64)
        pieces: list[np.ndarray] = []
        values = bins.get((int(coordinates[0]), int(coordinates[1])))
        if values is not None:
            pieces.append(values)
        if len(broad):
            pieces.append(broad)
        if not pieces:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(pieces))

    def _nearest_ray_hit(
        self,
        origin: np.ndarray,
        direction: np.ndarray,
        candidates: np.ndarray,
        minimum_distance: float = 0.0,
    ) -> tuple[float, int, np.ndarray] | None:
        if len(candidates) == 0:
            return None
        distances = _ray_triangle_parameters(
            origin,
            direction,
            self.obstacle_triangles[candidates],
            self.numerical_tolerance,
        )
        valid = np.flatnonzero(
            np.isfinite(distances) & (distances > minimum_distance)
        )
        if len(valid) == 0:
            return None
        minimum = float(np.min(distances[valid]))
        ties = valid[
            distances[valid] <= minimum + self.numerical_tolerance
        ]
        source_faces = self.obstacle_face_indices[candidates[ties]]
        chosen = int(ties[np.argmin(source_faces)])
        triangle_index = int(candidates[chosen])
        point = origin + distances[chosen] * direction
        return (
            float(distances[chosen]),
            int(self.obstacle_face_indices[triangle_index]),
            point,
        )

    def _upper_clearances(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        heights, supported = sample_triangle_mesh_y(
            self.footbed_mesh, points[:, (0, 2)]
        )
        clearances = np.full(len(points), np.nan, dtype=np.float64)
        faces = np.full(len(points), -1, dtype=np.int64)
        boundary_points = np.full((len(points), 3), np.nan, dtype=np.float64)
        direction = np.asarray([0.0, -1.0, 0.0], dtype=np.float64)
        for index in np.flatnonzero(supported):
            origin = np.asarray(
                [points[index, 0], heights[index], points[index, 2]],
                dtype=np.float64,
            )
            candidates = self._projected_candidates(
                origin,
                (0, 2),
                self.upper_projection_bins,
                self.upper_broad_indices,
            )
            hit = self._nearest_ray_hit(
                origin,
                direction,
                candidates,
                minimum_distance=8.0 * self.numerical_tolerance,
            )
            if hit is None:
                continue
            _, source_face, boundary = hit
            clearances[index] = points[index, 1] - boundary[1]
            faces[index] = source_face
            boundary_points[index] = boundary
        return clearances, faces, boundary_points

    def _side_clearances(
        self, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        clearances = np.full(len(points), np.nan, dtype=np.float64)
        faces = np.full(len(points), -1, dtype=np.int64)
        boundary_points = np.full((len(points), 3), np.nan, dtype=np.float64)
        x_values = points[:, 0]
        covered = (
            (x_values >= self.normalized_centerline_xz[0, 0])
            & (x_values <= self.normalized_centerline_xz[-1, 0])
        )
        center_z = np.interp(
            np.clip(
                x_values,
                self.normalized_centerline_xz[0, 0],
                self.normalized_centerline_xz[-1, 0],
            ),
            self.normalized_centerline_xz[:, 0],
            self.normalized_centerline_xz[:, 1],
        )
        for index in np.flatnonzero(covered):
            origin = np.asarray(
                [points[index, 0], points[index, 1], center_z[index]],
                dtype=np.float64,
            )
            candidates = self._projected_candidates(
                origin,
                (0, 1),
                self.side_projection_bins,
                self.side_broad_indices,
            )
            difference = float(points[index, 2] - center_z[index])
            directions = (-1.0, 1.0) if abs(difference) <= self.numerical_tolerance else (
                float(np.sign(difference)),
            )
            hits: list[tuple[float, int, np.ndarray]] = []
            for sign in directions:
                hit = self._nearest_ray_hit(
                    origin,
                    np.asarray([0.0, 0.0, sign], dtype=np.float64),
                    candidates,
                    minimum_distance=8.0 * self.numerical_tolerance,
                )
                if hit is not None:
                    hits.append(hit)
            if not hits:
                continue
            hit = min(hits, key=lambda item: (item[0], item[1]))
            distance, source_face, boundary = hit
            clearances[index] = distance - abs(difference)
            faces[index] = source_face
            boundary_points[index] = boundary
        return clearances, faces, boundary_points

    def signed_clearances(
        self,
        fitted_foot: TriangleMesh,
        signed_exempt_vertex_indices: np.ndarray | None = None,
        signed_exempt_face_indices: np.ndarray | None = None,
    ) -> SignedCavityClearance:
        """Measure local upper and side clearance without closing openings."""

        face_triangles = fitted_foot.vertices[fitted_foot.faces]
        face_centroids = face_triangles.mean(axis=1)
        exempt_vertices = _validate_indices(
            np.empty(0, dtype=np.int64)
            if signed_exempt_vertex_indices is None
            else signed_exempt_vertex_indices,
            len(fitted_foot.vertices),
            "signed_exempt_vertex_indices",
            allow_empty=True,
        )
        exempt_faces = _validate_indices(
            np.empty(0, dtype=np.int64)
            if signed_exempt_face_indices is None
            else signed_exempt_face_indices,
            len(fitted_foot.faces),
            "signed_exempt_face_indices",
            allow_empty=True,
        )
        vertex_upper, vertex_upper_faces, vertex_upper_points = (
            self._upper_clearances(fitted_foot.vertices)
        )
        vertex_side, vertex_side_faces, vertex_side_points = (
            self._side_clearances(fitted_foot.vertices)
        )
        face_upper, face_upper_faces, face_upper_points = (
            self._upper_clearances(face_centroids)
        )
        face_side, face_side_faces, face_side_points = (
            self._side_clearances(face_centroids)
        )
        vertex_combined = _combine_signed_clearances(
            vertex_upper, vertex_side
        )
        face_combined = _combine_signed_clearances(face_upper, face_side)
        vertex_combined[exempt_vertices] = np.nan
        face_combined[exempt_faces] = np.nan

        corner_values = vertex_combined[fitted_foot.faces]
        samples = np.concatenate((face_combined[:, None], corner_values), axis=1)
        finite = np.isfinite(samples)
        conservative = np.full(len(samples), np.nan, dtype=np.float64)
        any_finite = np.any(finite, axis=1)
        conservative[any_finite] = np.min(
            np.where(finite[any_finite], samples[any_finite], np.inf), axis=1
        )
        outside_faces = np.flatnonzero(
            conservative < -self.numerical_tolerance
        ).astype(np.int64)
        outside_vertices = np.flatnonzero(
            vertex_combined < -self.numerical_tolerance
        ).astype(np.int64)
        areas = 0.5 * _triangle_double_areas(face_triangles)
        total_area = float(np.sum(areas))
        outside_area = float(np.sum(areas[outside_faces]))
        violation_squared = np.where(
            finite, np.maximum(0.0, -samples) ** 2, 0.0
        )
        sample_counts = np.maximum(1, np.sum(finite, axis=1))
        face_energy = np.sum(violation_squared, axis=1) / sample_counts
        protrusion_energy = float(np.sum(areas * face_energy) / total_area)
        all_combined = np.concatenate((vertex_combined, face_combined))
        return SignedCavityClearance(
            vertex_upper_clearances=vertex_upper,
            vertex_upper_faces=vertex_upper_faces,
            vertex_upper_points=vertex_upper_points,
            vertex_side_clearances=vertex_side,
            vertex_side_faces=vertex_side_faces,
            vertex_side_points=vertex_side_points,
            vertex_combined_clearances=vertex_combined,
            face_upper_clearances=face_upper,
            face_upper_faces=face_upper_faces,
            face_upper_points=face_upper_points,
            face_side_clearances=face_side,
            face_side_faces=face_side_faces,
            face_side_points=face_side_points,
            face_combined_clearances=face_combined,
            signed_exempt_vertex_indices=exempt_vertices,
            signed_exempt_face_indices=exempt_faces,
            outside_vertex_indices=outside_vertices,
            outside_face_indices=outside_faces,
            outside_area=outside_area,
            outside_area_fraction=outside_area / total_area,
            protrusion_energy=protrusion_energy,
            protrusion_statistics=_signed_summary(
                all_combined, self.numerical_tolerance
            ),
            regional_summaries=_signed_regional_summaries(
                fitted_foot,
                vertex_combined,
                face_combined,
                self.numerical_tolerance,
            ),
        )

    def _spatial_candidates(self, triangle: np.ndarray) -> np.ndarray:
        lower = triangle.min(axis=0) - self.numerical_tolerance
        upper = triangle.max(axis=0) + self.numerical_tolerance
        grid_upper = (
            self.spatial_origin
            + self.spatial_shape * self.spatial_cell_size
        )
        if np.any(upper < self.spatial_origin) or np.any(lower > grid_upper):
            return np.empty(0, dtype=np.int64)
        first = np.clip(
            np.floor(
                (lower - self.spatial_origin) / self.spatial_cell_size
            ).astype(np.int64),
            0,
            self.spatial_shape - 1,
        )
        last = np.clip(
            np.floor(
                (upper - self.spatial_origin) / self.spatial_cell_size
            ).astype(np.int64),
            0,
            self.spatial_shape - 1,
        )
        pieces: list[np.ndarray] = []
        if len(self.broad_obstacle_indices):
            pieces.append(self.broad_obstacle_indices)
        for x_index in range(int(first[0]), int(last[0]) + 1):
            for y_index in range(int(first[1]), int(last[1]) + 1):
                for z_index in range(int(first[2]), int(last[2]) + 1):
                    values = self.spatial_bins.get(
                        (x_index, y_index, z_index)
                    )
                    if values is not None:
                        pieces.append(values)
        if not pieces:
            return np.empty(0, dtype=np.int64)
        return np.unique(np.concatenate(pieces))

    def collision_pairs(
        self, fitted_foot: TriangleMesh
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return exact obstacle contacts and ignored degenerate foot faces."""

        tolerance = self.numerical_tolerance
        triangles = fitted_foot.vertices[fitted_foot.faces]
        nondegenerate = (
            _triangle_double_areas(triangles) > tolerance * tolerance
        )
        foot_faces = np.flatnonzero(nondegenerate).astype(np.int64)
        ignored = np.flatnonzero(~nondegenerate).astype(np.int64)
        obstacle_minimum = self.obstacle_triangles.min(axis=1)
        obstacle_maximum = self.obstacle_triangles.max(axis=1)
        pairs: list[np.ndarray] = []
        for triangle, foot_index in zip(
            triangles[nondegenerate], foot_faces
        ):
            candidates = self._spatial_candidates(triangle)
            if len(candidates) == 0:
                continue
            minimum = triangle.min(axis=0) - tolerance
            maximum = triangle.max(axis=0) + tolerance
            overlaps = np.all(
                obstacle_maximum[candidates] >= minimum, axis=1
            ) & np.all(obstacle_minimum[candidates] <= maximum, axis=1)
            candidates = candidates[overlaps]
            if len(candidates) == 0:
                continue
            hits = _triangle_intersections(
                triangle, self.obstacle_triangles[candidates], tolerance
            )
            hit_faces = self.obstacle_face_indices[candidates[hits]]
            if len(hit_faces):
                pairs.append(
                    np.column_stack(
                        (
                            np.full(
                                len(hit_faces), foot_index, dtype=np.int64
                            ),
                            hit_faces,
                        )
                    )
                )
        if not pairs:
            return np.empty((0, 2), dtype=np.int64), ignored
        return np.unique(np.concatenate(pairs, axis=0), axis=0), ignored

    def analyze(
        self,
        fitted_foot: TriangleMesh,
        plantar_vertex_indices: np.ndarray,
        plantar_face_indices: np.ndarray,
        signed_exempt_vertex_indices: np.ndarray | None = None,
        signed_exempt_face_indices: np.ndarray | None = None,
    ) -> CavityAnalysis:
        """Run the complete Checkpoint 6 analysis for one fitted candidate."""

        plantar_vertices = _validate_indices(
            plantar_vertex_indices,
            len(fitted_foot.vertices),
            "plantar_vertex_indices",
        )
        plantar_faces = _validate_indices(
            plantar_face_indices,
            len(fitted_foot.faces),
            "plantar_face_indices",
        )
        collision_pairs, ignored_foot = self.collision_pairs(fitted_foot)
        foot_triangles = fitted_foot.vertices[fitted_foot.faces]
        vertex_closest, vertex_distances, vertex_nearest = (
            _closest_points_on_triangles(
                fitted_foot.vertices,
                self.obstacle_triangles,
                self.obstacle_face_indices,
                self.numerical_tolerance,
            )
        )
        face_centroids = foot_triangles.mean(axis=1)
        face_closest, face_distances, face_nearest = (
            _closest_points_on_triangles(
                face_centroids,
                self.obstacle_triangles,
                self.obstacle_face_indices,
                self.numerical_tolerance,
            )
        )
        support_contact = {
            "plantar_vertices": _support_record(
                self.footbed_mesh,
                fitted_foot.vertices[plantar_vertices],
                plantar_vertices,
                self.numerical_tolerance,
            ),
            "plantar_face_centroids": _support_record(
                self.footbed_mesh,
                face_centroids[plantar_faces],
                plantar_faces,
                self.numerical_tolerance,
            ),
        }
        signed = self.signed_clearances(
            fitted_foot,
            signed_exempt_vertex_indices,
            signed_exempt_face_indices,
        )
        return CavityAnalysis(
            numerical_tolerance=self.numerical_tolerance,
            footbed_source_face_indices=self.footbed_source_face_indices,
            obstacle_face_count=int(len(self.obstacle_face_indices)),
            ignored_degenerate_obstacle_face_indices=(
                self.ignored_degenerate_obstacle_face_indices
            ),
            ignored_degenerate_foot_face_indices=ignored_foot,
            support_contact=support_contact,
            collision_pairs=collision_pairs,
            vertex_closest_points=vertex_closest,
            vertex_vectors_to_obstacle=(
                vertex_closest - fitted_foot.vertices
            ),
            vertex_clearances=vertex_distances,
            vertex_nearest_shoe_faces=vertex_nearest,
            face_closest_points=face_closest,
            face_vectors_to_obstacle=face_closest - face_centroids,
            face_clearances=face_distances,
            face_nearest_shoe_faces=face_nearest,
            clearance_summaries=_clearance_summaries(
                fitted_foot,
                vertex_distances,
                face_distances,
                np.unique(collision_pairs[:, 0])
                if len(collision_pairs)
                else np.empty(0, dtype=np.int64),
            ),
            signed_clearance=signed,
        )


def analyze_fitted_foot_cavity(
    shoe_mesh: TriangleMesh,
    footbed_mesh: TriangleMesh,
    fitted_foot: TriangleMesh,
    footbed_source_face_indices: np.ndarray,
    plantar_vertex_indices: np.ndarray,
    plantar_face_indices: np.ndarray,
    normalized_centerline_xz: np.ndarray,
) -> CavityAnalysis:
    """Measure allowed support contact and forbidden shoe-surface collisions."""

    evaluator = CavityEvaluator.build(
        shoe_mesh,
        footbed_mesh,
        footbed_source_face_indices,
        fitted_foot,
        normalized_centerline_xz,
    )
    return evaluator.analyze(
        fitted_foot, plantar_vertex_indices, plantar_face_indices
    )
