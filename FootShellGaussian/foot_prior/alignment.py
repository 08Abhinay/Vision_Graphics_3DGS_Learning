"""Deterministic SUPR placement and support fitting in normalized shoes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .mesh import TriangleMesh, sample_triangle_mesh_y
from .supr_foot import (
    SUPR_ANKLE_PITCH_INDEX,
    SUPR_MIDFOOT_PITCH_INDEX,
    SuprFootModel,
)


PLANTAR_NORMAL_Y_MIN = float(np.cos(np.deg2rad(45.0)))
MIN_PLANTAR_SUPPORT_COVERAGE = 0.95

# Shoe-size policy, stated once. A normalized shoe's functional length is
# defined to admit a neutral 250 mm foot with 12.5 mm of toe allowance, which
# fixes the SUPR-to-normalized-shoe scale for every candidate. Foot length is
# then an output of the SUPR shape parameters, never a search variable.
REFERENCE_FOOT_LENGTH_MM = 250.0
ANCHOR_TOE_ALLOWANCE_MM = 12.5
SHOE_FUNCTIONAL_LENGTH_MM = REFERENCE_FOOT_LENGTH_MM + ANCHOR_TOE_ALLOWANCE_MM
ANCHOR_FOOT_LENGTH_RATIO = REFERENCE_FOOT_LENGTH_MM / SHOE_FUNCTIONAL_LENGTH_MM

# The front allowance is deliberately one-sided. A foot may not run past the
# functional toe, so MIN_TOE_ALLOWANCE_MM rejects. A shorter foot only leaves
# more room, so MAX_TOE_ALLOWANCE_MM reports and never rejects.
MIN_TOE_ALLOWANCE_MM = 10.0
MAX_TOE_ALLOWANCE_MM = 15.0
MIN_FOOT_LENGTH_RATIO = 0.80

MIN_HEEL_SUPPORT_COVERAGE = 0.95
MIN_FOREFOOT_SUPPORT_COVERAGE = 0.95
MIN_TOE_SUPPORT_COVERAGE = 0.90
MAX_TRIANGLE_AREA_DISTORTION = 1.5

CONTACT_REGION_RANGES = {
    "heel": (0.0, 0.18),
    "arch": (0.18, 0.55),
    "forefoot": (0.55, 0.80),
    "toes": (0.80, 1.0),
}


def toe_allowance_to_foot_length_ratio(toe_allowance_mm: float) -> float:
    """Convert physical front allowance to normalized heel-to-toe length.

    Both conversions read the anchored shoe: one unit of normalized functional
    length is SHOE_FUNCTIONAL_LENGTH_MM millimetres regardless of how long the
    fitted foot turns out to be.
    """

    allowance = float(toe_allowance_mm)
    if not np.isfinite(allowance) or allowance < 0.0:
        raise ValueError("toe_allowance_mm must be finite and nonnegative")
    return 1.0 - allowance / SHOE_FUNCTIONAL_LENGTH_MM


def foot_length_ratio_to_toe_allowance(
    foot_length_ratio: float,
    heel_offset_x: float = 0.0,
) -> float:
    """Return front allowance implied by a normalized length and heel position."""

    ratio = float(foot_length_ratio)
    heel = float(heel_offset_x)
    if not np.isfinite(ratio) or ratio <= 0.0:
        raise ValueError("foot_length_ratio must be finite and positive")
    if not np.isfinite(heel):
        raise ValueError("heel_offset_x must be finite")
    return SHOE_FUNCTIONAL_LENGTH_MM * (1.0 - heel - ratio)


@dataclass(frozen=True)
class SuprContactRegions:
    """Stable neutral-template indices for plantar support measurements."""

    plantar_vertex_indices: np.ndarray
    plantar_face_indices: np.ndarray
    vertex_regions: dict[str, np.ndarray]
    face_regions: dict[str, np.ndarray]

    def to_dict(self) -> dict[str, Any]:
        return {
            "normal_y_minimum": PLANTAR_NORMAL_Y_MIN,
            "length_ranges": {
                name: list(bounds) for name, bounds in CONTACT_REGION_RANGES.items()
            },
            "plantar_vertex_indices": self.plantar_vertex_indices.tolist(),
            "plantar_face_indices": self.plantar_face_indices.tolist(),
            "vertex_regions": {
                name: indices.tolist()
                for name, indices in self.vertex_regions.items()
            },
            "face_regions": {
                name: indices.tolist()
                for name, indices in self.face_regions.items()
            },
        }


@dataclass(frozen=True)
class SupportFootFit:
    """Selected articulated SUPR pose and rigid mapping into one shoe."""

    pose_parameters: np.ndarray
    betas: np.ndarray
    ankle_pitch_degrees: float
    midfoot_pitch_degrees: float
    posed_vertices: np.ndarray
    posed_joints: np.ndarray
    aligned_vertices: np.ndarray
    aligned_joints: np.ndarray
    regions: SuprContactRegions
    posed_supr_to_normalized_shoe: np.ndarray
    normalized_shoe_to_posed_supr: np.ndarray
    posed_supr_to_original_shoe: np.ndarray
    original_shoe_to_posed_supr: np.ndarray
    shoe_to_normalized: np.ndarray
    normalized_to_shoe: np.ndarray
    axis_remap: np.ndarray
    scale: float
    translation: np.ndarray
    reference_foot_length_mm: float
    toe_allowance_mm: float
    foot_length_ratio: float
    support_grid_cell_spacing: float
    search: dict[str, Any]
    neutral_comparison: dict[str, Any]
    region_contact: dict[str, dict[str, Any]]
    plantar_vertex_contact: dict[str, Any]
    lateral_fit: dict[str, float | int]
    distortion: dict[str, float | int]
    normalized_shoe_bounds: np.ndarray
    normalized_support_bounds: np.ndarray
    aligned_foot_bounds: np.ndarray

    def posed_points_to_normalized_shoe(self, points: np.ndarray) -> np.ndarray:
        return transform_points(points, self.posed_supr_to_normalized_shoe)

    def normalized_shoe_points_to_posed_supr(self, points: np.ndarray) -> np.ndarray:
        return transform_points(points, self.normalized_shoe_to_posed_supr)

    def posed_points_to_original_shoe(self, points: np.ndarray) -> np.ndarray:
        return transform_points(points, self.posed_supr_to_original_shoe)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible support-fit record without input paths."""

        return {
            "coordinate_conventions": {
                "posed_supr": {
                    "x": "foot_width",
                    "y": "anatomical_height",
                    "z": "foot_length_positive_heel_to_toe",
                },
                "normalized_shoe": {
                    "x": "functional_heel_to_toe_with_heel_at_zero",
                    "y": "vertical_positive_down_toward_sole",
                    "z": "shoe_width",
                },
            },
            "supr": {
                "pose_parameters_radians": self.pose_parameters.tolist(),
                "betas": self.betas.tolist(),
                "active_pose_indices": {
                    "ankle_pitch": SUPR_ANKLE_PITCH_INDEX,
                    "midfoot_pitch": SUPR_MIDFOOT_PITCH_INDEX,
                },
                "selected_angles_degrees": {
                    "ankle_pitch": self.ankle_pitch_degrees,
                    "midfoot_pitch": self.midfoot_pitch_degrees,
                },
                "posed_joints": self.posed_joints.tolist(),
                "aligned_joints": self.aligned_joints.tolist(),
                "neutral_to_posed_note": (
                    "SUPR articulation is non-rigid; reproduce it from the pose "
                    "parameters, zero betas, and stable vertex IDs"
                ),
            },
            "sizing": {
                "reference_foot_length_mm": self.reference_foot_length_mm,
                "shoe_functional_length_mm": SHOE_FUNCTIONAL_LENGTH_MM,
                "anchor_toe_allowance_mm": ANCHOR_TOE_ALLOWANCE_MM,
                "anchor_foot_length_ratio": ANCHOR_FOOT_LENGTH_RATIO,
                "foot_length_ratio": self.foot_length_ratio,
                "toe_allowance_mm": self.toe_allowance_mm,
                "standard_toe_allowance_mm": [
                    MIN_TOE_ALLOWANCE_MM,
                    MAX_TOE_ALLOWANCE_MM,
                ],
                "minimum_toe_allowance_mm": MIN_TOE_ALLOWANCE_MM,
                "minimum_foot_length_ratio": MIN_FOOT_LENGTH_RATIO,
            },
            "placement": {
                "scale": self.scale,
                "translation": self.translation.tolist(),
            },
            "transforms": {
                "axis_remap": self.axis_remap.tolist(),
                "posed_supr_to_normalized_shoe": (
                    self.posed_supr_to_normalized_shoe.tolist()
                ),
                "normalized_shoe_to_posed_supr": (
                    self.normalized_shoe_to_posed_supr.tolist()
                ),
                "posed_supr_to_original_shoe": (
                    self.posed_supr_to_original_shoe.tolist()
                ),
                "original_shoe_to_posed_supr": (
                    self.original_shoe_to_posed_supr.tolist()
                ),
                "shoe_to_normalized": self.shoe_to_normalized.tolist(),
                "normalized_to_shoe": self.normalized_to_shoe.tolist(),
            },
            "contact_regions": self.regions.to_dict(),
            "support_contact": {
                "face_centroids_by_region": self.region_contact,
                "plantar_vertices": self.plantar_vertex_contact,
            },
            "lateral_centerline_fit": self.lateral_fit,
            "mesh_distortion": self.distortion,
            "search": self.search,
            "neutral_comparison": self.neutral_comparison,
            "support_grid_cell_spacing": self.support_grid_cell_spacing,
            "bounds": {
                "normalized_shoe": self.normalized_shoe_bounds.tolist(),
                "normalized_support": self.normalized_support_bounds.tolist(),
                "aligned_foot": self.aligned_foot_bounds.tolist(),
            },
        }


@dataclass(frozen=True)
class SupportPlacementCandidate:
    """One valid posed-and-shaped SUPR placement on the saved support."""

    ankle_degrees: float
    midfoot_degrees: float
    pose: np.ndarray
    betas: np.ndarray
    posed_vertices: np.ndarray
    posed_joints: np.ndarray
    aligned_vertices: np.ndarray
    transform: np.ndarray
    scale: float
    translation: np.ndarray
    region_contact: dict[str, dict[str, Any]]
    plantar_vertex_contact: dict[str, Any]
    lateral_fit: dict[str, float | int]
    distortion: dict[str, float | int]
    primary_score: float
    heel_forefoot_sum: float
    toe_allowance_mm: float
    foot_length_ratio: float
    toe_x: float
    heel_offset_x: float
    lateral_offset_z: float

    def summary(self) -> dict[str, Any]:
        return {
            "ankle_pitch_degrees": self.ankle_degrees,
            "midfoot_pitch_degrees": self.midfoot_degrees,
            "toe_allowance_mm": self.toe_allowance_mm,
            "foot_length_ratio": self.foot_length_ratio,
            "toe_x": self.toe_x,
            "heel_offset_x": self.heel_offset_x,
            "lateral_offset_z": self.lateral_offset_z,
            "primary_contact_score": self.primary_score,
            "heel_plus_forefoot_rms": self.heel_forefoot_sum,
            "region_contact": self.region_contact,
        }


@dataclass(frozen=True)
class SupportPlacementContext:
    """Shoe and neutral-SUPR data shared by support-placement candidates."""

    faces: np.ndarray
    regions: SuprContactRegions
    neutral_crosses: np.ndarray
    neutral_areas: np.ndarray
    normalized_shoe_bounds: np.ndarray
    normalized_support_mesh: TriangleMesh
    centerline: np.ndarray
    length_scale: float


@dataclass(frozen=True)
class InitialFootPlacement:
    """Reversible raw-SUPR placement in normalized and original shoe frames."""

    foot_to_normalized_shoe: np.ndarray
    normalized_shoe_to_foot: np.ndarray
    shoe_to_normalized: np.ndarray
    normalized_to_shoe: np.ndarray
    foot_to_original_shoe: np.ndarray
    original_shoe_to_foot: np.ndarray
    axis_remap: np.ndarray
    scale: float
    translation: np.ndarray
    requested_plantar_length_ratio: float
    achieved_plantar_length_ratio: float
    plantar_face_indices: np.ndarray
    plantar_vertex_indices: np.ndarray
    heel_vertex_indices: np.ndarray
    heel_reference_remapped: np.ndarray
    heel_reference_normalized: np.ndarray
    remapped_plantar_bounds: np.ndarray
    input_foot_bounds: np.ndarray
    input_foot_extents: np.ndarray
    normalized_shoe_bounds: np.ndarray
    normalized_shoe_extents: np.ndarray
    normalized_support_bounds: np.ndarray
    normalized_support_extents: np.ndarray
    aligned_foot_bounds: np.ndarray
    aligned_foot_extents: np.ndarray
    lateral_face_count: int
    lateral_projected_area: float
    lateral_rms_before: float
    lateral_rms_after: float
    plantar_sample_count: int
    covered_plantar_sample_count: int
    plantar_support_coverage: float
    minimum_support_gap: float
    median_support_gap: float
    maximum_support_gap: float

    def foot_points_to_normalized_shoe(self, points: np.ndarray) -> np.ndarray:
        """Map raw SUPR points into the normalized shoe frame."""

        return transform_points(points, self.foot_to_normalized_shoe)

    def normalized_shoe_points_to_foot(self, points: np.ndarray) -> np.ndarray:
        """Map normalized-shoe points back into the raw SUPR frame."""

        return transform_points(points, self.normalized_shoe_to_foot)

    def foot_points_to_original_shoe(self, points: np.ndarray) -> np.ndarray:
        """Map raw SUPR points directly into the original shoe frame."""

        return transform_points(points, self.foot_to_original_shoe)

    def original_shoe_points_to_foot(self, points: np.ndarray) -> np.ndarray:
        """Map original-shoe points back into the raw SUPR frame."""

        return transform_points(points, self.original_shoe_to_foot)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible placement record without input paths."""

        return {
            "coordinate_conventions": {
                "foot_input": {
                    "x": "foot_width",
                    "y": "anatomical_height",
                    "z": "foot_length_positive_heel_to_toe",
                },
                "normalized_shoe": {
                    "x": "functional_heel_to_toe_with_heel_at_zero",
                    "y": "vertical_positive_down_toward_sole",
                    "z": "shoe_width",
                },
            },
            "transforms": {
                "axis_remap": self.axis_remap.tolist(),
                "foot_to_normalized_shoe": self.foot_to_normalized_shoe.tolist(),
                "normalized_shoe_to_foot": self.normalized_shoe_to_foot.tolist(),
                "shoe_to_normalized": self.shoe_to_normalized.tolist(),
                "normalized_to_shoe": self.normalized_to_shoe.tolist(),
                "foot_to_original_shoe": self.foot_to_original_shoe.tolist(),
                "original_shoe_to_foot": self.original_shoe_to_foot.tolist(),
            },
            "placement": {
                "scale": self.scale,
                "translation": self.translation.tolist(),
                "requested_plantar_length_ratio": (
                    self.requested_plantar_length_ratio
                ),
                "achieved_plantar_length_ratio": (
                    self.achieved_plantar_length_ratio
                ),
                "heel_reference_remapped": self.heel_reference_remapped.tolist(),
                "heel_reference_normalized": (
                    self.heel_reference_normalized.tolist()
                ),
                "remapped_plantar_bounds": self.remapped_plantar_bounds.tolist(),
            },
            "supr_plantar_region": {
                "normal_y_minimum": PLANTAR_NORMAL_Y_MIN,
                "face_indices": self.plantar_face_indices.tolist(),
                "vertex_indices": self.plantar_vertex_indices.tolist(),
                "heel_vertex_indices": self.heel_vertex_indices.tolist(),
            },
            "lateral_centerline_fit": {
                "face_count": self.lateral_face_count,
                "projected_area": self.lateral_projected_area,
                "rms_before_translation": self.lateral_rms_before,
                "rms_after_translation": self.lateral_rms_after,
            },
            "plantar_contact": {
                "minimum_required_coverage": MIN_PLANTAR_SUPPORT_COVERAGE,
                "sample_count": self.plantar_sample_count,
                "covered_sample_count": self.covered_plantar_sample_count,
                "coverage": self.plantar_support_coverage,
                "minimum_gap": self.minimum_support_gap,
                "median_gap": self.median_support_gap,
                "maximum_gap": self.maximum_support_gap,
            },
            "bounds": {
                "input_foot": self.input_foot_bounds.tolist(),
                "normalized_shoe": self.normalized_shoe_bounds.tolist(),
                "normalized_support": self.normalized_support_bounds.tolist(),
                "aligned_foot": self.aligned_foot_bounds.tolist(),
            },
            "extents": {
                "input_foot": self.input_foot_extents.tolist(),
                "normalized_shoe": self.normalized_shoe_extents.tolist(),
                "normalized_support": self.normalized_support_extents.tolist(),
                "aligned_foot": self.aligned_foot_extents.tolist(),
            },
        }


def make_supr_to_shoe_axis_remap() -> np.ndarray:
    """Return the fixed SUPR-width/height/length to shoe-X/Y/Z remap."""

    return np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Transform an N-by-3 point array with a finite homogeneous matrix."""

    values = np.asarray(points, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1:] != (3,):
        raise ValueError("points must have shape (N, 3)")
    if not np.isfinite(values).all():
        raise ValueError("points must contain only finite values")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("matrix must be a finite 4x4 array")
    homogeneous = np.column_stack(
        (values, np.ones(len(values), dtype=np.float64))
    )
    transformed = homogeneous @ transform.T
    if np.any(np.isclose(transformed[:, 3], 0.0)):
        raise ValueError(
            "matrix maps at least one point to invalid homogeneous coordinates"
        )
    return transformed[:, :3] / transformed[:, 3, None]


def neutral_length_scale(neutral_foot_mesh: TriangleMesh) -> float:
    """Return the fixed SUPR-to-normalized-shoe scale set by the anchor.

    The scale is derived from the neutral template alone, so a candidate's own
    pose and shape never change it. Foot length in the shoe is therefore an
    output of the SUPR parameters rather than something normalized away.
    """

    remapped = transform_points(
        neutral_foot_mesh.vertices, make_supr_to_shoe_axis_remap()
    )
    length = float(np.ptp(remapped[:, 0]))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("neutral SUPR foot must have positive remapped X length")
    return ANCHOR_FOOT_LENGTH_RATIO / length


def _uniform_scale_matrix(scale: float) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = scale
    matrix[1, 1] = scale
    matrix[2, 2] = scale
    return matrix


def _translation_matrix(translation: np.ndarray) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, 3] = translation
    return matrix


def _validated_inverse_pair(
    forward: np.ndarray, inverse: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(forward, dtype=np.float64)
    second = np.asarray(inverse, dtype=np.float64)
    if first.shape != (4, 4) or second.shape != (4, 4):
        raise ValueError("shoe normalization matrices must have shape (4, 4)")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("shoe normalization matrices must be finite")
    if not np.allclose(second @ first, np.eye(4), atol=1e-10, rtol=0.0):
        raise ValueError("shoe normalization matrices are not mutual inverses")
    return first, second


def _validated_centerline(centerline_xz: np.ndarray) -> np.ndarray:
    centerline = np.asarray(centerline_xz, dtype=np.float64)
    if centerline.ndim != 2 or centerline.shape[1:] != (2,):
        raise ValueError("normalized footbed centerline must have shape (N, 2)")
    if len(centerline) < 2 or not np.isfinite(centerline).all():
        raise ValueError(
            "normalized footbed centerline must contain at least two finite points"
        )
    if np.any(np.diff(centerline[:, 0]) <= 0.0):
        raise ValueError("normalized footbed centerline X must increase strictly")
    return centerline


def _face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(crosses, axis=1)
    return np.divide(
        crosses,
        lengths[:, None],
        out=np.zeros_like(crosses),
        where=lengths[:, None] > 0.0,
    )


def build_initial_placement(
    foot_mesh: TriangleMesh,
    normalized_shoe_mesh: TriangleMesh,
    normalized_support_mesh: TriangleMesh,
    normalized_centerline_xz: np.ndarray,
    shoe_to_normalized: np.ndarray,
    normalized_to_shoe: np.ndarray,
    foot_length_ratio: float = 0.85,
) -> InitialFootPlacement:
    """Place a neutral SUPR foot at first contact in a normalized normal shoe."""

    ratio = float(foot_length_ratio)
    if not np.isfinite(ratio) or not 0.0 < ratio <= 1.0:
        raise ValueError("foot_length_ratio must be finite and lie in (0, 1]")
    shoe_to_normalized, normalized_to_shoe = _validated_inverse_pair(
        shoe_to_normalized, normalized_to_shoe
    )
    centerline = _validated_centerline(normalized_centerline_xz)

    axis_remap = make_supr_to_shoe_axis_remap()
    remapped_vertices = transform_points(foot_mesh.vertices, axis_remap)
    normals = _face_normals(remapped_vertices, foot_mesh.faces)
    plantar_face_indices = np.flatnonzero(
        normals[:, 1] >= PLANTAR_NORMAL_Y_MIN
    )
    if len(plantar_face_indices) == 0:
        raise ValueError("SUPR foot has no plantar faces after axis remapping")
    plantar_vertex_indices = np.unique(foot_mesh.faces[plantar_face_indices])
    plantar_vertices = remapped_vertices[plantar_vertex_indices]
    plantar_bounds = np.stack(
        (plantar_vertices.min(axis=0), plantar_vertices.max(axis=0)), axis=0
    )
    plantar_length = float(plantar_bounds[1, 0] - plantar_bounds[0, 0])
    if not np.isfinite(plantar_length) or plantar_length <= 0.0:
        raise ValueError("SUPR plantar footprint must have positive X length")

    heel_x = float(plantar_bounds[0, 0])
    heel_tolerance = max(1.0, abs(heel_x)) * np.finfo(np.float64).eps * 16.0
    heel_local = np.flatnonzero(
        np.abs(plantar_vertices[:, 0] - heel_x) <= heel_tolerance
    )
    heel_vertex_indices = plantar_vertex_indices[heel_local]
    heel_reference_remapped = remapped_vertices[heel_vertex_indices].mean(axis=0)

    scale = ratio / plantar_length
    axis_and_scale = _uniform_scale_matrix(scale) @ axis_remap
    scaled_vertices = transform_points(foot_mesh.vertices, axis_and_scale)
    translation_x = -heel_x * scale
    x_aligned = scaled_vertices.copy()
    x_aligned[:, 0] += translation_x

    plantar_triangles = x_aligned[foot_mesh.faces[plantar_face_indices]]
    centroids = plantar_triangles.mean(axis=1)
    planar = plantar_triangles[:, :, (0, 2)]
    edge_a = planar[:, 1] - planar[:, 0]
    edge_b = planar[:, 2] - planar[:, 0]
    projected_areas = 0.5 * np.abs(
        edge_a[:, 0] * edge_b[:, 1] - edge_a[:, 1] * edge_b[:, 0]
    )
    usable = projected_areas > np.finfo(np.float64).eps
    if not np.any(usable):
        raise ValueError("SUPR plantar faces have zero projected X/Z area")
    centroid_x = centroids[usable, 0]
    tolerance = 1e-12
    if (
        float(np.min(centroid_x)) < centerline[0, 0] - tolerance
        or float(np.max(centroid_x)) > centerline[-1, 0] + tolerance
    ):
        raise ValueError(
            "normalized footbed centerline does not cover the placed plantar faces"
        )
    weights = projected_areas[usable]
    target_z = np.interp(centroid_x, centerline[:, 0], centerline[:, 1])
    residual_before = centroids[usable, 2] - target_z
    translation_z = float(-np.sum(weights * residual_before) / np.sum(weights))
    residual_after = residual_before + translation_z
    lateral_rms_before = float(
        np.sqrt(np.sum(weights * np.square(residual_before)) / np.sum(weights))
    )
    lateral_rms_after = float(
        np.sqrt(np.sum(weights * np.square(residual_after)) / np.sum(weights))
    )

    horizontal_translation = np.asarray(
        [translation_x, 0.0, translation_z], dtype=np.float64
    )
    horizontal_matrix = (
        _translation_matrix(horizontal_translation) @ axis_and_scale
    )
    horizontal_vertices = transform_points(foot_mesh.vertices, horizontal_matrix)
    support_y, valid = sample_triangle_mesh_y(
        normalized_support_mesh,
        horizontal_vertices[plantar_vertex_indices][:, (0, 2)],
    )
    covered_count = int(np.count_nonzero(valid))
    coverage = covered_count / len(plantar_vertex_indices)
    if coverage < MIN_PLANTAR_SUPPORT_COVERAGE:
        raise ValueError(
            "plantar support coverage is below the required threshold: "
            f"{covered_count}/{len(plantar_vertex_indices)} = "
            f"{coverage:.6f} < {MIN_PLANTAR_SUPPORT_COVERAGE:.6f}"
        )

    covered_vertices = plantar_vertex_indices[valid]
    candidate_shifts = support_y[valid] - horizontal_vertices[covered_vertices, 1]
    translation_y = float(np.min(candidate_shifts))
    vertical_matrix = _translation_matrix(
        np.asarray([0.0, translation_y, 0.0], dtype=np.float64)
    )
    foot_to_normalized_shoe = vertical_matrix @ horizontal_matrix
    normalized_shoe_to_foot = np.linalg.inv(foot_to_normalized_shoe)
    foot_to_original_shoe = normalized_to_shoe @ foot_to_normalized_shoe
    original_shoe_to_foot = np.linalg.inv(foot_to_original_shoe)

    aligned_vertices = transform_points(foot_mesh.vertices, foot_to_normalized_shoe)
    aligned_bounds = np.stack(
        (aligned_vertices.min(axis=0), aligned_vertices.max(axis=0)), axis=0
    )
    heel_reference_normalized = transform_points(
        heel_reference_remapped[None, :],
        vertical_matrix
        @ _translation_matrix(horizontal_translation)
        @ _uniform_scale_matrix(scale),
    )[0]
    gaps = support_y[valid] - aligned_vertices[covered_vertices, 1]
    gaps[np.abs(gaps) < 1e-14] = 0.0
    achieved_ratio = float(np.ptp(aligned_vertices[plantar_vertex_indices, 0]))
    translation = np.asarray(
        [translation_x, translation_y, translation_z], dtype=np.float64
    )
    return InitialFootPlacement(
        foot_to_normalized_shoe=foot_to_normalized_shoe,
        normalized_shoe_to_foot=normalized_shoe_to_foot,
        shoe_to_normalized=shoe_to_normalized,
        normalized_to_shoe=normalized_to_shoe,
        foot_to_original_shoe=foot_to_original_shoe,
        original_shoe_to_foot=original_shoe_to_foot,
        axis_remap=axis_remap,
        scale=scale,
        translation=translation,
        requested_plantar_length_ratio=ratio,
        achieved_plantar_length_ratio=achieved_ratio,
        plantar_face_indices=plantar_face_indices,
        plantar_vertex_indices=plantar_vertex_indices,
        heel_vertex_indices=heel_vertex_indices,
        heel_reference_remapped=heel_reference_remapped,
        heel_reference_normalized=heel_reference_normalized,
        remapped_plantar_bounds=plantar_bounds,
        input_foot_bounds=foot_mesh.bounds,
        input_foot_extents=foot_mesh.extents,
        normalized_shoe_bounds=normalized_shoe_mesh.bounds,
        normalized_shoe_extents=normalized_shoe_mesh.extents,
        normalized_support_bounds=normalized_support_mesh.bounds,
        normalized_support_extents=normalized_support_mesh.extents,
        aligned_foot_bounds=aligned_bounds,
        aligned_foot_extents=aligned_bounds[1] - aligned_bounds[0],
        lateral_face_count=int(np.count_nonzero(usable)),
        lateral_projected_area=float(np.sum(weights)),
        lateral_rms_before=lateral_rms_before,
        lateral_rms_after=lateral_rms_after,
        plantar_sample_count=int(len(plantar_vertex_indices)),
        covered_plantar_sample_count=covered_count,
        plantar_support_coverage=coverage,
        minimum_support_gap=float(np.min(gaps)),
        median_support_gap=float(np.median(gaps)),
        maximum_support_gap=float(np.max(gaps)),
    )


def identify_supr_contact_regions(
    neutral_foot_mesh: TriangleMesh,
) -> SuprContactRegions:
    """Identify stable plantar indices and heel-to-toe contact regions."""

    remapped = transform_points(
        neutral_foot_mesh.vertices, make_supr_to_shoe_axis_remap()
    )
    normals = _face_normals(remapped, neutral_foot_mesh.faces)
    plantar_faces = np.flatnonzero(normals[:, 1] >= PLANTAR_NORMAL_Y_MIN)
    if len(plantar_faces) == 0:
        raise ValueError("neutral SUPR foot has no plantar faces")
    plantar_vertices = np.unique(neutral_foot_mesh.faces[plantar_faces])
    length_min = float(np.min(remapped[:, 0]))
    length_span = float(np.ptp(remapped[:, 0]))
    if not np.isfinite(length_span) or length_span <= 0.0:
        raise ValueError("neutral SUPR foot must have positive length")
    vertex_u = (remapped[:, 0] - length_min) / length_span
    face_u = vertex_u[neutral_foot_mesh.faces].mean(axis=1)

    def partition(values: np.ndarray, source_indices: np.ndarray) -> dict[str, np.ndarray]:
        masks = {
            "heel": values <= 0.18,
            "arch": (values > 0.18) & (values < 0.55),
            "forefoot": (values >= 0.55) & (values <= 0.80),
            "toes": values > 0.80,
        }
        result = {
            name: source_indices[mask]
            for name, mask in masks.items()
        }
        if sum(len(indices) for indices in result.values()) != len(source_indices):
            raise RuntimeError("SUPR contact regions do not form a complete partition")
        if any(len(indices) == 0 for indices in result.values()):
            raise ValueError("every SUPR contact region must contain samples")
        return result

    return SuprContactRegions(
        plantar_vertex_indices=plantar_vertices,
        plantar_face_indices=plantar_faces,
        vertex_regions=partition(vertex_u[plantar_vertices], plantar_vertices),
        face_regions=partition(face_u[plantar_faces], plantar_faces),
    )


def _triangle_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    triangles = vertices[faces]
    crosses = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    return crosses, 0.5 * np.linalg.norm(crosses, axis=1)


def _candidate_distortion(
    posed_vertices: np.ndarray,
    faces: np.ndarray,
    neutral_crosses: np.ndarray,
    neutral_areas: np.ndarray,
) -> tuple[dict[str, float | int] | None, str | None]:
    posed_crosses, posed_areas = _triangle_geometry(posed_vertices, faces)
    if not np.isfinite(posed_vertices).all() or not np.isfinite(posed_areas).all():
        return None, "non_finite_geometry"
    area_floor = np.finfo(np.float64).eps
    if np.any(posed_areas <= area_floor):
        return None, "degenerate_triangle"
    neutral_lengths = np.linalg.norm(neutral_crosses, axis=1)
    posed_lengths = np.linalg.norm(posed_crosses, axis=1)
    directions = np.sum(neutral_crosses * posed_crosses, axis=1) / (
        neutral_lengths * posed_lengths
    )
    reversed_count = int(np.count_nonzero(directions <= 0.0))
    if reversed_count:
        return None, "reversed_triangle"
    ratios = np.maximum(
        posed_areas / neutral_areas,
        neutral_areas / posed_areas,
    )
    percentile = float(np.percentile(ratios, 99.0))
    if percentile > MAX_TRIANGLE_AREA_DISTORTION:
        return None, "triangle_area_distortion"
    return {
        "symmetric_area_ratio_p99": percentile,
        "symmetric_area_ratio_maximum": float(np.max(ratios)),
        "faces_above_1_5": int(
            np.count_nonzero(ratios > MAX_TRIANGLE_AREA_DISTORTION)
        ),
        "reversed_face_count": reversed_count,
        "degenerate_face_count": 0,
    }, None


def build_support_placement_context(
    neutral_foot_mesh: TriangleMesh,
    normalized_shoe_mesh: TriangleMesh,
    normalized_support_mesh: TriangleMesh,
    normalized_centerline_xz: np.ndarray,
) -> SupportPlacementContext:
    """Prepare immutable geometry shared by support-placement candidates."""

    centerline = _validated_centerline(normalized_centerline_xz)
    neutral_crosses, neutral_areas = _triangle_geometry(
        neutral_foot_mesh.vertices, neutral_foot_mesh.faces
    )
    if np.any(neutral_areas <= np.finfo(np.float64).eps):
        raise ValueError("neutral SUPR template contains a degenerate triangle")
    return SupportPlacementContext(
        faces=neutral_foot_mesh.faces.copy(),
        regions=identify_supr_contact_regions(neutral_foot_mesh),
        neutral_crosses=neutral_crosses,
        neutral_areas=neutral_areas,
        normalized_shoe_bounds=normalized_shoe_mesh.bounds.copy(),
        normalized_support_mesh=normalized_support_mesh,
        centerline=centerline,
        length_scale=neutral_length_scale(neutral_foot_mesh),
    )


def _weighted_rms(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weights * np.square(values)) / np.sum(weights)))


def _coverage_and_gap_record(
    mask: np.ndarray,
    valid: np.ndarray,
    weights: np.ndarray,
    gaps: np.ndarray,
) -> dict[str, Any]:
    total_area = float(np.sum(weights[mask]))
    covered = mask & valid
    covered_area = float(np.sum(weights[covered]))
    coverage = covered_area / total_area
    covered_gaps = gaps[covered]
    record: dict[str, Any] = {
        "sample_count": int(np.count_nonzero(mask)),
        "covered_sample_count": int(np.count_nonzero(covered)),
        "projected_area": total_area,
        "covered_projected_area": covered_area,
        "projected_area_coverage": coverage,
    }
    if len(covered_gaps) == 0:
        record.update(
            rms_gap=None,
            minimum_gap=None,
            median_gap=None,
            maximum_gap=None,
        )
    else:
        record.update(
            rms_gap=_weighted_rms(covered_gaps, weights[covered]),
            minimum_gap=float(np.min(covered_gaps)),
            median_gap=float(np.median(covered_gaps)),
            maximum_gap=float(np.max(covered_gaps)),
        )
    return record


def evaluate_support_placement(
    context: SupportPlacementContext,
    ankle_degrees: float,
    midfoot_degrees: float,
    pose: np.ndarray,
    betas: np.ndarray,
    posed_vertices: np.ndarray,
    posed_joints: np.ndarray,
    reference_crosses: np.ndarray,
    reference_areas: np.ndarray,
    heel_offset_x: float = 0.0,
    lateral_offset_z: float = 0.0,
) -> tuple[SupportPlacementCandidate | None, str | None]:
    """Place one SUPR pose/shape on support, or return a rejection reason.

    The SUPR-to-shoe scale is the context's anchored constant, so this foot's
    own length carries through to the shoe. ``reference_crosses`` and
    ``reference_areas`` describe the same betas at zero pose, which keeps the
    distortion gate measuring articulation rather than anatomy.
    """

    heel_offset = float(heel_offset_x)
    lateral_offset = float(lateral_offset_z)
    shape = np.asarray(betas, dtype=np.float64)
    if not np.isfinite(heel_offset) or not np.isfinite(lateral_offset):
        return None, "non_finite_placement_offset"
    if shape.ndim != 1 or not np.isfinite(shape).all():
        return None, "invalid_betas"

    faces = context.faces
    regions = context.regions
    distortion, rejection = _candidate_distortion(
        posed_vertices, faces, reference_crosses, reference_areas
    )
    if rejection is not None:
        return None, rejection
    assert distortion is not None

    remapped = transform_points(posed_vertices, make_supr_to_shoe_axis_remap())
    posed_length = float(np.ptp(remapped[:, 0]))
    if not np.isfinite(posed_length) or posed_length <= 0.0:
        return None, "invalid_foot_length"

    # Length is now an output: the anchor fixes the scale, and the betas decide
    # how much of the shoe the foot occupies.
    scale = context.length_scale
    foot_length_ratio = scale * posed_length
    target_toe_x = heel_offset + foot_length_ratio
    allowance = foot_length_ratio_to_toe_allowance(foot_length_ratio, heel_offset)
    if allowance < MIN_TOE_ALLOWANCE_MM - 1e-9:
        return None, "insufficient_toe_allowance"
    if foot_length_ratio < MIN_FOOT_LENGTH_RATIO - 1e-12:
        return None, "foot_shorter_than_diagnostic_limit"
    if (
        context.normalized_shoe_bounds[0, 0] > heel_offset + 1e-12
        or context.normalized_shoe_bounds[1, 0] < target_toe_x - 1e-12
    ):
        return None, "foot_interval_outside_shoe"

    axis_and_scale = _uniform_scale_matrix(scale) @ make_supr_to_shoe_axis_remap()
    scaled = transform_points(posed_vertices, axis_and_scale)
    translation_x = heel_offset - float(np.min(scaled[:, 0]))
    x_aligned = scaled.copy()
    x_aligned[:, 0] += translation_x

    plantar_faces = regions.plantar_face_indices
    plantar_triangles = x_aligned[faces[plantar_faces]]
    centroids = plantar_triangles.mean(axis=1)
    planar = plantar_triangles[:, :, (0, 2)]
    first_edge = planar[:, 1] - planar[:, 0]
    second_edge = planar[:, 2] - planar[:, 0]
    weights = 0.5 * np.abs(
        first_edge[:, 0] * second_edge[:, 1]
        - first_edge[:, 1] * second_edge[:, 0]
    )
    if np.any(weights <= np.finfo(np.float64).eps):
        return None, "degenerate_projected_plantar_face"
    centroid_x = centroids[:, 0]
    if (
        float(np.min(centroid_x)) < context.centerline[0, 0] - 1e-12
        or float(np.max(centroid_x)) > context.centerline[-1, 0] + 1e-12
    ):
        return None, "centerline_out_of_range"
    centerline_z = np.interp(
        centroid_x, context.centerline[:, 0], context.centerline[:, 1]
    )
    residual_before = centroids[:, 2] - centerline_z
    centerline_translation_z = float(
        -np.sum(weights * residual_before) / np.sum(weights)
    )
    translation_z = centerline_translation_z + lateral_offset
    residual_after = residual_before + translation_z
    lateral_fit: dict[str, float | int] = {
        "face_count": int(len(plantar_faces)),
        "projected_area": float(np.sum(weights)),
        "centerline_translation_z": centerline_translation_z,
        "lateral_offset_z": lateral_offset,
        "translation_z": translation_z,
        "rms_before_translation": _weighted_rms(residual_before, weights),
        "rms_after_translation": _weighted_rms(residual_after, weights),
    }

    horizontal_translation = np.asarray(
        [translation_x, 0.0, translation_z], dtype=np.float64
    )
    horizontal_transform = (
        _translation_matrix(horizontal_translation) @ axis_and_scale
    )
    horizontal_vertices = transform_points(posed_vertices, horizontal_transform)
    horizontal_centroids = horizontal_vertices[faces[plantar_faces]].mean(axis=1)
    plantar_vertices = regions.plantar_vertex_indices
    vertex_y, vertex_valid = sample_triangle_mesh_y(
        context.normalized_support_mesh,
        horizontal_vertices[plantar_vertices][:, (0, 2)],
    )
    centroid_y, centroid_valid = sample_triangle_mesh_y(
        context.normalized_support_mesh,
        horizontal_centroids[:, (0, 2)],
    )

    face_region_masks = {
        name: np.isin(plantar_faces, indices)
        for name, indices in regions.face_regions.items()
    }
    preliminary_coverage = {
        name: float(
            np.sum(weights[mask & centroid_valid]) / np.sum(weights[mask])
        )
        for name, mask in face_region_masks.items()
    }
    overall_coverage = float(
        np.sum(weights[centroid_valid]) / np.sum(weights)
    )
    if overall_coverage < MIN_PLANTAR_SUPPORT_COVERAGE:
        return None, "insufficient_overall_support"
    if preliminary_coverage["heel"] < MIN_HEEL_SUPPORT_COVERAGE:
        return None, "insufficient_heel_support"
    if preliminary_coverage["forefoot"] < MIN_FOREFOOT_SUPPORT_COVERAGE:
        return None, "insufficient_forefoot_support"
    if preliminary_coverage["toes"] < MIN_TOE_SUPPORT_COVERAGE:
        return None, "insufficient_toe_support"

    shifts: list[np.ndarray] = []
    if np.any(vertex_valid):
        shifts.append(
            vertex_y[vertex_valid]
            - horizontal_vertices[plantar_vertices[vertex_valid], 1]
        )
    if np.any(centroid_valid):
        shifts.append(
            centroid_y[centroid_valid]
            - horizontal_centroids[centroid_valid, 1]
        )
    if not shifts:
        return None, "no_supported_plantar_samples"
    translation_y = float(np.min(np.concatenate(shifts)))
    translation = np.asarray(
        [translation_x, translation_y, translation_z], dtype=np.float64
    )
    transform = _translation_matrix(translation) @ axis_and_scale
    aligned_vertices = transform_points(posed_vertices, transform)
    aligned_centroids = aligned_vertices[faces[plantar_faces]].mean(axis=1)
    vertex_gaps = np.full(len(plantar_vertices), np.nan, dtype=np.float64)
    vertex_gaps[vertex_valid] = (
        vertex_y[vertex_valid]
        - aligned_vertices[plantar_vertices[vertex_valid], 1]
    )
    centroid_gaps = np.full(len(plantar_faces), np.nan, dtype=np.float64)
    centroid_gaps[centroid_valid] = (
        centroid_y[centroid_valid] - aligned_centroids[centroid_valid, 1]
    )
    numerical_tolerance = 1e-10
    if (
        np.any(vertex_gaps[vertex_valid] < -numerical_tolerance)
        or np.any(centroid_gaps[centroid_valid] < -numerical_tolerance)
    ):
        return None, "sampled_plantar_penetration"
    vertex_gaps[np.abs(vertex_gaps) < numerical_tolerance] = 0.0
    centroid_gaps[np.abs(centroid_gaps) < numerical_tolerance] = 0.0

    overall_mask = np.ones(len(plantar_faces), dtype=bool)
    region_contact = {
        "overall": _coverage_and_gap_record(
            overall_mask, centroid_valid, weights, centroid_gaps
        )
    }
    for name, mask in face_region_masks.items():
        region_contact[name] = _coverage_and_gap_record(
            mask, centroid_valid, weights, centroid_gaps
        )

    vertex_region_records: dict[str, Any] = {}
    for name, indices in regions.vertex_regions.items():
        mask = np.isin(plantar_vertices, indices)
        covered = mask & vertex_valid
        vertex_region_records[name] = {
            "sample_count": int(np.count_nonzero(mask)),
            "covered_sample_count": int(np.count_nonzero(covered)),
            "coverage": float(np.count_nonzero(covered) / np.count_nonzero(mask)),
        }
    valid_vertex_gaps = vertex_gaps[vertex_valid]
    plantar_vertex_contact: dict[str, Any] = {
        "sample_count": int(len(plantar_vertices)),
        "covered_sample_count": int(np.count_nonzero(vertex_valid)),
        "coverage": float(np.mean(vertex_valid)),
        "minimum_gap": float(np.min(valid_vertex_gaps)),
        "median_gap": float(np.median(valid_vertex_gaps)),
        "maximum_gap": float(np.max(valid_vertex_gaps)),
        "regions": vertex_region_records,
    }
    heel_rms = float(region_contact["heel"]["rms_gap"])
    forefoot_rms = float(region_contact["forefoot"]["rms_gap"])
    return SupportPlacementCandidate(
        ankle_degrees=float(ankle_degrees),
        midfoot_degrees=float(midfoot_degrees),
        pose=pose.copy(),
        betas=shape.copy(),
        posed_vertices=posed_vertices.copy(),
        posed_joints=posed_joints.copy(),
        aligned_vertices=aligned_vertices,
        transform=transform,
        scale=scale,
        translation=translation,
        region_contact=region_contact,
        plantar_vertex_contact=plantar_vertex_contact,
        lateral_fit=lateral_fit,
        distortion=distortion,
        primary_score=max(heel_rms, forefoot_rms),
        heel_forefoot_sum=heel_rms + forefoot_rms,
        toe_allowance_mm=allowance,
        foot_length_ratio=foot_length_ratio,
        toe_x=target_toe_x,
        heel_offset_x=heel_offset,
        lateral_offset_z=lateral_offset,
    ), None


def _angle_pairs(
    ankle_values: np.ndarray, midfoot_values: np.ndarray
) -> list[tuple[float, float]]:
    return [
        (float(ankle), float(midfoot))
        for ankle in ankle_values
        for midfoot in midfoot_values
    ]


def _evaluate_angle_pairs(
    pairs: list[tuple[float, float]],
    supr_model: SuprFootModel,
    context: SupportPlacementContext,
) -> tuple[list[SupportPlacementCandidate], dict[str, int]]:
    """Evaluate pose candidates at zero betas against the neutral reference."""

    poses = np.zeros(
        (len(pairs), supr_model.num_pose_parameters), dtype=np.float32
    )
    for index, (ankle, midfoot) in enumerate(pairs):
        poses[index, SUPR_ANKLE_PITCH_INDEX] = np.deg2rad(ankle)
        poses[index, SUPR_MIDFOOT_PITCH_INDEX] = np.deg2rad(midfoot)
    betas = np.zeros((len(pairs), supr_model.num_betas), dtype=np.float32)
    posed_batch, joint_batch = supr_model.evaluate(poses, betas)
    candidates: list[SupportPlacementCandidate] = []
    rejections: dict[str, int] = {}
    for index, (ankle, midfoot) in enumerate(pairs):
        candidate, rejection = evaluate_support_placement(
            context,
            ankle,
            midfoot,
            poses[index],
            betas[index],
            posed_batch[index],
            joint_batch[index],
            context.neutral_crosses,
            context.neutral_areas,
        )
        if candidate is None:
            assert rejection is not None
            rejections[rejection] = rejections.get(rejection, 0) + 1
        else:
            candidates.append(candidate)
    return candidates, dict(sorted(rejections.items()))


def build_support_foot_fit(
    supr_model: SuprFootModel,
    neutral_foot_mesh: TriangleMesh,
    normalized_shoe_mesh: TriangleMesh,
    normalized_support_mesh: TriangleMesh,
    normalized_centerline_xz: np.ndarray,
    shoe_to_normalized: np.ndarray,
    normalized_to_shoe: np.ndarray,
    support_grid_cell_spacing: float,
) -> SupportFootFit:
    """Fit ankle and midfoot pitch for balanced heel/forefoot support contact.

    Sizing is not searched here. The anchor fixes the scale, so each pose keeps
    whatever heel-to-toe length it actually has and the resulting front
    allowance is reported rather than imposed.
    """

    cell_spacing = float(support_grid_cell_spacing)
    if not np.isfinite(cell_spacing) or cell_spacing <= 0.0:
        raise ValueError("support_grid_cell_spacing must be finite and positive")
    shoe_to_normalized, normalized_to_shoe = _validated_inverse_pair(
        shoe_to_normalized, normalized_to_shoe
    )
    if not np.array_equal(supr_model.faces, neutral_foot_mesh.faces):
        raise ValueError("posable and neutral SUPR models must use identical faces")
    if (
        normalized_shoe_mesh.bounds[0, 0] > 1e-12
        or normalized_shoe_mesh.bounds[1, 0] < ANCHOR_FOOT_LENGTH_RATIO - 1e-12
    ):
        raise ValueError(
            "normalized shoe outer X bounds do not contain the anchored foot interval"
        )

    context = build_support_placement_context(
        neutral_foot_mesh,
        normalized_shoe_mesh,
        normalized_support_mesh,
        normalized_centerline_xz,
    )
    regions = context.regions

    coarse_values = np.arange(-20.0, 20.0 + 1e-9, 2.0)
    coarse_pairs = _angle_pairs(coarse_values, coarse_values)
    coarse, coarse_rejections = _evaluate_angle_pairs(
        coarse_pairs,
        supr_model,
        context,
    )
    if not coarse:
        raise ValueError(
            "no valid coarse SUPR support-fit candidate; rejection counts: "
            f"{coarse_rejections}"
        )
    best_coarse = min(
        coarse,
        key=lambda candidate: (
            candidate.primary_score,
            candidate.ankle_degrees**2 + candidate.midfoot_degrees**2,
            candidate.heel_forefoot_sum,
            candidate.ankle_degrees,
            candidate.midfoot_degrees,
        ),
    )
    fine_ankles = np.arange(
        best_coarse.ankle_degrees - 2.0,
        best_coarse.ankle_degrees + 2.0 + 1e-9,
        0.25,
    )
    fine_midfeet = np.arange(
        best_coarse.midfoot_degrees - 2.0,
        best_coarse.midfoot_degrees + 2.0 + 1e-9,
        0.25,
    )
    fine_ankles = np.unique(np.clip(fine_ankles, -20.0, 20.0))
    fine_midfeet = np.unique(np.clip(fine_midfeet, -20.0, 20.0))
    fine_pairs = _angle_pairs(fine_ankles, fine_midfeet)
    fine, fine_rejections = _evaluate_angle_pairs(
        fine_pairs,
        supr_model,
        context,
    )
    if not fine:
        raise ValueError(
            "no valid fine SUPR support-fit candidate; rejection counts: "
            f"{fine_rejections}"
        )

    unique: dict[tuple[float, float], SupportPlacementCandidate] = {}
    for candidate in (*coarse, *fine):
        key = (round(candidate.ankle_degrees, 8), round(candidate.midfoot_degrees, 8))
        unique[key] = candidate
    combined = list(unique.values())
    minimum_score = min(candidate.primary_score for candidate in combined)
    equivalent = [
        candidate
        for candidate in combined
        if candidate.primary_score <= minimum_score + cell_spacing
    ]
    selected = min(
        equivalent,
        key=lambda candidate: (
            candidate.ankle_degrees**2 + candidate.midfoot_degrees**2,
            candidate.primary_score,
            candidate.heel_forefoot_sum,
            candidate.ankle_degrees,
            candidate.midfoot_degrees,
        ),
    )
    neutral = unique.get((0.0, 0.0))
    neutral_comparison: dict[str, Any]
    if neutral is None:
        neutral_comparison = {"valid": False}
    else:
        neutral_comparison = {"valid": True, **neutral.summary()}

    posed_to_normalized = selected.transform
    normalized_to_posed = np.linalg.inv(posed_to_normalized)
    posed_to_original = normalized_to_shoe @ posed_to_normalized
    original_to_posed = np.linalg.inv(posed_to_original)
    aligned_joints = transform_points(selected.posed_joints, posed_to_normalized)
    search = {
        "coarse": {
            "ankle_range_degrees": [-20.0, 20.0],
            "midfoot_range_degrees": [-20.0, 20.0],
            "step_degrees": 2.0,
            "candidate_count": len(coarse_pairs),
            "valid_candidate_count": len(coarse),
            "rejection_counts": coarse_rejections,
            "best_candidate": best_coarse.summary(),
        },
        "fine": {
            "center_degrees": [
                best_coarse.ankle_degrees,
                best_coarse.midfoot_degrees,
            ],
            "radius_degrees": 2.0,
            "step_degrees": 0.25,
            "candidate_count": len(fine_pairs),
            "valid_candidate_count": len(fine),
            "rejection_counts": fine_rejections,
        },
        "combined_unique_valid_candidate_count": len(combined),
        "minimum_primary_contact_score": minimum_score,
        "geometric_equivalence_tolerance": cell_spacing,
        "equivalent_candidate_count": len(equivalent),
        "selection_order": [
            "smallest ankle_squared_plus_midfoot_squared",
            "smallest primary_contact_score",
            "smallest heel_plus_forefoot_rms",
            "lexicographic ankle_pitch_then_midfoot_pitch",
        ],
        "selected_candidate": selected.summary(),
    }
    return SupportFootFit(
        pose_parameters=selected.pose,
        betas=np.zeros(supr_model.num_betas, dtype=np.float64),
        ankle_pitch_degrees=selected.ankle_degrees,
        midfoot_pitch_degrees=selected.midfoot_degrees,
        posed_vertices=selected.posed_vertices,
        posed_joints=selected.posed_joints,
        aligned_vertices=selected.aligned_vertices,
        aligned_joints=aligned_joints,
        regions=regions,
        posed_supr_to_normalized_shoe=posed_to_normalized,
        normalized_shoe_to_posed_supr=normalized_to_posed,
        posed_supr_to_original_shoe=posed_to_original,
        original_shoe_to_posed_supr=original_to_posed,
        shoe_to_normalized=shoe_to_normalized,
        normalized_to_shoe=normalized_to_shoe,
        axis_remap=make_supr_to_shoe_axis_remap(),
        scale=selected.scale,
        translation=selected.translation,
        reference_foot_length_mm=REFERENCE_FOOT_LENGTH_MM,
        toe_allowance_mm=selected.toe_allowance_mm,
        foot_length_ratio=selected.foot_length_ratio,
        support_grid_cell_spacing=cell_spacing,
        search=search,
        neutral_comparison=neutral_comparison,
        region_contact=selected.region_contact,
        plantar_vertex_contact=selected.plantar_vertex_contact,
        lateral_fit=selected.lateral_fit,
        distortion=selected.distortion,
        normalized_shoe_bounds=normalized_shoe_mesh.bounds,
        normalized_support_bounds=normalized_support_mesh.bounds,
        aligned_foot_bounds=np.stack(
            (
                selected.aligned_vertices.min(axis=0),
                selected.aligned_vertices.max(axis=0),
            ),
            axis=0,
        ),
    )
