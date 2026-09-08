"""Signed-cavity SUPR containment fitting for normalized normal shoes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .alignment import (
    ANCHOR_FOOT_LENGTH_RATIO,
    ANCHOR_TOE_ALLOWANCE_MM,
    REFERENCE_FOOT_LENGTH_MM,
    SHOE_FUNCTIONAL_LENGTH_MM,
    SupportPlacementCandidate,
    SupportPlacementContext,
    _triangle_geometry,
    build_support_placement_context,
    evaluate_support_placement,
    identify_supr_contact_regions,
    toe_allowance_to_foot_length_ratio,
    transform_points,
)
from .cavity import CavityAnalysis, CavityEvaluator, SignedCavityClearance
from .mesh import TriangleMesh
from .supr_foot import (
    SUPR_ANKLE_PITCH_INDEX,
    SUPR_MIDFOOT_PITCH_INDEX,
    SuprMeshSubdivision,
    SuprFootModel,
)


MAX_BETA_ABS = 3.0
MAX_PITCH_CHANGE_DEGREES = 4.0
MAX_JOINT_ITERATIONS = 8
MAX_RESTARTS = 10

# SUPR shape, rather than a second uniform scale, controls the fitted size.
# The accepted interval is deliberately wider than a single target because the
# learned beta directions change length together with width and instep height.
TARGET_TOE_ALLOWANCE_MM = 20.0
MIN_TARGET_TOE_ALLOWANCE_MM = 18.0
MAX_TARGET_TOE_ALLOWANCE_MM = 22.0
BETA_AXIS_MAGNITUDES = (1.0, 2.0)
BETA_COUPLED_MAGNITUDES = (0.75, 1.5)
BETA0_SOLVE_ITERATIONS = 10
LATIN_HYPERCUBE_SAMPLE_COUNT = 256
LATIN_HYPERCUBE_SEED = 0
BETA0_TARGET_ALLOWANCES_MM = (18.0, 20.0, 22.0)
JACOBIAN_BETA_STEP = 0.05
JACOBIAN_PITCH_STEP_DEGREES = 0.1
JACOBIAN_OFFSET_STEP_CELLS = 0.1
LINE_SEARCH_FACTORS = (2.0, 1.0, 0.5, 0.25)
INITIAL_BETA_POLL_STEP = 0.5
INITIAL_PITCH_POLL_DEGREES = 0.5
INITIAL_OFFSET_POLL_CELLS = 1.0
MIN_BETA_POLL_STEP = 0.125
MIN_PITCH_POLL_DEGREES = 0.125
MIN_OFFSET_POLL_CELLS = 0.25

TARGET_MAX_RATIO = toe_allowance_to_foot_length_ratio(
    MIN_TARGET_TOE_ALLOWANCE_MM
)
TARGET_MIN_RATIO = toe_allowance_to_foot_length_ratio(
    MAX_TARGET_TOE_ALLOWANCE_MM
)
TARGET_TOE_X = toe_allowance_to_foot_length_ratio(TARGET_TOE_ALLOWANCE_MM)


def _minimum_norm_joint_update(
    jacobian: np.ndarray, desired_clearance_change: np.ndarray
) -> np.ndarray:
    """Solve one simultaneous least-squares correction for active controls."""

    matrix = np.asarray(jacobian, dtype=np.float64)
    target = np.asarray(desired_clearance_change, dtype=np.float64)
    if matrix.ndim != 2 or target.shape != (matrix.shape[0],):
        raise ValueError("joint update inputs have incompatible shapes")
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("joint update inputs must be finite")
    return np.linalg.lstsq(matrix, target, rcond=None)[0]


def collision_area_fraction(
    fitted_foot: TriangleMesh, collision_pairs: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """Measure collision on unique SUPR faces, independent of shoe density."""

    pairs = np.asarray(collision_pairs, dtype=np.int64)
    if pairs.ndim != 2 or pairs.shape[1:] != (2,):
        raise ValueError("collision_pairs must have shape (N, 2)")
    triangles = fitted_foot.vertices[fitted_foot.faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    total = float(np.sum(areas))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("fitted foot must have positive finite surface area")
    colliding = (
        np.unique(pairs[:, 0])
        if len(pairs)
        else np.empty(0, dtype=np.int64)
    )
    if len(colliding) and (
        np.any(colliding < 0) or np.any(colliding >= len(fitted_foot.faces))
    ):
        raise ValueError("collision pairs contain invalid foot-face indices")
    area = float(np.sum(areas[colliding]))
    return area, area / total, colliding


def affected_area_fraction(
    fitted_foot: TriangleMesh,
    colliding_faces: np.ndarray,
    outside_faces: np.ndarray,
) -> tuple[float, float, np.ndarray]:
    """Measure the union of exact-collision and signed-outside foot faces."""

    affected = np.union1d(
        np.asarray(colliding_faces, dtype=np.int64),
        np.asarray(outside_faces, dtype=np.int64),
    )
    triangles = fitted_foot.vertices[fitted_foot.faces]
    areas = 0.5 * np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    )
    total = float(np.sum(areas))
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("fitted foot must have positive finite surface area")
    if len(affected) and (
        np.any(affected < 0) or np.any(affected >= len(fitted_foot.faces))
    ):
        raise ValueError("affected foot-face indices are invalid")
    area = float(np.sum(areas[affected]))
    return area, area / total, affected


@dataclass(frozen=True)
class _Parameters:
    """One searched control vector.

    Foot length is deliberately absent: it is an output of the betas under the
    anchored scale, so it is read back off the resulting placement instead of
    being dialled in here.
    """

    heel_offset_x: float
    lateral_offset_z: float
    ankle_degrees: float
    midfoot_degrees: float
    betas: np.ndarray

    def cache_key(self) -> tuple[float, ...]:
        values = (
            self.heel_offset_x,
            self.lateral_offset_z,
            self.ankle_degrees,
            self.midfoot_degrees,
            *self.betas.tolist(),
        )
        return tuple(round(float(value), 8) for value in values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "heel_offset_x": self.heel_offset_x,
            "lateral_offset_z": self.lateral_offset_z,
            "ankle_pitch_degrees": self.ankle_degrees,
            "midfoot_pitch_degrees": self.midfoot_degrees,
            "betas": self.betas.tolist(),
        }


@dataclass(frozen=True)
class _ScoredCandidate:
    parameters: _Parameters
    placement: SupportPlacementCandidate
    signed: SignedCavityClearance
    outside_area_tier: int
    beta_l2_norm: float
    placement_deviation: float
    collision_pairs: np.ndarray | None = None
    colliding_foot_faces: np.ndarray | None = None
    collision_area: float | None = None
    collision_area_fraction: float | None = None
    collision_area_tier: int | None = None
    affected_foot_faces: np.ndarray | None = None
    affected_area: float | None = None
    affected_area_fraction: float | None = None
    affected_area_tier: int | None = None

    @property
    def exact_evaluated(self) -> bool:
        return self.collision_pairs is not None

    @property
    def target_band_distance(self) -> float:
        allowance = self.placement.toe_allowance_mm
        if allowance < MIN_TARGET_TOE_ALLOWANCE_MM:
            return MIN_TARGET_TOE_ALLOWANCE_MM - allowance
        if allowance > MAX_TARGET_TOE_ALLOWANCE_MM:
            return allowance - MAX_TARGET_TOE_ALLOWANCE_MM
        return 0.0

    @property
    def in_target_band(self) -> bool:
        return self.target_band_distance <= 1e-9

    @property
    def contained(self) -> bool:
        return (
            self.exact_evaluated
            and len(self.collision_pairs) == 0
            and len(self.signed.outside_face_indices) == 0
        )

    @property
    def maximum_protrusion(self) -> float:
        value = self.signed.protrusion_statistics["maximum_protrusion_depth"]
        return float(value or 0.0)

    def rank_key(self) -> tuple[Any, ...]:
        """Order exact candidates, requiring realistic size before containment."""

        if not self.exact_evaluated:
            raise RuntimeError("exact collision evaluation is required for ranking")
        assert self.affected_area_tier is not None
        assert self.collision_area_tier is not None
        assert self.collision_area_fraction is not None

        return (
            0 if self.in_target_band else 1,
            self.target_band_distance,
            0 if self.contained else 1,
            self.affected_area_tier,
            self.collision_area_tier,
            self.collision_area_fraction,
            self.outside_area_tier,
            self.maximum_protrusion,
            self.signed.protrusion_energy,
            abs(self.placement.toe_allowance_mm - TARGET_TOE_ALLOWANCE_MM),
            self.beta_l2_norm,
            self.placement_deviation,
            self.placement.primary_score,
            self.placement.heel_forefoot_sum,
            self.parameters.cache_key(),
        )

    def summary(self) -> dict[str, Any]:
        record = {
            **self.parameters.to_dict(),
            "foot_length_ratio": self.placement.foot_length_ratio,
            "toe_allowance_mm": self.placement.toe_allowance_mm,
            "target_toe_allowance_mm": TARGET_TOE_ALLOWANCE_MM,
            "in_target_toe_allowance_band": self.in_target_band,
            "contained_by_fast_evaluator": self.contained,
            "outside_foot_face_count": int(len(self.signed.outside_face_indices)),
            "outside_area": self.signed.outside_area,
            "outside_area_fraction": self.signed.outside_area_fraction,
            "outside_area_tier": self.outside_area_tier,
            "signed_score_exempt_vertex_count": int(
                len(self.signed.signed_exempt_vertex_indices)
            ),
            "signed_score_exempt_face_count": int(
                len(self.signed.signed_exempt_face_indices)
            ),
            "maximum_protrusion_depth": self.maximum_protrusion,
            "protrusion_energy": self.signed.protrusion_energy,
            "beta_l2_norm": self.beta_l2_norm,
            "placement_deviation": self.placement_deviation,
            "primary_contact_score": self.placement.primary_score,
        }
        if self.exact_evaluated:
            assert self.collision_pairs is not None
            assert self.colliding_foot_faces is not None
            assert self.affected_foot_faces is not None
            record.update(
                collision_pair_count=int(len(self.collision_pairs)),
                colliding_foot_face_count=int(len(self.colliding_foot_faces)),
                collision_area=self.collision_area,
                collision_area_fraction=self.collision_area_fraction,
                collision_area_tier=self.collision_area_tier,
                affected_foot_face_count=int(len(self.affected_foot_faces)),
                affected_area=self.affected_area,
                affected_area_fraction=self.affected_area_fraction,
                affected_area_tier=self.affected_area_tier,
            )
        return record


@dataclass(frozen=True)
class ContainmentFootFit:
    """Selected SUPR shape, pose, placement, and final cavity diagnosis."""

    pose_parameters: np.ndarray
    betas: np.ndarray
    foot_faces: np.ndarray
    posed_vertices: np.ndarray
    posed_joints: np.ndarray
    aligned_vertices: np.ndarray
    aligned_joints: np.ndarray
    posed_supr_to_normalized_shoe: np.ndarray
    normalized_shoe_to_posed_supr: np.ndarray
    posed_supr_to_original_shoe: np.ndarray
    original_shoe_to_posed_supr: np.ndarray
    shoe_to_normalized: np.ndarray
    normalized_to_shoe: np.ndarray
    placement: SupportPlacementCandidate
    baseline_cavity: CavityAnalysis
    final_cavity: CavityAnalysis
    baseline_collision_score: dict[str, Any]
    final_collision_score: dict[str, Any]
    search: dict[str, Any]

    @property
    def status(self) -> str:
        if self.final_cavity.status != "clear":
            return "residual_target_fit"
        return "contained_target_fit"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "coordinate_conventions": {
                "normalized_shoe": {
                    "x": "functional_heel_to_toe",
                    "y": "vertical_positive_down_toward_sole",
                    "z": "shoe_width",
                }
            },
            "supr": {
                "pose_parameters_radians": self.pose_parameters.tolist(),
                "betas": self.betas.tolist(),
                "active_pose_indices": {
                    "ankle_pitch": SUPR_ANKLE_PITCH_INDEX,
                    "midfoot_pitch": SUPR_MIDFOOT_PITCH_INDEX,
                },
                "selected_angles_degrees": {
                    "ankle_pitch": self.placement.ankle_degrees,
                    "midfoot_pitch": self.placement.midfoot_degrees,
                },
                "posed_joints": self.posed_joints.tolist(),
                "aligned_joints": self.aligned_joints.tolist(),
                "reproduction_note": (
                    "pose and shape are non-rigid; reproduce them from the "
                    "stored SUPR pose, betas, and stable vertex IDs"
                ),
            },
            "sizing": {
                "reference_foot_length_mm": REFERENCE_FOOT_LENGTH_MM,
                "shoe_functional_length_mm": SHOE_FUNCTIONAL_LENGTH_MM,
                "anchor_toe_allowance_mm": ANCHOR_TOE_ALLOWANCE_MM,
                "anchor_foot_length_ratio": ANCHOR_FOOT_LENGTH_RATIO,
                "foot_length_ratio": self.placement.foot_length_ratio,
                "toe_allowance_mm": self.placement.toe_allowance_mm,
                "heel_offset_x": self.placement.heel_offset_x,
                "toe_x": self.placement.toe_x,
                "target_toe_allowance_mm": TARGET_TOE_ALLOWANCE_MM,
                "accepted_toe_allowance_mm": [
                    MIN_TARGET_TOE_ALLOWANCE_MM,
                    MAX_TARGET_TOE_ALLOWANCE_MM,
                ],
                "length_source": (
                    "SUPR shape parameters under a fixed anchored scale; not "
                    "searched and not renormalized"
                ),
                "aligned_dimensions_normalized": {
                    "length": float(np.ptp(self.aligned_vertices[:, 0])),
                    "height": float(np.ptp(self.aligned_vertices[:, 1])),
                    "width": float(np.ptp(self.aligned_vertices[:, 2])),
                },
                "aligned_dimensions_mm": {
                    "length": float(
                        np.ptp(self.aligned_vertices[:, 0])
                        * SHOE_FUNCTIONAL_LENGTH_MM
                    ),
                    "height": float(
                        np.ptp(self.aligned_vertices[:, 1])
                        * SHOE_FUNCTIONAL_LENGTH_MM
                    ),
                    "width": float(
                        np.ptp(self.aligned_vertices[:, 2])
                        * SHOE_FUNCTIONAL_LENGTH_MM
                    ),
                },
            },
            "placement": {
                "lateral_offset_z": self.placement.lateral_offset_z,
                "scale": self.placement.scale,
                "translation": self.placement.translation.tolist(),
                "support_contact": {
                    "face_centroids_by_region": self.placement.region_contact,
                    "plantar_vertices": self.placement.plantar_vertex_contact,
                },
                "lateral_centerline_fit": self.placement.lateral_fit,
                "mesh_distortion": self.placement.distortion,
            },
            "transforms": {
                "posed_supr_to_normalized_shoe": self.posed_supr_to_normalized_shoe.tolist(),
                "normalized_shoe_to_posed_supr": self.normalized_shoe_to_posed_supr.tolist(),
                "posed_supr_to_original_shoe": self.posed_supr_to_original_shoe.tolist(),
                "original_shoe_to_posed_supr": self.original_shoe_to_posed_supr.tolist(),
                "shoe_to_normalized": self.shoe_to_normalized.tolist(),
                "normalized_to_shoe": self.normalized_to_shoe.tolist(),
            },
            "collision_score": {
                "definition": (
                    "area of unique colliding SUPR faces divided by complete "
                    "fitted SUPR surface area"
                ),
                "baseline": self.baseline_collision_score,
                "final": self.final_collision_score,
            },
            "baseline_cavity_analysis": _cavity_summary(self.baseline_cavity),
            "final_cavity_analysis": self.final_cavity.to_dict(),
            "search": self.search,
            "bounds": {
                "aligned_foot": np.stack(
                    (self.aligned_vertices.min(axis=0), self.aligned_vertices.max(axis=0))
                ).tolist()
            },
        }


def _ankle_signed_exemptions(
    placement: SupportPlacementCandidate,
    foot_faces: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Identify the anatomical exit region omitted from signed scoring only."""

    if len(placement.posed_joints) <= 2:
        raise ValueError("SUPR placement must contain ankle and midfoot joints")
    aligned_joints = transform_points(placement.posed_joints, placement.transform)
    ankle = aligned_joints[1]
    midfoot = aligned_joints[2]
    vertices = placement.aligned_vertices

    # Shoe +Y points downward. A smaller Y is therefore above the ankle.
    vertex_mask = (
        (vertices[:, 1] < ankle[1] - tolerance)
        & (vertices[:, 0] <= midfoot[0] + tolerance)
    )
    exempt_vertices = np.flatnonzero(vertex_mask).astype(np.int64)
    faces = np.asarray(foot_faces, dtype=np.int64)
    exempt_faces = np.flatnonzero(np.all(vertex_mask[faces], axis=1)).astype(
        np.int64
    )
    return exempt_vertices, exempt_faces


class _Search:
    """Screen broad SUPR shapes, then refine a few exact candidates."""

    def __init__(
        self,
        supr_model: SuprFootModel,
        neutral_foot: TriangleMesh,
        context: SupportPlacementContext,
        cavity: CavityEvaluator,
        initial: _Parameters,
        grid_spacing: float,
        cavity_subdivision: SuprMeshSubdivision | None = None,
    ) -> None:
        self.supr_model = supr_model
        self.neutral_foot = neutral_foot
        self.context = context
        self.cavity = cavity
        self.initial = initial
        self.grid_spacing = grid_spacing
        self.cavity_subdivision = cavity_subdivision
        if cavity_subdivision is None:
            self.cavity_faces = context.faces
            self.cavity_regions = context.regions
            cavity_neutral = neutral_foot
        else:
            cavity_neutral = cavity_subdivision.apply_mesh(neutral_foot)
            self.cavity_faces = cavity_subdivision.faces
            self.cavity_regions = identify_supr_contact_regions(cavity_neutral)
        _, cavity_areas = _triangle_geometry(
            cavity_neutral.vertices, self.cavity_faces
        )
        self.area_equivalence_fraction = float(
            np.median(cavity_areas) / np.sum(cavity_areas)
        )
        self.cache: dict[tuple[float, ...], _ScoredCandidate | str] = {}
        self.shape_reference_cache: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}
        self.rejections: dict[str, int] = {}
        self.history: list[dict[str, Any]] = []
        self.exact_evaluation_count = 0

    def cavity_mesh(self, vertices: np.ndarray) -> TriangleMesh:
        """Return the topology used for signed and exact containment scoring."""

        values = np.asarray(vertices, dtype=np.float64)
        if self.cavity_subdivision is not None:
            values = self.cavity_subdivision.apply_vertices(values)
        return TriangleMesh(values, self.cavity_faces)

    def cavity_placement(
        self, placement: SupportPlacementCandidate
    ) -> SupportPlacementCandidate:
        """Expose dense candidate vertices while retaining its rigid transform."""

        if self.cavity_subdivision is None:
            return placement
        return replace(
            placement,
            posed_vertices=self.cavity_subdivision.apply_vertices(
                placement.posed_vertices
            ),
            aligned_vertices=self.cavity_subdivision.apply_vertices(
                placement.aligned_vertices
            ),
        )

    def _shape_references(
        self, shapes: np.ndarray
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return per-candidate triangle references at the same betas, zero pose.

        The distortion gate must measure articulation, not anatomy. Comparing a
        beta-shaped foot against the zero-beta template charges shape change to
        the pose budget and rejects the narrow feet we now depend on.
        """

        keys = [tuple(round(float(value), 8) for value in row) for row in shapes]
        missing = sorted({key for key in keys if key not in self.shape_reference_cache})
        if missing:
            batch = np.asarray(missing, dtype=np.float32)
            rest, _ = self.supr_model.evaluate(
                np.zeros((len(batch), self.supr_model.num_pose_parameters), dtype=np.float32),
                batch,
            )
            for key, vertices in zip(missing, rest):
                self.shape_reference_cache[key] = _triangle_geometry(
                    vertices, self.context.faces
                )
        return [self.shape_reference_cache[key] for key in keys]

    def _parameter_rejection(self, item: _Parameters) -> str | None:
        scalars = np.asarray(
            [
                item.heel_offset_x,
                item.lateral_offset_z,
                item.ankle_degrees,
                item.midfoot_degrees,
            ],
            dtype=np.float64,
        )
        if not np.isfinite(scalars).all() or not np.isfinite(item.betas).all():
            return "non_finite_parameters"
        if item.heel_offset_x < -1e-12:
            return "heel_offset_out_of_range"
        ankle_bounds = (
            max(-20.0, self.initial.ankle_degrees - MAX_PITCH_CHANGE_DEGREES),
            min(20.0, self.initial.ankle_degrees + MAX_PITCH_CHANGE_DEGREES),
        )
        midfoot_bounds = (
            max(-20.0, self.initial.midfoot_degrees - MAX_PITCH_CHANGE_DEGREES),
            min(20.0, self.initial.midfoot_degrees + MAX_PITCH_CHANGE_DEGREES),
        )
        if not ankle_bounds[0] <= item.ankle_degrees <= ankle_bounds[1]:
            return "ankle_pitch_out_of_range"
        if not midfoot_bounds[0] <= item.midfoot_degrees <= midfoot_bounds[1]:
            return "midfoot_pitch_out_of_range"
        if np.any(np.abs(item.betas) > MAX_BETA_ABS + 1e-12):
            return "beta_out_of_range"
        return None

    def _placement_deviation(self, item: _Parameters) -> float:
        return float(
            (item.heel_offset_x / self.grid_spacing) ** 2
            + (item.lateral_offset_z / self.grid_spacing) ** 2
            + ((item.ankle_degrees - self.initial.ankle_degrees) / MAX_PITCH_CHANGE_DEGREES) ** 2
            + ((item.midfoot_degrees - self.initial.midfoot_degrees) / MAX_PITCH_CHANGE_DEGREES) ** 2
        )

    def screen(self, items: list[_Parameters]) -> list[_ScoredCandidate]:
        """Evaluate support and signed cavity clearance without exact SAT."""

        unique: list[_Parameters] = []
        seen: set[tuple[float, ...]] = set()
        for item in items:
            key = item.cache_key()
            if key in self.cache or key in seen:
                continue
            seen.add(key)
            rejection = self._parameter_rejection(item)
            if rejection is not None:
                self.cache[key] = rejection
                self.rejections[rejection] = self.rejections.get(rejection, 0) + 1
            else:
                unique.append(item)

        if unique:
            poses = np.zeros(
                (len(unique), self.supr_model.num_pose_parameters), dtype=np.float32
            )
            shapes = np.stack([item.betas for item in unique]).astype(np.float32)
            for index, item in enumerate(unique):
                poses[index, SUPR_ANKLE_PITCH_INDEX] = np.deg2rad(item.ankle_degrees)
                poses[index, SUPR_MIDFOOT_PITCH_INDEX] = np.deg2rad(item.midfoot_degrees)
            posed_batch, joint_batch = self.supr_model.evaluate(poses, shapes)
            references = self._shape_references(shapes)
            for index, item in enumerate(unique):
                reference_crosses, reference_areas = references[index]
                placement, rejection = evaluate_support_placement(
                    self.context,
                    item.ankle_degrees,
                    item.midfoot_degrees,
                    poses[index],
                    shapes[index],
                    posed_batch[index],
                    joint_batch[index],
                    reference_crosses,
                    reference_areas,
                    item.heel_offset_x,
                    item.lateral_offset_z,
                )
                key = item.cache_key()
                if placement is None:
                    assert rejection is not None
                    self.cache[key] = rejection
                    self.rejections[rejection] = self.rejections.get(rejection, 0) + 1
                    continue
                cavity_placement = self.cavity_placement(placement)
                mesh = self.cavity_mesh(placement.aligned_vertices)
                exempt_vertices, exempt_faces = _ankle_signed_exemptions(
                    cavity_placement,
                    self.cavity_faces,
                    self.cavity.numerical_tolerance,
                )
                signed = self.cavity.signed_clearances(
                    mesh,
                    exempt_vertices,
                    exempt_faces,
                )
                outside_tier = int(
                    np.ceil(signed.outside_area_fraction / self.area_equivalence_fraction - 1e-12)
                ) if len(signed.outside_face_indices) else 0
                self.cache[key] = _ScoredCandidate(
                    parameters=item,
                    placement=placement,
                    signed=signed,
                    outside_area_tier=outside_tier,
                    beta_l2_norm=float(np.linalg.norm(item.betas)),
                    placement_deviation=self._placement_deviation(item),
                )
        result: list[_ScoredCandidate] = []
        for item in items:
            value = self.cache[item.cache_key()]
            if isinstance(value, _ScoredCandidate):
                result.append(value)
        return result

    def _with_exact_collisions(
        self, candidate: _ScoredCandidate
    ) -> _ScoredCandidate | None:
        if candidate.exact_evaluated:
            return candidate
        mesh = self.cavity_mesh(candidate.placement.aligned_vertices)
        pairs, ignored = self.cavity.collision_pairs(mesh)
        if len(ignored):
            key = candidate.parameters.cache_key()
            self.cache[key] = "degenerate_aligned_triangle"
            self.rejections["degenerate_aligned_triangle"] = (
                self.rejections.get("degenerate_aligned_triangle", 0) + 1
            )
            return None
        collision_area, collision_fraction, colliding = collision_area_fraction(
            mesh, pairs
        )
        affected_area, affected_fraction, affected = affected_area_fraction(
            mesh, colliding, candidate.signed.outside_face_indices
        )
        collision_tier = (
            int(
                np.ceil(
                    collision_fraction / self.area_equivalence_fraction - 1e-12
                )
            )
            if len(colliding)
            else 0
        )
        affected_tier = (
            int(
                np.ceil(
                    affected_fraction / self.area_equivalence_fraction - 1e-12
                )
            )
            if len(affected)
            else 0
        )
        result = replace(
            candidate,
            collision_pairs=pairs,
            colliding_foot_faces=colliding,
            collision_area=collision_area,
            collision_area_fraction=collision_fraction,
            collision_area_tier=collision_tier,
            affected_foot_faces=affected,
            affected_area=affected_area,
            affected_area_fraction=affected_fraction,
            affected_area_tier=affected_tier,
        )
        self.cache[candidate.parameters.cache_key()] = result
        self.exact_evaluation_count += 1
        return result

    def evaluate(self, items: list[_Parameters]) -> list[_ScoredCandidate]:
        """Return candidates with exact forbidden-surface intersections."""

        result: list[_ScoredCandidate] = []
        for candidate in self.screen(items):
            exact = self._with_exact_collisions(candidate)
            if exact is not None:
                result.append(exact)
        return result

    @staticmethod
    def _parameter_vector(item: _Parameters) -> np.ndarray:
        return np.concatenate(
            (
                item.betas,
                np.asarray(
                    [item.ankle_degrees, item.midfoot_degrees, item.heel_offset_x, item.lateral_offset_z]
                ),
            )
        )

    @staticmethod
    def _from_vector(source: _Parameters, vector: np.ndarray) -> _Parameters:
        return replace(
            source,
            betas=np.clip(vector[:10], -MAX_BETA_ABS, MAX_BETA_ABS),
            ankle_degrees=float(vector[10]),
            midfoot_degrees=float(vector[11]),
            heel_offset_x=float(vector[12]),
            lateral_offset_z=float(vector[13]),
        )

    def _clip_vector(self, vector: np.ndarray) -> np.ndarray:
        result = vector.copy()
        result[:10] = np.clip(result[:10], -MAX_BETA_ABS, MAX_BETA_ABS)
        result[10] = np.clip(
            result[10],
            max(-20.0, self.initial.ankle_degrees - MAX_PITCH_CHANGE_DEGREES),
            min(20.0, self.initial.ankle_degrees + MAX_PITCH_CHANGE_DEGREES),
        )
        result[11] = np.clip(
            result[11],
            max(-20.0, self.initial.midfoot_degrees - MAX_PITCH_CHANGE_DEGREES),
            min(20.0, self.initial.midfoot_degrees + MAX_PITCH_CHANGE_DEGREES),
        )
        result[12] = max(0.0, result[12])
        return result

    @staticmethod
    def _combined_clearances(candidate: _ScoredCandidate) -> np.ndarray:
        """Return one clearance per vertex sample followed by one per face sample."""

        return np.concatenate(
            (
                candidate.signed.vertex_combined_clearances,
                candidate.signed.face_combined_clearances,
            )
        )

    def _active_constraints(
        self, candidate: _ScoredCandidate
    ) -> tuple[np.ndarray, np.ndarray]:
        combined = self._combined_clearances(candidate)
        active = np.flatnonzero(np.isfinite(combined) & (combined <= self.grid_spacing))
        desired = np.maximum(
            0.0,
            -combined[active] + self.cavity.numerical_tolerance,
        )
        return active, desired

    def _joint_proposal(self, current: _ScoredCandidate) -> _Parameters | None:
        active, desired = self._active_constraints(current)
        base_combined = self._combined_clearances(current)
        base_vector = self._parameter_vector(current.parameters)
        actual_steps = np.asarray(
            [JACOBIAN_BETA_STEP] * 10
            + [JACOBIAN_PITCH_STEP_DEGREES] * 2
            + [JACOBIAN_OFFSET_STEP_CELLS * self.grid_spacing] * 2,
            dtype=np.float64,
        )
        parameter_scales = np.asarray(
            [1.0] * 10 + [1.0, 1.0, self.grid_spacing, self.grid_spacing],
            dtype=np.float64,
        )
        perturbations: list[_Parameters] = []
        signed_steps: list[float] = []
        columns: list[int] = []
        for column, step in enumerate(actual_steps):
            vector = base_vector.copy()
            vector[column] += step
            vector = self._clip_vector(vector)
            actual = vector[column] - base_vector[column]
            if abs(actual) <= 1e-14:
                vector[column] = base_vector[column] - step
                vector = self._clip_vector(vector)
                actual = vector[column] - base_vector[column]
            if abs(actual) <= 1e-14:
                continue
            perturbations.append(self._from_vector(current.parameters, vector))
            signed_steps.append(actual)
            columns.append(column)
        evaluated = {
            item.parameters.cache_key(): item for item in self.screen(perturbations)
        }
        # The last row steers the longest toe toward the requested physical
        # allowance. Unlike rescaling, this row asks the learned SUPR shape
        # directions and placement controls to reach that length naturally.
        jacobian = np.zeros((len(active) + 1, 14), dtype=np.float64)
        usable_columns: list[int] = []
        for item, actual, column in zip(perturbations, signed_steps, columns):
            candidate = evaluated.get(item.cache_key())
            if candidate is None:
                continue
            # Differentiate the clearance the constraint is actually written in
            # rather than projecting point motion onto an assumed direction.
            # Each perturbed candidate has already paid for its signed
            # clearances, so this is both free and exact to finite differences,
            # including the way a slanted wall couples the axes.
            normalized_step = actual / parameter_scales[column]
            delta = self._combined_clearances(candidate)[active] - base_combined[active]
            jacobian[: len(active), column] = (
                np.where(np.isfinite(delta), delta, 0.0) / normalized_step
            )
            jacobian[-1, column] = (
                candidate.placement.toe_x - current.placement.toe_x
            ) / normalized_step
            usable_columns.append(column)
        if not usable_columns:
            return None
        reduced = jacobian[:, usable_columns]
        if not np.any(np.abs(reduced) > 1e-14):
            return None
        normalized_delta = np.zeros(14, dtype=np.float64)
        target = np.concatenate(
            (desired, np.asarray([TARGET_TOE_X - current.placement.toe_x]))
        )
        normalized_delta[usable_columns] = _minimum_norm_joint_update(
            reduced, target
        )
        maximum_step = np.asarray([1.0] * 12 + [0.5, 0.5])
        normalized_delta = np.clip(normalized_delta, -maximum_step, maximum_step)
        if np.linalg.norm(normalized_delta) <= 1e-12:
            return None
        proposed = self._clip_vector(base_vector + normalized_delta * parameter_scales)
        result = self._from_vector(current.parameters, proposed)
        return result if result.cache_key() != current.parameters.cache_key() else None

    def _poll_parameters(
        self,
        current: _ScoredCandidate,
        beta_step: float,
        pitch_step: float,
        offset_step: float,
    ) -> list[_Parameters]:
        """Probe discontinuous collision changes around one local solution."""

        base = self._parameter_vector(current.parameters)
        directions: list[np.ndarray] = []
        for index in (10, 11):
            for sign in (-1.0, 1.0):
                direction = np.zeros(14, dtype=np.float64)
                direction[index] = sign * pitch_step
                directions.append(direction)
        for index in (12, 13):
            for sign in (-1.0, 1.0):
                direction = np.zeros(14, dtype=np.float64)
                direction[index] = sign * offset_step
                directions.append(direction)
        hadamard = _hadamard(16)[:4, :10]
        for row in hadamard:
            for sign in (-1.0, 1.0):
                direction = np.zeros(14, dtype=np.float64)
                direction[:10] = sign * beta_step * row
                directions.append(direction)
        return [
            self._from_vector(
                current.parameters,
                self._clip_vector(base + direction),
            )
            for direction in directions
        ]

    def optimize(self, start: _Parameters, label: str) -> _ScoredCandidate:
        candidates = self.evaluate([start])
        if not candidates:
            raise ValueError(
                f"no valid starting candidate at {label}; rejection counts: "
                f"{dict(sorted(self.rejections.items()))}"
            )
        current = min(candidates, key=lambda item: item.rank_key())
        beta_step = INITIAL_BETA_POLL_STEP
        pitch_step = INITIAL_PITCH_POLL_DEGREES
        offset_step = INITIAL_OFFSET_POLL_CELLS * self.grid_spacing
        for iteration in range(MAX_JOINT_ITERATIONS):
            proposal = self._joint_proposal(current)
            base_vector = self._parameter_vector(current.parameters)
            line_parameters: list[_Parameters] = []
            if proposal is not None:
                proposed_vector = self._parameter_vector(proposal)
                line_parameters.extend(
                    self._from_vector(
                        current.parameters,
                        self._clip_vector(
                            base_vector
                            + factor * (proposed_vector - base_vector)
                        ),
                    )
                    for factor in LINE_SEARCH_FACTORS
                )
            poll_parameters = self._poll_parameters(
                current,
                beta_step,
                pitch_step,
                offset_step,
            )
            local_candidates = self.evaluate(
                [*line_parameters, *poll_parameters]
            )
            best = min(
                [current, *local_candidates], key=lambda item: item.rank_key()
            )
            improved = best.parameters.cache_key() != current.parameters.cache_key()
            self.history.append(
                {
                    "stage": label,
                    "iteration": iteration,
                    "method": "joint_clearance_update_with_exact_collision_poll",
                    "line_search_factors": list(LINE_SEARCH_FACTORS),
                    "valid_local_candidate_count": len(local_candidates),
                    "beta_poll_step": beta_step,
                    "pitch_poll_degrees": pitch_step,
                    "offset_poll_grid_cells": offset_step / self.grid_spacing,
                    "improved": improved,
                    "selected": best.summary(),
                }
            )
            if not improved:
                beta_step *= 0.5
                pitch_step *= 0.5
                offset_step *= 0.5
                if (
                    beta_step < MIN_BETA_POLL_STEP
                    and pitch_step < MIN_PITCH_POLL_DEGREES
                    and offset_step
                    < MIN_OFFSET_POLL_CELLS * self.grid_spacing
                ):
                    break
            else:
                current = best
                if (
                    current.contained
                    and abs(
                        current.placement.toe_allowance_mm
                        - TARGET_TOE_ALLOWANCE_MM
                    )
                    <= 0.1
                ):
                    break
        return current


def _collision_score_from_pairs(
    mesh: TriangleMesh, pairs: np.ndarray, area_equivalence: float
) -> dict[str, Any]:
    area, fraction, colliding = collision_area_fraction(mesh, pairs)
    tier = int(np.ceil(fraction / area_equivalence - 1e-12)) if len(colliding) else 0
    return {
        "collision_pair_count": int(len(pairs)),
        "colliding_foot_face_count": int(len(colliding)),
        "colliding_foot_face_indices": colliding.tolist(),
        "collision_area": area,
        "collision_area_fraction": fraction,
        "collision_area_tier": tier,
    }


def _collision_score(candidate: _ScoredCandidate) -> dict[str, Any]:
    if not candidate.exact_evaluated:
        raise ValueError("candidate has not received exact collision evaluation")
    assert candidate.collision_pairs is not None
    assert candidate.colliding_foot_faces is not None
    assert candidate.affected_foot_faces is not None
    return {
        "collision_pair_count": int(len(candidate.collision_pairs)),
        "colliding_foot_face_count": int(len(candidate.colliding_foot_faces)),
        "colliding_foot_face_indices": candidate.colliding_foot_faces.tolist(),
        "collision_area": candidate.collision_area,
        "collision_area_fraction": candidate.collision_area_fraction,
        "collision_area_tier": candidate.collision_area_tier,
        "affected_foot_face_count": int(len(candidate.affected_foot_faces)),
        "affected_foot_face_indices": candidate.affected_foot_faces.tolist(),
        "affected_area": candidate.affected_area,
        "affected_area_fraction": candidate.affected_area_fraction,
        "affected_area_tier": candidate.affected_area_tier,
    }


def _cavity_summary(analysis: CavityAnalysis) -> dict[str, Any]:
    return {
        "status": analysis.status,
        "support_contact": analysis.support_contact,
        "obstacle_collisions": {
            "intersecting_pair_count": int(len(analysis.collision_pairs)),
            "foot_face_indices": analysis.colliding_foot_face_indices.tolist(),
            "shoe_face_indices": analysis.colliding_shoe_face_indices.tolist(),
        },
        "unsigned_clearance_summaries": analysis.clearance_summaries,
        "signed_clearance": {
            "score_exempt_vertex_indices": (
                analysis.signed_clearance.signed_exempt_vertex_indices.tolist()
            ),
            "score_exempt_face_indices": (
                analysis.signed_clearance.signed_exempt_face_indices.tolist()
            ),
            "outside_area_fraction": analysis.signed_clearance.outside_area_fraction,
            "protrusion_energy": analysis.signed_clearance.protrusion_energy,
            "protrusion_statistics": analysis.signed_clearance.protrusion_statistics,
            "regional_summaries": analysis.signed_clearance.regional_summaries,
        },
    }


def _hadamard(order: int) -> np.ndarray:
    """Return a deterministic ±1 Hadamard matrix for a power-of-two order."""

    if order < 1 or order & (order - 1):
        raise ValueError("Hadamard order must be a positive power of two")
    result = np.ones((1, 1), dtype=np.float64)
    while len(result) < order:
        result = np.block([[result, result], [result, -result]])
    return result


def _deduplicate_parameters(items: list[_Parameters]) -> list[_Parameters]:
    unique: list[_Parameters] = []
    seen: set[tuple[float, ...]] = set()
    for item in items:
        key = item.cache_key()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique


def _beta_templates(initial: _Parameters) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Build individual and coupled ten-beta exploration templates."""

    targeted = [initial.betas.copy()]
    for index in range(1, 10):
        for magnitude in BETA_AXIS_MAGNITUDES:
            for sign in (-1.0, 1.0):
                values = initial.betas.copy()
                values[index] = np.clip(
                    values[index] + sign * magnitude,
                    -MAX_BETA_ABS,
                    MAX_BETA_ABS,
                )
                targeted.append(values)

    directions = _hadamard(16)[:, :9]
    for magnitude in BETA_COUPLED_MAGNITUDES:
        for row in directions:
            for sign in (-1.0, 1.0):
                values = initial.betas.copy()
                values[1:] = np.clip(
                    values[1:] + sign * magnitude * row,
                    -MAX_BETA_ABS,
                    MAX_BETA_ABS,
                )
                targeted.append(values)

    # Beta 0 is also inspected independently. The targeted templates below
    # subsequently use it to reach the requested length without external scale.
    direct = [initial.betas.copy()]
    for magnitude in BETA_AXIS_MAGNITUDES:
        for sign in (-1.0, 1.0):
            values = initial.betas.copy()
            values[0] = np.clip(
                values[0] + sign * magnitude,
                -MAX_BETA_ABS,
                MAX_BETA_ABS,
            )
            direct.append(values)
    return targeted, direct


def _posed_length_ratios(
    search: _Search,
    shapes: np.ndarray,
) -> np.ndarray:
    poses = np.zeros(
        (len(shapes), search.supr_model.num_pose_parameters), dtype=np.float32
    )
    poses[:, SUPR_ANKLE_PITCH_INDEX] = np.deg2rad(
        search.initial.ankle_degrees
    )
    poses[:, SUPR_MIDFOOT_PITCH_INDEX] = np.deg2rad(
        search.initial.midfoot_degrees
    )
    vertices, _ = search.supr_model.evaluate(poses, shapes.astype(np.float32))
    # The fixed reviewed remap sends SUPR Z to shoe X. Translation does not
    # affect the span, so this avoids constructing hundreds of 4x4 matrices.
    return search.context.length_scale * np.ptp(vertices[:, :, 2], axis=1)


def _latin_hypercube_beta_templates(initial: _Parameters) -> np.ndarray:
    """Return deterministic, independently stratified beta 1--9 shapes."""

    rng = np.random.default_rng(LATIN_HYPERCUBE_SEED)
    unit = np.empty((LATIN_HYPERCUBE_SAMPLE_COUNT, 9), dtype=np.float64)
    for dimension in range(9):
        strata = rng.permutation(LATIN_HYPERCUBE_SAMPLE_COUNT)
        unit[:, dimension] = (
            strata + rng.random(LATIN_HYPERCUBE_SAMPLE_COUNT)
        ) / LATIN_HYPERCUBE_SAMPLE_COUNT
    result = np.repeat(
        initial.betas[None, :], LATIN_HYPERCUBE_SAMPLE_COUNT, axis=0
    )
    result[:, 1:] = -MAX_BETA_ABS + 2.0 * MAX_BETA_ABS * unit
    return result


def _solve_beta0_for_length(
    search: _Search,
    templates: np.ndarray,
    target_length_ratio: float,
) -> np.ndarray:
    """Solve beta 0 in parallel without changing the fixed SUPR scale."""

    template_array = np.asarray(templates, dtype=np.float64)
    if template_array.ndim != 2 or template_array.shape[1] != 10:
        raise ValueError("beta templates must have shape (N, 10)")
    if not len(template_array):
        return template_array.copy()
    lower = template_array.copy()
    upper = template_array.copy()
    lower[:, 0] = -MAX_BETA_ABS
    upper[:, 0] = MAX_BETA_ABS
    lower_ratio = _posed_length_ratios(search, lower)
    upper_ratio = _posed_length_ratios(search, upper)
    target = float(target_length_ratio)
    bracketed = (lower_ratio - target) * (upper_ratio - target) <= 0.0

    for _ in range(BETA0_SOLVE_ITERATIONS):
        middle = 0.5 * (lower + upper)
        middle_ratio = _posed_length_ratios(search, middle)
        lower_half = (lower_ratio - target) * (middle_ratio - target) <= 0.0
        choose_upper = bracketed & lower_half
        choose_lower = bracketed & ~lower_half
        upper[choose_upper] = middle[choose_upper]
        upper_ratio[choose_upper] = middle_ratio[choose_upper]
        lower[choose_lower] = middle[choose_lower]
        lower_ratio[choose_lower] = middle_ratio[choose_lower]

    candidates = np.stack((lower, upper), axis=1)
    ratios = np.stack((lower_ratio, upper_ratio), axis=1)
    closest = np.argmin(np.abs(ratios - target), axis=1)
    return candidates[np.arange(len(candidates)), closest]


def _broad_starts(search: _Search, initial: _Parameters) -> list[_Parameters]:
    """Keep legacy starts and add varied-magnitude ten-beta shapes."""

    legacy_templates, legacy_direct = _beta_templates(initial)
    legacy_solved = _solve_beta0_for_length(
        search,
        np.asarray(legacy_templates, dtype=np.float64),
        TARGET_TOE_X,
    )
    latin_templates = _latin_hypercube_beta_templates(initial)
    latin_solved: list[np.ndarray] = []
    for allowance in BETA0_TARGET_ALLOWANCES_MM:
        latin_solved.extend(
            _solve_beta0_for_length(
                search,
                latin_templates,
                toe_allowance_to_foot_length_ratio(allowance),
            )
        )
    shapes = [*legacy_direct, *legacy_solved, *latin_solved]
    return _deduplicate_parameters(
        [replace(initial, betas=np.asarray(shape).copy()) for shape in shapes]
    )


def _restart_candidates(
    candidates: list[_ScoredCandidate], limit: int
) -> list[_ScoredCandidate]:
    """Choose strong but beta-diverse exact candidates for local refinement."""

    ordered = sorted(candidates, key=lambda item: item.rank_key())
    chosen: list[_ScoredCandidate] = []
    for candidate in ordered:
        if all(
            np.linalg.norm(candidate.parameters.betas - other.parameters.betas)
            >= 0.5
            for other in chosen
        ):
            chosen.append(candidate)
            if len(chosen) == limit:
                return chosen
    chosen_keys = {candidate.parameters.cache_key() for candidate in chosen}
    for candidate in ordered:
        if candidate.parameters.cache_key() not in chosen_keys:
            chosen.append(candidate)
            chosen_keys.add(candidate.parameters.cache_key())
            if len(chosen) == limit:
                break
    return chosen


def build_containment_foot_fit(
    supr_model: SuprFootModel,
    neutral_foot_mesh: TriangleMesh,
    normalized_shoe_mesh: TriangleMesh,
    normalized_support_mesh: TriangleMesh,
    normalized_centerline_xz: np.ndarray,
    footbed_source_face_indices: np.ndarray,
    shoe_to_normalized: np.ndarray,
    normalized_to_shoe: np.ndarray,
    support_grid_cell_spacing: float,
    initial_pose_parameters: np.ndarray,
    initial_betas: np.ndarray,
    baseline_fitted_foot: TriangleMesh,
    expected_baseline_collision_pairs: np.ndarray | None = None,
    expected_baseline_status: str | None = None,
    cavity_subdivision: SuprMeshSubdivision | None = None,
) -> ContainmentFootFit:
    """Find the largest support-valid SUPR foot contained by local boundaries.

    Size is not a search variable. The anchored scale carries each candidate's
    own heel-to-toe length into the shoe, so the ten SUPR shape parameters
    change the foot the way real anatomy varies instead of photocopying one
    template smaller.
    """

    if not np.array_equal(supr_model.faces, neutral_foot_mesh.faces):
        raise ValueError("posable and neutral SUPR models must use identical faces")
    grid_spacing = float(support_grid_cell_spacing)
    if not np.isfinite(grid_spacing) or grid_spacing <= 0.0:
        raise ValueError("support_grid_cell_spacing must be finite and positive")
    pose = np.asarray(initial_pose_parameters, dtype=np.float64)
    betas = np.asarray(initial_betas, dtype=np.float64)
    if pose.shape != (supr_model.num_pose_parameters,) or not np.isfinite(pose).all():
        raise ValueError("initial pose must be one finite SUPR pose vector")
    if betas.shape != (supr_model.num_betas,) or not np.isfinite(betas).all():
        raise ValueError("initial betas must be one finite SUPR shape vector")
    inactive = np.ones(len(pose), dtype=bool)
    inactive[[SUPR_ANKLE_PITCH_INDEX, SUPR_MIDFOOT_PITCH_INDEX]] = False
    if np.any(np.abs(pose[inactive]) > 1e-8):
        raise ValueError("only ankle and midfoot pitch may be active")
    if np.any(np.abs(betas) > MAX_BETA_ABS + 1e-12):
        raise ValueError("initial SUPR betas lie outside [-3, 3]")

    forward = np.asarray(shoe_to_normalized, dtype=np.float64)
    inverse = np.asarray(normalized_to_shoe, dtype=np.float64)
    if forward.shape != (4, 4) or inverse.shape != (4, 4):
        raise ValueError("shoe normalization matrices must have shape (4, 4)")
    if not np.isfinite(forward).all() or not np.isfinite(inverse).all():
        raise ValueError("shoe normalization matrices must be finite")
    if not np.allclose(inverse @ forward, np.eye(4), atol=1e-9, rtol=0.0):
        raise ValueError("shoe normalization matrices are not mutual inverses")

    context = build_support_placement_context(
        neutral_foot_mesh,
        normalized_shoe_mesh,
        normalized_support_mesh,
        normalized_centerline_xz,
    )
    initial = _Parameters(
        heel_offset_x=0.0,
        lateral_offset_z=0.0,
        ankle_degrees=float(np.rad2deg(pose[SUPR_ANKLE_PITCH_INDEX])),
        midfoot_degrees=float(np.rad2deg(pose[SUPR_MIDFOOT_PITCH_INDEX])),
        betas=betas.copy(),
    )
    cavity = CavityEvaluator.build(
        normalized_shoe_mesh,
        normalized_support_mesh,
        footbed_source_face_indices,
        baseline_fitted_foot,
        normalized_centerline_xz,
    )
    source_baseline_analysis = cavity.analyze(
        baseline_fitted_foot,
        context.regions.plantar_vertex_indices,
        context.regions.plantar_face_indices,
    )
    if expected_baseline_collision_pairs is not None:
        expected = np.asarray(expected_baseline_collision_pairs, dtype=np.int64)
        expected = expected.reshape(-1, 2) if expected.size else np.empty((0, 2), dtype=np.int64)
        if not np.array_equal(source_baseline_analysis.collision_pairs, expected):
            raise ValueError(
                "stored Checkpoint 5 mesh does not reproduce the Checkpoint 6 collision pairs"
            )
    if (
        expected_baseline_status is not None
        and source_baseline_analysis.status != expected_baseline_status
    ):
        raise ValueError("stored Checkpoint 5 mesh does not reproduce the Checkpoint 6 status")

    search = _Search(
        supr_model,
        neutral_foot_mesh,
        context,
        cavity,
        initial,
        grid_spacing,
        cavity_subdivision,
    )
    if cavity_subdivision is None:
        baseline_analysis = source_baseline_analysis
        baseline_scoring_mesh = baseline_fitted_foot
    else:
        baseline_scoring_mesh = search.cavity_mesh(baseline_fitted_foot.vertices)
        baseline_analysis = cavity.analyze(
            baseline_scoring_mesh,
            search.cavity_regions.plantar_vertex_indices,
            search.cavity_regions.plantar_face_indices,
        )
    full_analysis_cache: dict[tuple[float, ...], CavityAnalysis] = {}

    def full_analysis(candidate: _ScoredCandidate) -> CavityAnalysis:
        key = candidate.parameters.cache_key()
        if key not in full_analysis_cache:
            cavity_placement = search.cavity_placement(candidate.placement)
            mesh = search.cavity_mesh(candidate.placement.aligned_vertices)
            exempt_vertices, exempt_faces = _ankle_signed_exemptions(
                cavity_placement,
                search.cavity_faces,
                cavity.numerical_tolerance,
            )
            full_analysis_cache[key] = cavity.analyze(
                mesh,
                search.cavity_regions.plantar_vertex_indices,
                search.cavity_regions.plantar_face_indices,
                exempt_vertices,
                exempt_faces,
            )
        return full_analysis_cache[key]

    baseline_screened = search.screen([initial])
    if not baseline_screened:
        raise ValueError("Checkpoint 5 parameters no longer pass support placement")
    corrected_baseline = baseline_screened[0]
    broad_starts = _broad_starts(search, initial)
    broad_screened = search.screen(broad_starts)
    broad_target = [
        candidate for candidate in broad_screened if candidate.in_target_band
    ]
    if not broad_target:
        raise ValueError(
            "SUPR could not produce a support-valid foot with 18-22 mm toe space"
        )
    broad_exact = search.evaluate(
        [candidate.parameters for candidate in broad_target]
    )
    if not broad_exact:
        raise ValueError("no broad SUPR candidate passed exact evaluation")
    restarts = _restart_candidates(broad_exact, MAX_RESTARTS)
    refined = [
        search.optimize(candidate.parameters, f"restart_{index}")
        for index, candidate in enumerate(restarts)
    ]

    baseline_score = _collision_score_from_pairs(
        baseline_scoring_mesh,
        baseline_analysis.collision_pairs,
        search.area_equivalence_fraction,
    )
    baseline_outside_tier = corrected_baseline.outside_area_tier
    exact_by_key = {
        candidate.parameters.cache_key(): candidate
        for candidate in [*broad_exact, *refined]
    }
    target_candidates = [
        candidate
        for candidate in exact_by_key.values()
        if candidate.in_target_band
        and candidate.collision_area_tier <= baseline_score["collision_area_tier"]
        and candidate.outside_area_tier <= baseline_outside_tier
    ]
    if not target_candidates:
        raise ValueError(
            "no 18-22 mm candidate preserved both baseline collision and "
            "signed-outside area"
        )
    selected = min(target_candidates, key=lambda item: item.rank_key())
    final_analysis = full_analysis(selected)
    contained = selected.contained and final_analysis.status == "clear"
    posed_to_normalized = selected.placement.transform
    normalized_to_posed = np.linalg.inv(posed_to_normalized)
    posed_to_original = inverse @ posed_to_normalized
    original_to_posed = np.linalg.inv(posed_to_original)
    aligned_joints = transform_points(selected.placement.posed_joints, posed_to_normalized)
    search_record = {
        "method": "broad_ten_beta_search_with_joint_containment_refinement",
        "sizing_policy": (
            "anchored scale; foot length follows the SUPR shape parameters and "
            "is never independently scaled or renormalized"
        ),
        "stopping_reason": (
            "contained_target_fit_found"
            if contained
            else "best_residual_target_fit"
        ),
        "candidate_count": len(search.cache),
        "valid_candidate_count": sum(isinstance(value, _ScoredCandidate) for value in search.cache.values()),
        "exact_candidate_count": search.exact_evaluation_count,
        "rejection_counts": dict(sorted(search.rejections.items())),
        "broad_search": {
            "generated_start_count": len(broad_starts),
            "valid_screened_count": len(broad_screened),
            "target_band_screened_count": len(broad_target),
            "exact_candidate_count": len(broad_exact),
            "restart_count": len(restarts),
            "individual_beta_magnitudes": list(BETA_AXIS_MAGNITUDES),
            "coupled_beta_magnitudes": list(BETA_COUPLED_MAGNITUDES),
            "latin_hypercube": {
                "sample_count": LATIN_HYPERCUBE_SAMPLE_COUNT,
                "random_seed": LATIN_HYPERCUBE_SEED,
                "dimensions": "betas_1_through_9",
                "range": [-MAX_BETA_ABS, MAX_BETA_ABS],
                "beta0_target_toe_allowances_mm": list(
                    BETA0_TARGET_ALLOWANCES_MM
                ),
            },
            "beta0_target_solve_iterations": BETA0_SOLVE_ITERATIONS,
            "restart_candidates": [candidate.summary() for candidate in restarts],
            "refined_candidates": [candidate.summary() for candidate in refined],
        },
        "limits": {
            "anchor_foot_length_ratio": ANCHOR_FOOT_LENGTH_RATIO,
            "target_toe_allowance_mm": TARGET_TOE_ALLOWANCE_MM,
            "accepted_toe_allowance_mm": [
                MIN_TARGET_TOE_ALLOWANCE_MM,
                MAX_TARGET_TOE_ALLOWANCE_MM,
            ],
            "accepted_foot_length_ratio_at_zero_heel_offset": [
                TARGET_MIN_RATIO,
                TARGET_MAX_RATIO,
            ],
            "heel_offset": "nonnegative; bounded by toe allowance and support",
            "lateral_offset": "no fixed clamp; bounded by support validity",
            "pitch_change_degrees": MAX_PITCH_CHANGE_DEGREES,
            "beta": [-MAX_BETA_ABS, MAX_BETA_ABS],
        },
        "joint_update": {
            "maximum_iterations": MAX_JOINT_ITERATIONS,
            "jacobian_source": "finite difference of signed combined clearance",
            "beta_perturbation": JACOBIAN_BETA_STEP,
            "pitch_perturbation_degrees": JACOBIAN_PITCH_STEP_DEGREES,
            "offset_perturbation_grid_cells": JACOBIAN_OFFSET_STEP_CELLS,
            "line_search_factors": list(LINE_SEARCH_FACTORS),
            "maximum_restarts": MAX_RESTARTS,
        },
        "baseline_no_regression": {
            "maximum_collision_area_tier": baseline_score["collision_area_tier"],
            "maximum_signed_outside_area_tier": baseline_outside_tier,
            "signed_score_exempt_vertex_count": int(
                len(corrected_baseline.signed.signed_exempt_vertex_indices)
            ),
            "signed_score_exempt_face_count": int(
                len(corrected_baseline.signed.signed_exempt_face_indices)
            ),
        },
        "baseline": _cavity_summary(baseline_analysis),
        "selected": selected.summary(),
        "history": search.history,
    }
    selected_cavity_placement = search.cavity_placement(selected.placement)
    return ContainmentFootFit(
        pose_parameters=selected.placement.pose,
        betas=selected.parameters.betas,
        foot_faces=search.cavity_faces,
        posed_vertices=selected_cavity_placement.posed_vertices,
        posed_joints=selected.placement.posed_joints,
        aligned_vertices=selected_cavity_placement.aligned_vertices,
        aligned_joints=aligned_joints,
        posed_supr_to_normalized_shoe=posed_to_normalized,
        normalized_shoe_to_posed_supr=normalized_to_posed,
        posed_supr_to_original_shoe=posed_to_original,
        original_shoe_to_posed_supr=original_to_posed,
        shoe_to_normalized=forward,
        normalized_to_shoe=inverse,
        placement=selected.placement,
        baseline_cavity=baseline_analysis,
        final_cavity=final_analysis,
        baseline_collision_score=baseline_score,
        final_collision_score=_collision_score(selected),
        search=search_record,
    )
