"""Natural, collision-aware lower-leg exit fitting for accepted SUPR feet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import trimesh

from .cavity import CavityEvaluator
from .mesh import TriangleMesh
from .supr_lower_leg import (
    FittedFootLowerLeg,
    PosableSuprLowerLegModel,
    PosedSuprLowerLeg,
    RIGHT_ANKLE_JOINT_INDEX,
    RIGHT_FOOT_JOINT_INDEX,
    RIGHT_KNEE_JOINT_INDEX,
    attach_lower_leg_to_fitted_dense_foot,
)


ANKLE_PITCH_RANGE_DEGREES = (-20.0, 20.0)
ANKLE_ROLL_RANGE_DEGREES = (-15.0, 15.0)
COARSE_POSE_STEP_DEGREES = 5.0
MAX_BETA_ABSOLUTE = 1.0
MAX_BETA_L2 = 1.5
MIN_GIRTH_RATIO = 0.95
MAX_GIRTH_RATIO = 1.05
MIN_SHANK_LENGTH_RATIO = 0.97
MAX_SHANK_LENGTH_RATIO = 1.03


@dataclass(frozen=True)
class LowerLegShapeMetrics:
    knee_to_ankle_length: float
    low_shaft_girth: float
    calf_girth: float
    knee_girth: float

    def to_dict(self) -> dict[str, float]:
        return {
            "knee_to_ankle_length": self.knee_to_ankle_length,
            "low_shaft_girth": self.low_shaft_girth,
            "calf_girth": self.calf_girth,
            "knee_girth": self.knee_girth,
        }


@dataclass(frozen=True)
class _Candidate:
    posed_leg: PosedSuprLowerLeg
    attachment: FittedFootLowerLeg
    query_mesh: TriangleMesh
    query_face_indices: np.ndarray
    collision_pairs: np.ndarray
    ignored_degenerate_query_face_indices: np.ndarray
    colliding_query_face_indices: np.ndarray
    collision_area: float
    collision_area_fraction: float
    shape_metrics: LowerLegShapeMetrics
    shape_ratios: dict[str, float]

    @property
    def beta_l2(self) -> float:
        return float(np.linalg.norm(self.posed_leg.betas))


@dataclass(frozen=True)
class LowerLegCollarFit:
    """Selected natural shank attachment and exact collar-intersection result."""

    selected: _Candidate
    baseline: _Candidate
    best_pose_only: _Candidate
    evaluated_candidate_count: int
    rejected_unnatural_shape_count: int
    stopping_reason: str
    collision_equivalence_area_fraction: float

    @property
    def status(self) -> str:
        return (
            "clear_exit"
            if len(self.selected.collision_pairs) == 0
            else "residual_collar_intersections"
        )

    def _candidate_record(self, candidate: _Candidate) -> dict[str, Any]:
        leg_count = len(candidate.attachment.lower_leg_face_indices)
        hits = candidate.colliding_query_face_indices
        return {
            "ankle_pitch_degrees": candidate.posed_leg.ankle_pitch_degrees,
            "ankle_roll_degrees": candidate.posed_leg.ankle_roll_degrees,
            "betas": candidate.posed_leg.betas.tolist(),
            "beta_l2": candidate.beta_l2,
            "donor_foot_anchor_rms": candidate.posed_leg.donor_foot_anchor_rms,
            "shape_metrics": candidate.shape_metrics.to_dict(),
            "shape_ratios_to_neutral": candidate.shape_ratios,
            "exact_pair_count": int(len(candidate.collision_pairs)),
            "unique_query_face_count": int(len(hits)),
            "unique_lower_leg_face_count": int(np.count_nonzero(hits < leg_count)),
            "unique_bridge_face_count": int(np.count_nonzero(hits >= leg_count)),
            "affected_surface_area": candidate.collision_area,
            "affected_surface_area_fraction": candidate.collision_area_fraction,
        }

    def to_dict(self) -> dict[str, Any]:
        selected = self.selected
        pairs = selected.collision_pairs
        leg_count = len(selected.attachment.lower_leg_face_indices)
        leg_hits = selected.colliding_query_face_indices[
            selected.colliding_query_face_indices < leg_count
        ]
        bridge_hits = selected.colliding_query_face_indices[
            selected.colliding_query_face_indices >= leg_count
        ]
        return {
            "method": "pose_first_natural_supr_lower_leg_fit",
            "status": self.status,
            "policy": {
                "fitted_foot": "preserved exactly",
                "primary_controls": ["ankle_pitch", "ankle_roll"],
                "secondary_controls": "first ten full-body SUPR betas",
                "shape_interpretation": (
                    "representative near-neutral male lower leg; not an inferred wearer"
                ),
                "residual_policy": (
                    "retain natural anatomy rather than over-thin the shaft"
                ),
            },
            "natural_shape_limits": {
                "maximum_absolute_beta": MAX_BETA_ABSOLUTE,
                "maximum_beta_l2": MAX_BETA_L2,
                "girth_ratio_to_neutral": [MIN_GIRTH_RATIO, MAX_GIRTH_RATIO],
                "shank_length_ratio_to_neutral": [
                    MIN_SHANK_LENGTH_RATIO,
                    MAX_SHANK_LENGTH_RATIO,
                ],
            },
            "pose_limits_degrees": {
                "ankle_pitch": list(ANKLE_PITCH_RANGE_DEGREES),
                "ankle_roll": list(ANKLE_ROLL_RANGE_DEGREES),
            },
            "baseline": self._candidate_record(self.baseline),
            "best_pose_only": self._candidate_record(self.best_pose_only),
            "selected": self._candidate_record(selected),
            "search": {
                "evaluated_candidate_count": self.evaluated_candidate_count,
                "rejected_unnatural_shape_count": self.rejected_unnatural_shape_count,
                "collision_equivalence_area_fraction": (
                    self.collision_equivalence_area_fraction
                ),
                "stopping_reason": self.stopping_reason,
            },
            "attachment": selected.attachment.to_dict(),
            "collar_intersections": {
                "exact_pair_count": int(len(pairs)),
                "shoe_face_indices": (
                    np.unique(pairs[:, 1]).tolist() if len(pairs) else []
                ),
                "lower_leg_face_indices": selected.attachment.lower_leg_face_indices[
                    leg_hits
                ].tolist(),
                "bridge_face_indices": selected.attachment.bridge_face_indices[
                    bridge_hits - leg_count
                ].tolist(),
                "ignored_degenerate_query_face_indices": (
                    selected.ignored_degenerate_query_face_indices.tolist()
                ),
            },
        }


def _section_girth(
    vertices: np.ndarray,
    faces: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
) -> float:
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    segments = trimesh.intersections.mesh_plane(
        mesh, plane_normal=normal, plane_origin=origin
    )
    if len(segments) == 0:
        raise ValueError("lower-leg cross section does not intersect the shank")
    return float(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1).sum())


def _shape_metrics(
    posed: PosedSuprLowerLeg,
    native_faces: np.ndarray,
) -> LowerLegShapeMetrics:
    joints = posed.joints_reference
    ankle = joints[RIGHT_ANKLE_JOINT_INDEX]
    knee = joints[RIGHT_KNEE_JOINT_INDEX]
    axis = knee - ankle
    length = float(np.linalg.norm(axis))
    if not np.isfinite(length) or length <= np.finfo(np.float64).eps:
        raise ValueError("posed lower leg has invalid knee-to-ankle length")
    axis /= length
    girths = [
        _section_girth(
            posed.native_vertices,
            native_faces,
            ankle + fraction * length * axis,
            axis,
        )
        for fraction in (0.20, 0.55, 0.95)
    ]
    if not np.isfinite(girths).all() or np.any(np.asarray(girths) <= 0.0):
        raise ValueError("posed lower leg has invalid cross-sectional girth")
    return LowerLegShapeMetrics(length, *girths)


def _metric_ratios(
    metrics: LowerLegShapeMetrics,
    neutral: LowerLegShapeMetrics,
) -> dict[str, float]:
    return {
        key: float(getattr(metrics, key) / getattr(neutral, key))
        for key in (
            "knee_to_ankle_length",
            "low_shaft_girth",
            "calf_girth",
            "knee_girth",
        )
    }


def _natural_shape(betas: np.ndarray, ratios: dict[str, float]) -> bool:
    return bool(
        np.max(np.abs(betas)) <= MAX_BETA_ABSOLUTE + 1.0e-12
        and np.linalg.norm(betas) <= MAX_BETA_L2 + 1.0e-12
        and MIN_SHANK_LENGTH_RATIO
        <= ratios["knee_to_ankle_length"]
        <= MAX_SHANK_LENGTH_RATIO
        and all(
            MIN_GIRTH_RATIO <= ratios[name] <= MAX_GIRTH_RATIO
            for name in ("low_shaft_girth", "calf_girth", "knee_girth")
        )
    )


def build_lower_leg_collar_fit(
    shoe_mesh: TriangleMesh,
    footbed_mesh: TriangleMesh,
    footbed_source_face_indices: np.ndarray,
    fitted_dense_foot: TriangleMesh,
    normalized_centerline_xz: np.ndarray,
    canonical_dense_foot_vertices: np.ndarray,
    dense_foot_ankle_loop: np.ndarray,
    ankle_loop_correspondence: np.ndarray,
    lower_leg_model: PosableSuprLowerLegModel,
) -> LowerLegCollarFit:
    """Fit a natural lower-leg exit while preserving the accepted foot exactly."""

    native_faces = lower_leg_model.neutral_lower_leg.mesh.faces
    neutral_posed = lower_leg_model.evaluate(np.zeros(10), 0.0, 0.0)
    neutral_metrics = _shape_metrics(neutral_posed, native_faces)
    evaluator = CavityEvaluator.build(
        shoe_mesh,
        footbed_mesh,
        footbed_source_face_indices,
        fitted_dense_foot,
        normalized_centerline_xz,
    )
    cache: dict[tuple[float, ...], _Candidate] = {}
    rejected_unnatural = 0
    collision_quantum: float | None = None

    def evaluate(betas: np.ndarray, pitch: float, roll: float) -> _Candidate | None:
        nonlocal rejected_unnatural, collision_quantum
        shape = np.asarray(betas, dtype=np.float64)
        key = tuple(np.round(np.r_[shape, pitch, roll], 8).tolist())
        if key in cache:
            return cache[key]
        posed = lower_leg_model.evaluate(shape, pitch, roll)
        metrics = _shape_metrics(posed, native_faces)
        ratios = _metric_ratios(metrics, neutral_metrics)
        if not _natural_shape(shape, ratios):
            rejected_unnatural += 1
            return None
        attachment = attach_lower_leg_to_fitted_dense_foot(
            fitted_dense_foot,
            canonical_dense_foot_vertices,
            posed.dense_vertices,
            lower_leg_model.subdivision.faces,
            dense_foot_ankle_loop,
            ankle_loop_correspondence,
        )
        query_face_indices = np.concatenate(
            (attachment.lower_leg_face_indices, attachment.bridge_face_indices)
        )
        query = TriangleMesh(
            attachment.mesh.vertices,
            attachment.mesh.faces[query_face_indices],
        )
        pairs, ignored = evaluator.collision_pairs(query)
        hits = (
            np.unique(pairs[:, 0])
            if len(pairs)
            else np.empty(0, dtype=np.int64)
        )
        triangles = query.vertices[query.faces]
        areas = 0.5 * np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        total_area = float(np.sum(areas))
        affected_area = float(np.sum(areas[hits]))
        if collision_quantum is None:
            collision_quantum = float(np.median(areas) / total_area)
        candidate = _Candidate(
            posed_leg=posed,
            attachment=attachment,
            query_mesh=query,
            query_face_indices=query_face_indices,
            collision_pairs=pairs,
            ignored_degenerate_query_face_indices=ignored,
            colliding_query_face_indices=hits,
            collision_area=affected_area,
            collision_area_fraction=affected_area / total_area,
            shape_metrics=metrics,
            shape_ratios=ratios,
        )
        cache[key] = candidate
        return candidate

    def rank(candidate: _Candidate) -> tuple[Any, ...]:
        if collision_quantum is None:
            raise RuntimeError("collision equivalence has not been initialized")
        tier = int(
            np.floor(candidate.collision_area_fraction / collision_quantum + 0.5)
        )
        pose_norm = (
            (candidate.posed_leg.ankle_pitch_degrees / 20.0) ** 2
            + (candidate.posed_leg.ankle_roll_degrees / 15.0) ** 2
        )
        return (
            bool(len(candidate.collision_pairs)),
            tier,
            candidate.beta_l2,
            pose_norm,
            candidate.collision_area_fraction,
            int(len(candidate.collision_pairs)),
            tuple(np.round(candidate.posed_leg.betas, 8)),
            candidate.posed_leg.ankle_pitch_degrees,
            candidate.posed_leg.ankle_roll_degrees,
        )

    zeros = np.zeros(10, dtype=np.float64)
    baseline = evaluate(zeros, 0.0, 0.0)
    if baseline is None:
        raise RuntimeError("neutral lower-leg candidate was rejected")
    if len(baseline.collision_pairs) == 0:
        return LowerLegCollarFit(
            selected=baseline,
            baseline=baseline,
            best_pose_only=baseline,
            evaluated_candidate_count=len(cache),
            rejected_unnatural_shape_count=rejected_unnatural,
            stopping_reason="neutral_lower_leg_already_clear",
            collision_equivalence_area_fraction=float(collision_quantum),
        )

    pose_candidates = [baseline]
    for pitch in np.arange(
        ANKLE_PITCH_RANGE_DEGREES[0],
        ANKLE_PITCH_RANGE_DEGREES[1] + 0.5 * COARSE_POSE_STEP_DEGREES,
        COARSE_POSE_STEP_DEGREES,
    ):
        for roll in np.arange(
            ANKLE_ROLL_RANGE_DEGREES[0],
            ANKLE_ROLL_RANGE_DEGREES[1] + 0.5 * COARSE_POSE_STEP_DEGREES,
            COARSE_POSE_STEP_DEGREES,
        ):
            candidate = evaluate(zeros, float(pitch), float(roll))
            if candidate is not None:
                pose_candidates.append(candidate)
    best_pose = min(pose_candidates, key=rank)

    for step in (2.5, 1.25):
        neighbours = [best_pose]
        for pitch_delta, roll_delta in ((-step, 0.0), (step, 0.0), (0.0, -step), (0.0, step)):
            pitch = float(
                np.clip(
                    best_pose.posed_leg.ankle_pitch_degrees + pitch_delta,
                    *ANKLE_PITCH_RANGE_DEGREES,
                )
            )
            roll = float(
                np.clip(
                    best_pose.posed_leg.ankle_roll_degrees + roll_delta,
                    *ANKLE_ROLL_RANGE_DEGREES,
                )
            )
            candidate = evaluate(zeros, pitch, roll)
            if candidate is not None:
                neighbours.append(candidate)
        best_pose = min(neighbours, key=rank)
    if len(best_pose.collision_pairs) == 0:
        return LowerLegCollarFit(
            selected=best_pose,
            baseline=baseline,
            best_pose_only=best_pose,
            evaluated_candidate_count=len(cache),
            rejected_unnatural_shape_count=rejected_unnatural,
            stopping_reason="pose_only_clear_exit",
            collision_equivalence_area_fraction=float(collision_quantum),
        )

    current = best_pose
    for beta_step in (0.5, 0.25):
        for _ in range(2):
            neighbours = [current]
            for beta_index in range(10):
                for sign in (-1.0, 1.0):
                    betas = current.posed_leg.betas.copy()
                    betas[beta_index] += sign * beta_step
                    candidate = evaluate(
                        betas,
                        current.posed_leg.ankle_pitch_degrees,
                        current.posed_leg.ankle_roll_degrees,
                    )
                    if candidate is not None:
                        neighbours.append(candidate)
            selected = min(neighbours, key=rank)
            if rank(selected) >= rank(current):
                break
            current = selected
            if len(current.collision_pairs) == 0:
                break
        if len(current.collision_pairs) == 0:
            break

    return LowerLegCollarFit(
        selected=current,
        baseline=baseline,
        best_pose_only=best_pose,
        evaluated_candidate_count=len(cache),
        rejected_unnatural_shape_count=rejected_unnatural,
        stopping_reason=(
            "natural_shape_clear_exit"
            if len(current.collision_pairs) == 0
            else "natural_limits_reached_with_residual_intersections"
        ),
        collision_equivalence_area_fraction=float(collision_quantum),
    )
