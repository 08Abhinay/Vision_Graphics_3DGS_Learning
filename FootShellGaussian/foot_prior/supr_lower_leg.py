"""Deterministic neutral lower-leg donor for the canonical SUPR foot."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
import sys
import tempfile
import types
from typing import Any

import numpy as np

from .mesh import TriangleMesh, transform_mesh
from .supr_foot import SuprMeshSubdivision


RIGHT_KNEE_JOINT_INDEX = 5
RIGHT_ANKLE_JOINT_INDEX = 8
RIGHT_FOOT_JOINT_INDEX = 11
RIGHT_TOE_JOINT_INDICES = tuple(range(65, 75))
FOOT_ROOT_JOINT_INDEX = 0
FOOT_ANKLE_JOINT_INDEX = 1
FOOT_MIDFOOT_JOINT_INDEX = 2
EXPECTED_NATIVE_ANKLE_COUNT = 15
EXPECTED_NATIVE_KNEE_COUNT = 17
FULL_BODY_VERTEX_COUNT = 10475
FULL_BODY_FACE_COUNT = 20908
FULL_BODY_JOINT_COUNT = 75
FULL_BODY_POSE_PARAMETER_COUNT = 225
RIGHT_ANKLE_PITCH_PARAMETER_INDEX = RIGHT_ANKLE_JOINT_INDEX * 3
RIGHT_ANKLE_ROLL_PARAMETER_INDEX = RIGHT_ANKLE_JOINT_INDEX * 3 + 2


@dataclass(frozen=True)
class SuprLowerLeg:
    """One aligned, open right shank with full-body source provenance."""

    mesh: TriangleMesh
    source_vertex_indices: np.ndarray
    source_face_indices: np.ndarray
    proximal_boundary_vertex_indices: np.ndarray
    distal_boundary_vertex_indices: np.ndarray
    source_joints_reference: np.ndarray
    body_to_reference: np.ndarray
    frame_rotation_degrees: float
    in_plane_translation: np.ndarray
    distal_centroid_residual: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "rigid_full_body_alignment_and_anatomical_plane_cut",
            "side": "right",
            "included_joint_indices": {
                "right_knee": RIGHT_KNEE_JOINT_INDEX,
                "right_ankle": RIGHT_ANKLE_JOINT_INDEX,
                "right_foot": RIGHT_FOOT_JOINT_INDEX,
                "right_toes": list(RIGHT_TOE_JOINT_INDICES),
            },
            "vertex_count": int(len(self.mesh.vertices)),
            "face_count": int(len(self.mesh.faces)),
            "source_vertex_indices": self.source_vertex_indices.tolist(),
            "source_face_indices": self.source_face_indices.tolist(),
            "proximal_boundary_vertex_indices": (
                self.proximal_boundary_vertex_indices.tolist()
            ),
            "distal_boundary_vertex_indices": (
                self.distal_boundary_vertex_indices.tolist()
            ),
            "body_to_reference": self.body_to_reference.tolist(),
            "frame_rotation_degrees": self.frame_rotation_degrees,
            "in_plane_translation": self.in_plane_translation.tolist(),
            "distal_centroid_residual": self.distal_centroid_residual,
            "bounds": self.mesh.bounds.tolist(),
            "full_body_shape_parameters_transferred": False,
        }


@dataclass(frozen=True)
class FittedFootLowerLeg:
    """One accepted dense fitted foot with a rigid neutral shank attached."""

    mesh: TriangleMesh
    foot_vertex_count: int
    foot_face_count: int
    lower_leg_vertex_indices: np.ndarray
    lower_leg_face_indices: np.ndarray
    bridge_face_indices: np.ndarray
    canonical_leg_to_fitted_ankle: np.ndarray
    ankle_loop_rms_residual: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "rigid_ankle_loop_alignment_with_triangular_bridge",
            "foot_geometry_changed": False,
            "lower_leg_shape": "neutral male SUPR donor",
            "counts": {
                "foot_vertices": self.foot_vertex_count,
                "foot_faces": self.foot_face_count,
                "lower_leg_vertices": int(len(self.lower_leg_vertex_indices)),
                "lower_leg_faces": int(len(self.lower_leg_face_indices)),
                "bridge_faces": int(len(self.bridge_face_indices)),
            },
            "canonical_leg_to_fitted_ankle": (
                self.canonical_leg_to_fitted_ankle.tolist()
            ),
            "ankle_loop_rms_residual": self.ankle_loop_rms_residual,
        }


@dataclass(frozen=True)
class PosedSuprLowerLeg:
    """One full-body SUPR shank posed relative to an unchanged donor foot."""

    native_vertices: np.ndarray
    dense_vertices: np.ndarray
    joints_reference: np.ndarray
    betas: np.ndarray
    ankle_pitch_degrees: float
    ankle_roll_degrees: float
    donor_foot_anchor_rms: float


class PosableSuprLowerLegModel:
    """NumPy-facing wrapper for conservative full-body shank evaluation."""

    def __init__(
        self,
        model: Any,
        torch_module: Any,
        num_betas: int,
        neutral_lower_leg: SuprLowerLeg,
        subdivision: SuprMeshSubdivision,
        donor_foot_vertex_indices: np.ndarray,
        neutral_reference_vertices: np.ndarray,
        neutral_reference_joints: np.ndarray,
    ) -> None:
        self._model = model
        self._torch = torch_module
        self.num_betas = int(num_betas)
        self.neutral_lower_leg = neutral_lower_leg
        self.subdivision = subdivision
        self.donor_foot_vertex_indices = np.asarray(
            donor_foot_vertex_indices, dtype=np.int64
        )
        self.neutral_reference_vertices = np.asarray(
            neutral_reference_vertices, dtype=np.float64
        )
        self.neutral_reference_joints = np.asarray(
            neutral_reference_joints, dtype=np.float64
        )

    def evaluate(
        self,
        betas: np.ndarray,
        ankle_pitch_degrees: float,
        ankle_roll_degrees: float,
    ) -> PosedSuprLowerLeg:
        """Shape the shank and pose it while holding the donor foot rigidly fixed."""

        shape = np.asarray(betas, dtype=np.float64)
        pitch = float(ankle_pitch_degrees)
        roll = float(ankle_roll_degrees)
        if shape.shape != (self.num_betas,) or not np.isfinite(shape).all():
            raise ValueError(f"betas must have finite shape ({self.num_betas},)")
        if not np.isfinite((pitch, roll)).all():
            raise ValueError("ankle pitch and roll must be finite")
        if np.all(shape == 0.0) and pitch == 0.0 and roll == 0.0:
            native = self.neutral_lower_leg.mesh.vertices.copy()
            return PosedSuprLowerLeg(
                native_vertices=native,
                dense_vertices=self.subdivision.apply_vertices(native),
                joints_reference=self.neutral_reference_joints.copy(),
                betas=shape.copy(),
                ankle_pitch_degrees=0.0,
                ankle_roll_degrees=0.0,
                donor_foot_anchor_rms=0.0,
            )

        device = self._torch.device("cuda", self._torch.cuda.current_device())
        pose = self._torch.zeros(
            (1, FULL_BODY_POSE_PARAMETER_COUNT),
            dtype=self._torch.float32,
            device=device,
        )
        pose[0, RIGHT_ANKLE_PITCH_PARAMETER_INDEX] = np.deg2rad(pitch)
        pose[0, RIGHT_ANKLE_ROLL_PARAMETER_INDEX] = np.deg2rad(roll)
        shape_tensor = self._torch.as_tensor(
            shape.astype(np.float32, copy=False)[None, :], device=device
        )
        translation = self._torch.zeros(
            (1, 3), dtype=self._torch.float32, device=device
        )
        with self._torch.no_grad():
            output = self._model(pose, shape_tensor, translation)
        vertices = output[0].detach().cpu().numpy().astype(np.float64, copy=False)
        joints = (
            output.J_transformed[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64, copy=False)
        )
        vertices = _transform_points(
            vertices, self.neutral_lower_leg.body_to_reference
        )
        joints = _transform_points(
            joints, self.neutral_lower_leg.body_to_reference
        )
        anchor, residual = _rigid_correspondence_transform(
            vertices[self.donor_foot_vertex_indices],
            self.neutral_reference_vertices[self.donor_foot_vertex_indices],
        )
        vertices = _transform_points(vertices, anchor)
        joints = _transform_points(joints, anchor)
        native = vertices[self.neutral_lower_leg.source_vertex_indices]
        return PosedSuprLowerLeg(
            native_vertices=native,
            dense_vertices=self.subdivision.apply_vertices(native),
            joints_reference=joints,
            betas=shape.copy(),
            ankle_pitch_degrees=pitch,
            ankle_roll_degrees=roll,
            donor_foot_anchor_rms=float(residual),
        )


class _SerializedChumpyArray:
    """Minimal pickle target for the one legacy shapedirs value we do not use."""

    def __setstate__(self, state: object) -> None:
        if isinstance(state, dict):
            self.__dict__.update(state)
        else:
            self.state = state


def _load_legacy_chumpy_container(path: Path) -> object:
    package = types.ModuleType("chumpy")
    module = types.ModuleType("chumpy.ch")
    module.Ch = _SerializedChumpyArray
    package.ch = module
    previous_package = sys.modules.get("chumpy")
    previous_module = sys.modules.get("chumpy.ch")
    sys.modules["chumpy"] = package
    sys.modules["chumpy.ch"] = module
    try:
        return np.load(path, allow_pickle=True, encoding="latin1")
    finally:
        if previous_package is None:
            sys.modules.pop("chumpy", None)
        else:
            sys.modules["chumpy"] = previous_package
        if previous_module is None:
            sys.modules.pop("chumpy.ch", None)
        else:
            sys.modules["chumpy.ch"] = previous_module


def _load_full_body_model(path: str | Path) -> Mapping[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    try:
        container = np.load(source, allow_pickle=True, encoding="latin1")
    except ModuleNotFoundError as error:
        if error.name != "chumpy":
            raise
        container = _load_legacy_chumpy_container(source)
    payload = container.item()
    if not isinstance(payload, Mapping):
        raise ValueError("full-body SUPR model must contain one mapping")
    required = {"v_template", "f", "weights", "J", "kintree_table"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"full-body SUPR model is missing fields: {missing}")
    return payload


def _ordered_boundary_loops(faces: np.ndarray) -> tuple[np.ndarray, ...]:
    directed = np.concatenate(
        (faces[:, (0, 1)], faces[:, (1, 2)], faces[:, (2, 0)]), axis=0
    )
    undirected = np.sort(directed, axis=1)
    _, inverse, counts = np.unique(
        undirected, axis=0, return_inverse=True, return_counts=True
    )
    if np.any(counts > 2):
        raise ValueError("lower-leg surface contains a non-manifold edge")
    boundary = directed[counts[inverse] == 1]
    following: dict[int, int] = {}
    incoming: dict[int, int] = {}
    for first, second in boundary:
        first_int, second_int = int(first), int(second)
        if first_int in following or second_int in incoming:
            raise ValueError("lower-leg boundary is not consistently wound")
        following[first_int] = second_int
        incoming[second_int] = first_int
    if set(following) != set(incoming):
        raise ValueError("lower-leg boundary contains an open chain")

    remaining = set(following)
    loops: list[np.ndarray] = []
    while remaining:
        start = min(remaining)
        current = start
        ordered: list[int] = []
        while current not in ordered:
            ordered.append(current)
            remaining.discard(current)
            current = following[current]
        if current != start:
            raise ValueError("lower-leg boundary contains a self-intersection")
        loops.append(np.asarray(ordered, dtype=np.int64))
    return tuple(loops)


def _validate_connected_surface(vertex_count: int, faces: np.ndarray) -> None:
    used = np.unique(faces)
    if len(used) != vertex_count:
        raise ValueError("lower-leg extraction contains unused vertices")
    adjacency: list[list[int]] = [[] for _ in range(vertex_count)]
    for first, second, third in faces:
        adjacency[int(first)].extend((int(second), int(third)))
        adjacency[int(second)].extend((int(first), int(third)))
        adjacency[int(third)].extend((int(first), int(second)))
    visited = {0}
    pending = [0]
    while pending:
        current = pending.pop()
        for neighbour in adjacency[current]:
            if neighbour not in visited:
                visited.add(neighbour)
                pending.append(neighbour)
    if len(visited) != vertex_count:
        raise ValueError("right lower-leg extraction contains disconnected surfaces")


def _transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    homogeneous = np.column_stack((values, np.ones(len(values))))
    transformed = homogeneous @ transform.T
    return transformed[:, :3] / transformed[:, 3, None]


def _joint_frame(knee: np.ndarray, ankle: np.ndarray, foot: np.ndarray) -> np.ndarray:
    up = np.asarray(knee, dtype=np.float64) - np.asarray(ankle, dtype=np.float64)
    up_norm = float(np.linalg.norm(up))
    if up_norm <= np.finfo(np.float64).eps:
        raise ValueError("knee and ankle joints occupy the same position")
    up /= up_norm
    forward = np.asarray(foot, dtype=np.float64) - np.asarray(ankle, dtype=np.float64)
    forward -= up * float(np.dot(forward, up))
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm <= np.finfo(np.float64).eps:
        raise ValueError("foot direction is parallel to the lower-leg axis")
    forward /= forward_norm
    lateral = np.cross(forward, up)
    lateral /= np.linalg.norm(lateral)
    forward = np.cross(up, lateral)
    forward /= np.linalg.norm(forward)
    return np.column_stack((forward, up, lateral))


def _translation_matrix(offset: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, 3] = np.asarray(offset, dtype=np.float64)
    return result


def _rigid_correspondence_transform(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, float]:
    source_points = np.asarray(source, dtype=np.float64)
    target_points = np.asarray(target, dtype=np.float64)
    if (
        source_points.shape != target_points.shape
        or source_points.ndim != 2
        or source_points.shape[1:] != (3,)
        or len(source_points) < 3
        or not np.isfinite(source_points).all()
        or not np.isfinite(target_points).all()
    ):
        raise ValueError("ankle correspondence must contain matching finite points")
    source_center = source_points.mean(axis=0)
    target_center = target_points.mean(axis=0)
    covariance = (source_points - source_center).T @ (
        target_points - target_center
    )
    left, _, right_transposed = np.linalg.svd(covariance)
    rotation = right_transposed.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_transposed[-1] *= -1.0
        rotation = right_transposed.T @ left.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = target_center - rotation @ source_center
    fitted = _transform_points(source_points, transform)
    residual = float(np.sqrt(np.mean(np.sum((fitted - target_points) ** 2, axis=1))))
    return transform, residual


def attach_lower_leg_to_fitted_dense_foot(
    fitted_dense_foot: TriangleMesh,
    canonical_dense_foot_vertices: np.ndarray,
    canonical_lower_leg_vertices: np.ndarray,
    canonical_lower_leg_faces: np.ndarray,
    dense_foot_ankle_loop: np.ndarray,
    ankle_loop_correspondence: np.ndarray,
) -> FittedFootLowerLeg:
    """Rigidly place the canonical shank at one fitted foot's ankle loop."""

    reference_vertices = np.asarray(canonical_dense_foot_vertices, dtype=np.float64)
    leg_vertices = np.asarray(canonical_lower_leg_vertices, dtype=np.float64)
    leg_faces = np.asarray(canonical_lower_leg_faces, dtype=np.int64)
    foot_loop = np.asarray(dense_foot_ankle_loop, dtype=np.int64)
    correspondence = np.asarray(ankle_loop_correspondence, dtype=np.int64)
    if reference_vertices.shape != fitted_dense_foot.vertices.shape:
        raise ValueError("canonical and fitted dense feet have different vertex counts")
    if foot_loop.ndim != 1 or len(foot_loop) != 60:
        raise ValueError("dense foot ankle loop must contain 60 vertex indices")
    if correspondence.shape != (len(foot_loop), 2) or not np.array_equal(
        correspondence[:, 0], foot_loop
    ):
        raise ValueError("stored ankle correspondence does not match the foot loop")
    if (
        leg_faces.ndim != 2
        or leg_faces.shape[1:] != (3,)
        or np.any(leg_faces < 0)
        or np.any(leg_faces >= len(leg_vertices))
    ):
        raise ValueError("canonical lower-leg topology is invalid")
    paired_leg = correspondence[:, 1]
    if np.any(paired_leg < 0) or np.any(paired_leg >= len(leg_vertices)):
        raise ValueError("ankle correspondence contains invalid lower-leg indices")

    transform, residual = _rigid_correspondence_transform(
        reference_vertices[foot_loop], fitted_dense_foot.vertices[foot_loop]
    )
    fitted_leg_vertices = _transform_points(leg_vertices, transform)
    leg_offset = len(fitted_dense_foot.vertices)
    fitted_leg_faces = leg_faces + leg_offset
    paired_global = paired_leg + leg_offset
    bridge_faces: list[tuple[int, int, int]] = []
    for index in range(len(foot_loop)):
        following = (index + 1) % len(foot_loop)
        bridge_faces.append(
            (
                int(foot_loop[index]),
                int(paired_global[index]),
                int(foot_loop[following]),
            )
        )
        bridge_faces.append(
            (
                int(foot_loop[following]),
                int(paired_global[index]),
                int(paired_global[following]),
            )
        )
    bridge = np.asarray(bridge_faces, dtype=np.int64)
    vertices = np.vstack((fitted_dense_foot.vertices, fitted_leg_vertices))
    faces = np.vstack((fitted_dense_foot.faces, fitted_leg_faces, bridge))
    if not np.array_equal(vertices[: len(fitted_dense_foot.vertices)], fitted_dense_foot.vertices):
        raise RuntimeError("lower-leg attachment changed a fitted-foot vertex")
    if not np.array_equal(faces[: len(fitted_dense_foot.faces)], fitted_dense_foot.faces):
        raise RuntimeError("lower-leg attachment changed a fitted-foot face")
    _validate_connected_surface(len(vertices), faces)
    loops = _ordered_boundary_loops(faces)
    if len(loops) != 1 or len(loops[0]) != 68:
        raise ValueError("fitted foot and lower leg must leave one 68-vertex knee opening")

    lower_leg_face_indices = np.arange(
        len(fitted_dense_foot.faces),
        len(fitted_dense_foot.faces) + len(fitted_leg_faces),
        dtype=np.int64,
    )
    bridge_face_indices = np.arange(
        len(fitted_dense_foot.faces) + len(fitted_leg_faces),
        len(faces),
        dtype=np.int64,
    )
    return FittedFootLowerLeg(
        mesh=TriangleMesh(vertices, faces),
        foot_vertex_count=len(fitted_dense_foot.vertices),
        foot_face_count=len(fitted_dense_foot.faces),
        lower_leg_vertex_indices=np.arange(leg_offset, len(vertices), dtype=np.int64),
        lower_leg_face_indices=lower_leg_face_indices,
        bridge_face_indices=bridge_face_indices,
        canonical_leg_to_fitted_ankle=transform,
        ankle_loop_rms_residual=residual,
    )


def build_canonical_right_lower_leg(
    model_path: str | Path,
    raw_supr_to_reference: np.ndarray,
    foot_reference_joints: np.ndarray,
    foot_ankle_loop_points: np.ndarray,
) -> SuprLowerLeg:
    """Align and cut the neutral male SUPR donor into one canonical shank."""

    model = _load_full_body_model(model_path)
    vertices = np.asarray(model["v_template"], dtype=np.float64)
    faces = np.asarray(model["f"], dtype=np.int64)
    weights = np.asarray(model["weights"], dtype=np.float64)
    joints = np.asarray(model["J"], dtype=np.float64)
    tree = np.asarray(model["kintree_table"], dtype=np.int64)
    transform = np.asarray(raw_supr_to_reference, dtype=np.float64)
    target_joints = np.asarray(foot_reference_joints, dtype=np.float64)
    ankle_points = np.asarray(foot_ankle_loop_points, dtype=np.float64)

    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not np.isfinite(vertices).all():
        raise ValueError("full-body SUPR vertices must have finite shape (V, 3)")
    if faces.ndim != 2 or faces.shape[1:] != (3,) or np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError("full-body SUPR faces are invalid")
    if weights.ndim != 2 or weights.shape[0] != len(vertices) or weights.shape[1] <= max(RIGHT_TOE_JOINT_INDICES):
        raise ValueError("full-body SUPR skinning weights are invalid")
    if not np.isfinite(weights).all() or not np.allclose(weights.sum(axis=1), 1.0, atol=1.0e-8, rtol=0.0):
        raise ValueError("full-body SUPR skinning weights are invalid")
    if joints.shape != (weights.shape[1], 3) or not np.isfinite(joints).all():
        raise ValueError("full-body SUPR joints do not match its skinning weights")
    if tree.shape != (2, weights.shape[1]):
        raise ValueError("full-body SUPR kinematic tree is invalid")
    if int(tree[0, RIGHT_ANKLE_JOINT_INDEX]) != RIGHT_KNEE_JOINT_INDEX:
        raise ValueError("full-body SUPR right knee/ankle hierarchy is unexpected")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("raw_supr_to_reference must be a finite 4x4 matrix")
    if target_joints.shape[0] <= FOOT_MIDFOOT_JOINT_INDEX or target_joints.shape[1:] != (3,):
        raise ValueError("foot reference joints are invalid")
    if ankle_points.shape != (EXPECTED_NATIVE_ANKLE_COUNT, 3) or not np.isfinite(ankle_points).all():
        raise ValueError("foot ankle loop must contain the ordered 15 native points")

    canonical_body_joints = _transform_points(joints, transform)
    source_frame = _joint_frame(
        canonical_body_joints[RIGHT_KNEE_JOINT_INDEX],
        canonical_body_joints[RIGHT_ANKLE_JOINT_INDEX],
        canonical_body_joints[RIGHT_FOOT_JOINT_INDEX],
    )
    target_frame = _joint_frame(
        target_joints[FOOT_ROOT_JOINT_INDEX],
        target_joints[FOOT_ANKLE_JOINT_INDEX],
        target_joints[FOOT_MIDFOOT_JOINT_INDEX],
    )
    rotation = target_frame @ source_frame.T
    rigid = np.eye(4, dtype=np.float64)
    rigid[:3, :3] = rotation
    rigid[:3, 3] = target_joints[FOOT_ANKLE_JOINT_INDEX] - rotation @ (
        canonical_body_joints[RIGHT_ANKLE_JOINT_INDEX]
    )

    body_to_reference = rigid @ transform
    aligned_vertices = _transform_points(vertices, body_to_reference)
    aligned_joints = _transform_points(joints, body_to_reference)

    dominant = np.argmax(weights, axis=1)
    limb_joints = (
        RIGHT_KNEE_JOINT_INDEX,
        RIGHT_ANKLE_JOINT_INDEX,
        RIGHT_FOOT_JOINT_INDEX,
        *RIGHT_TOE_JOINT_INDICES,
    )
    limb_vertices = np.isin(dominant, limb_joints)
    limb_faces = np.all(limb_vertices[faces], axis=1)

    plane_origin = ankle_points.mean(axis=0)
    plane_normal = target_frame[:, 1]
    signed_height = (aligned_vertices - plane_origin) @ plane_normal
    retained = limb_faces & np.all(signed_height[faces] >= -1.0e-10, axis=1)
    source_face_indices = np.flatnonzero(retained).astype(np.int64)
    if len(source_face_indices) == 0:
        raise ValueError("anatomical cut produced no right lower-leg faces")
    source_faces = faces[source_face_indices]
    source_vertex_indices = np.unique(source_faces).astype(np.int64)
    source_to_local = np.full(len(vertices), -1, dtype=np.int64)
    source_to_local[source_vertex_indices] = np.arange(len(source_vertex_indices))
    local_faces = source_to_local[source_faces]
    mesh = TriangleMesh(aligned_vertices[source_vertex_indices], local_faces)
    _validate_connected_surface(len(mesh.vertices), mesh.faces)

    loops = _ordered_boundary_loops(mesh.faces)
    loop_sizes = sorted(len(loop) for loop in loops)
    if len(loops) != 2 or loop_sizes != [EXPECTED_NATIVE_ANKLE_COUNT, EXPECTED_NATIVE_KNEE_COUNT]:
        raise ValueError(
            "right shank must have 15-vertex ankle and 17-vertex knee openings; "
            f"found {loop_sizes}"
        )
    distal = next(loop for loop in loops if len(loop) == EXPECTED_NATIVE_ANKLE_COUNT)
    proximal = next(loop for loop in loops if len(loop) == EXPECTED_NATIVE_KNEE_COUNT)

    centroid_delta = plane_origin - mesh.vertices[distal].mean(axis=0)
    in_plane_translation = centroid_delta - plane_normal * float(
        np.dot(centroid_delta, plane_normal)
    )
    final_transform = _translation_matrix(in_plane_translation) @ body_to_reference
    mesh = transform_mesh(mesh, _translation_matrix(in_plane_translation))
    aligned_joints = _transform_points(joints, final_transform)
    remaining = mesh.vertices[distal].mean(axis=0) - plane_origin
    residual = remaining - plane_normal * float(np.dot(remaining, plane_normal))

    rotation_cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return SuprLowerLeg(
        mesh=mesh,
        source_vertex_indices=source_vertex_indices,
        source_face_indices=source_face_indices,
        proximal_boundary_vertex_indices=proximal,
        distal_boundary_vertex_indices=distal,
        source_joints_reference=aligned_joints,
        body_to_reference=final_transform,
        frame_rotation_degrees=float(np.degrees(np.arccos(rotation_cosine))),
        in_plane_translation=in_plane_translation,
        distal_centroid_residual=float(np.linalg.norm(residual)),
    )


def load_posable_supr_lower_leg(
    model_path: str | Path,
    neutral_lower_leg: SuprLowerLeg,
    subdivision: SuprMeshSubdivision,
    num_betas: int = 10,
) -> PosableSuprLowerLegModel:
    """Load full-body SUPR for ankle-relative lower-leg pose and shape changes."""

    source = Path(model_path).expanduser().resolve(strict=True)
    if source.suffix.lower() != ".npy":
        raise ValueError("full-body SUPR model must be a .npy file")
    if int(num_betas) != num_betas or not 1 <= int(num_betas) <= 400:
        raise ValueError("num_betas must be an integer in [1, 400]")
    if subdivision.source_vertex_count != len(neutral_lower_leg.mesh.vertices):
        raise ValueError("lower-leg subdivision does not match the neutral shank")
    try:
        import torch
    except ImportError as error:
        raise RuntimeError(
            "lower-leg fitting requires PyTorch; install the fitting extra"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError(
            "lower-leg fitting requires CUDA because the official SUPR "
            "implementation constructs CUDA buffers"
        )
    try:
        from supr.pytorch.supr import SUPR
    except ImportError as error:
        raise RuntimeError(
            "the official SUPR package is unavailable; install ../baselines/SUPR "
            "in editable mode"
        ) from error

    payload = dict(_load_full_body_model(source))
    shapedirs = payload["shapedirs"]
    if not isinstance(shapedirs, np.ndarray):
        shapedirs = getattr(shapedirs, "__dict__", {}).get("x")
    shapedirs = np.asarray(shapedirs, dtype=np.float64)
    if shapedirs.shape[:2] != (FULL_BODY_VERTEX_COUNT, 3):
        raise ValueError("full-body SUPR shapedirs have unexpected dimensions")
    payload["shapedirs"] = shapedirs
    with tempfile.TemporaryDirectory(
        prefix=".supr-full-body-", dir=source.parent
    ) as temporary:
        plain_path = Path(temporary) / "model.npy"
        np.save(plain_path, payload, allow_pickle=True)
        model = SUPR(str(plain_path), num_betas=int(num_betas)).cuda().eval()
    if (
        model.num_verts != FULL_BODY_VERTEX_COUNT
        or len(model.f) != FULL_BODY_FACE_COUNT
        or model.num_joints != FULL_BODY_JOINT_COUNT
        or model.num_pose != FULL_BODY_POSE_PARAMETER_COUNT
    ):
        raise ValueError("full-body SUPR topology or parameter layout is unexpected")

    weights = np.asarray(payload["weights"], dtype=np.float64)
    dominant = np.argmax(weights, axis=1)
    donor_foot_indices = np.flatnonzero(
        np.isin(
            dominant,
            (RIGHT_FOOT_JOINT_INDEX, *RIGHT_TOE_JOINT_INDICES),
        )
    ).astype(np.int64)
    if len(donor_foot_indices) < 3:
        raise ValueError("full-body SUPR contains no usable right-foot anchor")
    neutral_reference_vertices = _transform_points(
        np.asarray(payload["v_template"], dtype=np.float64),
        neutral_lower_leg.body_to_reference,
    )
    padded_template = np.concatenate(
        (np.asarray(payload["v_template"], dtype=np.float64).reshape(-1), [1.0])
    )
    neutral_raw_joints = (
        np.asarray(payload["J_regressor"], dtype=np.float64) @ padded_template
    ).reshape(FULL_BODY_JOINT_COUNT, 3)
    neutral_reference_joints = _transform_points(
        neutral_raw_joints, neutral_lower_leg.body_to_reference
    )
    return PosableSuprLowerLegModel(
        model=model,
        torch_module=torch,
        num_betas=int(num_betas),
        neutral_lower_leg=neutral_lower_leg,
        subdivision=subdivision,
        donor_foot_vertex_indices=donor_foot_indices,
        neutral_reference_vertices=neutral_reference_vertices,
        neutral_reference_joints=neutral_reference_joints,
    )
