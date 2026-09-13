#!/usr/bin/env python3
"""Build validated Checkpoint 11-B2/B3 instance anatomical volumes."""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any
import zipfile

import numpy as np

from foot_prior.anatomical_volume import (
    InstanceVolumeContinuation,
    InstanceVolumeProblem,
    _build_instance_deformation_system,
    _load_extended_reference,
    _validate_continuation_candidate,
    continue_instance_volume,
    instance_continuation_configuration,
    load_canonical_anatomical_volume,
    load_instance_volume_problem,
)
from foot_prior.anatomy import array_digest
from foot_prior.instance_volume_optimization import (
    B3_EXACT_CORRECTION_RESOLUTIONS,
    InstanceVolumeOptimization,
    _final_validation,
    build_instance_optimization_system,
    optimization_configuration,
    optimize_instance_volume,
)


CONTINUATION_ARTIFACT_NAMES = (
    "continuation_state.json",
    "continuation_state.npz",
)
FINAL_ARTIFACT_NAMES = (
    "instance_volume.json",
    "instance_volume.npz",
    "instance_volume.vtk",
)


@dataclass(frozen=True)
class _ResumeState:
    continuation: InstanceVolumeContinuation | None
    final_status: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Checkpoint 11-B1 inputs, build the safe Checkpoint 11-B2 "
            "continuation, and by default finish Checkpoint 11-B3."
        )
    )
    parser.add_argument("--anatomical-volume-root", required=True, type=Path)
    parser.add_argument(
        "--extended-anatomical-surface-root", required=True, type=Path
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--stop-after",
        choices=("11-b2", "11-b3"),
        default="11-b3",
        help="Stop after the intermediate continuation or finish B3.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="SHOE",
        help="Exclude one shoe name; may be supplied more than once.",
    )
    replacement = parser.add_mutually_exclusive_group()
    replacement.add_argument("--overwrite", action="store_true")
    replacement.add_argument(
        "--resume",
        action="store_true",
        help="Reuse only complete, matching, geometrically valid saved states.",
    )
    parser.add_argument("shoes", nargs="*")
    return parser.parse_args()


def _write_deterministic_npz(path: Path, **arrays: np.ndarray) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
            member = zipfile.ZipInfo(
                f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
            )
            member.compress_type = zipfile.ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue())


def _validated_names(
    volume_root: Path,
    requested: list[str],
    excluded: list[str],
) -> list[str]:
    names = requested or [
        path.name
        for path in volume_root.iterdir()
        if path.is_dir()
        and (path / "boundary_target.json").is_file()
        and (path / "boundary_target.npz").is_file()
    ]
    for name in [*names, *excluded]:
        if not name or Path(name).name != name or name in {".", "..", "reference"}:
            raise ValueError(f"invalid shoe directory name: {name!r}")
    if len(set(names)) != len(names) or len(set(excluded)) != len(excluded):
        raise ValueError("shoe and exclusion names must be unique")
    selected = sorted(set(names).difference(excluded))
    if not selected:
        raise ValueError("no instance boundary targets were selected")
    if "sneaker_vibe" in selected:
        raise ValueError("sneaker_vibe is excluded from the accepted B3 scope")
    return selected


def _read_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"saved state is not a JSON object: {path}")
    return payload


def _load_resumable_continuation(
    directory: Path,
    problem: InstanceVolumeProblem,
) -> InstanceVolumeContinuation:
    json_path = directory / "continuation_state.json"
    npz_path = directory / "continuation_state.npz"
    payload = _read_json_object(json_path)
    if (
        payload.get("schema_version") != 1
        or payload.get("stage") != "instance_volume_continuation"
        or payload.get("shoe_name") != problem.shoe_name
        or payload.get("status")
        not in {"baseline_reached_target", "needs_11_b3"}
        or payload.get("configuration") != instance_continuation_configuration()
        or payload.get("source_boundary_target_status")
        != problem.boundary_target.status
    ):
        raise ValueError(
            f"{problem.shoe_name}: saved B2 state is incompatible; use --overwrite"
        )
    digests = payload.get("digests", {})
    if (
        digests.get("canonical_volume_topology_sha256")
        != problem.canonical_volume.topology_digest
        or digests.get("boundary_target_geometry_sha256")
        != problem.boundary_target.geometry_digest
        or digests.get("fitted_extended_surface_sha256")
        != problem.boundary_target.fitted_surface_digest
    ):
        raise ValueError(
            f"{problem.shoe_name}: saved B2 input digests mismatch; use --overwrite"
        )
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != {"last_valid_volume_vertices"}:
            raise ValueError(
                f"{problem.shoe_name}: saved B2 NPZ fields are invalid; use --overwrite"
            )
        vertices = np.asarray(
            archive["last_valid_volume_vertices"], dtype=np.float64
        )
    volume = problem.canonical_volume
    if (
        vertices.shape != volume.volume_vertices.shape
        or not np.isfinite(vertices).all()
        or digests.get("continuation_vertices_sha256")
        != array_digest(vertices)
    ):
        raise ValueError(
            f"{problem.shoe_name}: saved B2 vertices are invalid; use --overwrite"
        )
    validation, determinants = _validate_continuation_candidate(volume, vertices)
    if not validation.accepted:
        raise ValueError(
            f"{problem.shoe_name}: saved B2 geometry is invalid; use --overwrite"
        )
    try:
        reached_alpha = float(payload["reached_alpha"])
        accepted_alphas = np.asarray(payload["accepted_alphas"], dtype=np.float64)
        history = tuple(payload["history"])
        counts = payload["counts"]
        final = dict(payload["final"])
        stopping_reason = str(payload["stopping_reason"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"{problem.shoe_name}: saved B2 metadata is malformed; use --overwrite"
        ) from error
    status = str(payload["status"])
    if (
        not 0.0 <= reached_alpha <= 1.0
        or accepted_alphas.ndim != 1
        or not len(accepted_alphas)
        or not np.isfinite(accepted_alphas).all()
        or accepted_alphas[0] != 0.0
        or accepted_alphas[-1] != reached_alpha
        or np.any(np.diff(accepted_alphas) <= 0.0)
        or (status == "baseline_reached_target") != (reached_alpha == 1.0)
        or counts.get("volume_vertices") != len(vertices)
        or counts.get("attempts") != len(history)
        or counts.get("accepted_steps") != len(accepted_alphas) - 1
        or not isinstance(final.get("jacobian_determinant"), dict)
    ):
        raise ValueError(
            f"{problem.shoe_name}: saved B2 progress is inconsistent; use --overwrite"
        )
    final.pop("jacobian_determinant")
    return InstanceVolumeContinuation(
        shoe_name=problem.shoe_name,
        volume_vertices=vertices,
        reached_alpha=reached_alpha,
        status=status,
        stopping_reason=stopping_reason,
        jacobian_determinants=determinants,
        accepted_alphas=accepted_alphas,
        attempts=history,
        canonical_volume_topology_digest=volume.topology_digest,
        boundary_target_geometry_digest=problem.boundary_target.geometry_digest,
        diagnostics=final,
    )


def _validate_resumable_final(
    directory: Path,
    problem: InstanceVolumeProblem,
    continuation: InstanceVolumeContinuation,
    optimization_system: Any,
) -> str | None:
    json_path = directory / "instance_volume.json"
    npz_path = directory / "instance_volume.npz"
    vtk_path = directory / "instance_volume.vtk"
    if not json_path.exists():
        if npz_path.exists() or vtk_path.exists():
            raise ValueError(
                f"{problem.shoe_name}: partial B3 artifacts exist; use --overwrite"
            )
        return None
    payload = _read_json_object(json_path)
    status = payload.get("status")
    if (
        payload.get("schema_version") != 2
        or payload.get("stage") != "instance_volume_optimization"
        or payload.get("shoe_name") != problem.shoe_name
        or status
        not in {"final_exact_target", "final_corrected_target", "failed_11_b3"}
        or payload.get("configuration") != optimization_configuration()
    ):
        raise ValueError(
            f"{problem.shoe_name}: saved B3 state is incompatible; use --overwrite"
        )
    digests = payload.get("digests", {})
    if (
        digests.get("canonical_volume_topology_sha256")
        != problem.canonical_volume.topology_digest
        or digests.get("boundary_target_geometry_sha256")
        != problem.boundary_target.geometry_digest
        or digests.get("fitted_extended_surface_sha256")
        != problem.boundary_target.fitted_surface_digest
        or digests.get("continuation_vertices_sha256")
        != array_digest(continuation.volume_vertices)
    ):
        raise ValueError(
            f"{problem.shoe_name}: saved B3 input digests mismatch; use --overwrite"
        )
    if status == "failed_11_b3":
        if npz_path.exists() or vtk_path.exists():
            raise ValueError(
                f"{problem.shoe_name}: failed B3 state has stale finals; use --overwrite"
            )
        return None
    if not npz_path.is_file() or not vtk_path.is_file() or vtk_path.stat().st_size == 0:
        raise ValueError(
            f"{problem.shoe_name}: completed B3 artifacts are partial; use --overwrite"
        )
    expected_fields = {
        "volume_vertices",
        "tetrahedra",
        "harmonic_r",
        "jacobian_determinants",
        "jacobian_singular_values",
        "condition_numbers",
        "target_correction_vectors",
    }
    with np.load(npz_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_fields:
            raise ValueError(
                f"{problem.shoe_name}: saved B3 NPZ fields are invalid; use --overwrite"
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    volume = problem.canonical_volume
    vertices = np.asarray(arrays["volume_vertices"], dtype=np.float64)
    tetrahedra = np.asarray(arrays["tetrahedra"], dtype=np.int64)
    harmonic_r = np.asarray(arrays["harmonic_r"], dtype=np.float64)
    determinants = np.asarray(arrays["jacobian_determinants"], dtype=np.float64)
    singular_values = np.asarray(
        arrays["jacobian_singular_values"], dtype=np.float64
    )
    condition_numbers = np.asarray(arrays["condition_numbers"], dtype=np.float64)
    corrections = np.asarray(arrays["target_correction_vectors"], dtype=np.float64)
    if (
        vertices.shape != volume.volume_vertices.shape
        or determinants.shape != (len(volume.tetrahedra),)
        or singular_values.shape != (len(volume.tetrahedra), 3)
        or condition_numbers.shape != (len(volume.tetrahedra),)
        or corrections.shape
        != (len(volume.computational_inner_vertex_indices), 3)
        or not all(
            np.isfinite(array).all()
            for array in (
                vertices,
                harmonic_r,
                determinants,
                singular_values,
                condition_numbers,
                corrections,
            )
        )
        or not np.array_equal(tetrahedra, volume.tetrahedra)
        or not np.array_equal(harmonic_r, volume.harmonic_r)
        or digests.get("instance_volume_vertices_sha256")
        != array_digest(vertices)
    ):
        raise ValueError(
            f"{problem.shoe_name}: saved B3 arrays are invalid; use --overwrite"
        )
    validation = _final_validation(problem, optimization_system, vertices, beta=1.0)
    expected_corrections = (
        vertices[volume.computational_inner_vertex_indices]
        - problem.boundary_target.vertices
    )
    expected_status = (
        "final_exact_target"
        if float(np.max(np.linalg.norm(expected_corrections, axis=1)))
        <= B3_EXACT_CORRECTION_RESOLUTIONS
        * problem.boundary_target.surface_resolution
        else "final_corrected_target"
    )
    if (
        not validation.accepted
        or status != expected_status
        or payload.get("reached_beta") != 1.0
        or not np.array_equal(corrections, expected_corrections)
        or not np.allclose(
            determinants, validation.quality.determinants, atol=1.0e-12, rtol=1.0e-12
        )
        or not np.allclose(
            singular_values,
            validation.quality.singular_values,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
        or not np.allclose(
            condition_numbers,
            validation.quality.condition_numbers,
            atol=1.0e-12,
            rtol=1.0e-12,
        )
    ):
        raise ValueError(
            f"{problem.shoe_name}: saved B3 geometry failed validation; use --overwrite"
        )
    return str(status)


def _preflight_resume_state(
    directory: Path,
    problem: InstanceVolumeProblem,
    optimization_system: Any | None,
) -> _ResumeState:
    continuation_paths = [directory / name for name in CONTINUATION_ARTIFACT_NAMES]
    continuation_present = [path.exists() for path in continuation_paths]
    final_paths = [directory / name for name in FINAL_ARTIFACT_NAMES]
    if any(continuation_present) and not all(continuation_present):
        raise ValueError(
            f"{problem.shoe_name}: partial B2 artifacts exist; use --overwrite"
        )
    if not any(continuation_present):
        if optimization_system is not None and any(path.exists() for path in final_paths):
            raise ValueError(
                f"{problem.shoe_name}: B3 artifacts exist without B2; use --overwrite"
            )
        return _ResumeState(None, None)
    continuation = _load_resumable_continuation(directory, problem)
    final_status = (
        _validate_resumable_final(
            directory, problem, continuation, optimization_system
        )
        if optimization_system is not None
        else None
    )
    return _ResumeState(continuation, final_status)


def _continuation_payload(
    problem: InstanceVolumeProblem,
    result: InstanceVolumeContinuation,
    volume_root: Path,
    surface_root: Path,
    directory: Path,
) -> dict[str, Any]:
    payload = result.to_dict()
    payload["source_boundary_target_status"] = problem.boundary_target.status
    payload["inputs"] = {
        "canonical_volume": str(volume_root / "reference"),
        "boundary_target": str(volume_root / problem.shoe_name),
        "fitted_surface": str(
            surface_root / problem.shoe_name / "foot_lower_leg.ply"
        ),
    }
    payload["digests"]["fitted_extended_surface_sha256"] = (
        problem.boundary_target.fitted_surface_digest
    )
    payload["digests"]["continuation_vertices_sha256"] = array_digest(
        result.volume_vertices
    )
    payload["artifacts"] = {
        artifact: str(directory / artifact)
        for artifact in CONTINUATION_ARTIFACT_NAMES
    }
    return payload


def _final_payload(
    problem: InstanceVolumeProblem,
    continuation: InstanceVolumeContinuation,
    result: InstanceVolumeOptimization,
    volume_root: Path,
    surface_root: Path,
    directory: Path,
) -> dict[str, Any]:
    payload = result.to_dict()
    payload["inputs"] = {
        "canonical_volume": str(volume_root / "reference"),
        "boundary_target": str(volume_root / problem.shoe_name),
        "fitted_surface": str(
            surface_root / problem.shoe_name / "foot_lower_leg.ply"
        ),
        "continuation_state": str(directory / "continuation_state.npz"),
    }
    payload["digests"].update(
        {
            "fitted_extended_surface_sha256": (
                problem.boundary_target.fitted_surface_digest
            ),
            "continuation_vertices_sha256": array_digest(
                continuation.volume_vertices
            ),
            "instance_volume_vertices_sha256": array_digest(
                result.volume_vertices
            ),
        }
    )
    payload["artifacts"] = {
        "instance_volume.json": str(directory / "instance_volume.json")
    }
    if result.status != "failed_11_b3":
        payload["artifacts"].update(
            {
                artifact: str(directory / artifact)
                for artifact in FINAL_ARTIFACT_NAMES[1:]
            }
        )
    return payload


def _write_instance_vtk(
    path: Path,
    problem: InstanceVolumeProblem,
    result: InstanceVolumeOptimization,
) -> None:
    volume = problem.canonical_volume
    correction = np.zeros(len(result.volume_vertices), dtype=np.float64)
    correction[volume.computational_inner_vertex_indices] = np.linalg.norm(
        result.target_correction_vectors, axis=1
    )
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("# vtk DataFile Version 3.0\n")
        stream.write(f"Checkpoint 11-B3 instance volume: {problem.shoe_name}\n")
        stream.write("ASCII\nDATASET UNSTRUCTURED_GRID\n")
        stream.write(f"POINTS {len(result.volume_vertices)} double\n")
        np.savetxt(stream, result.volume_vertices, fmt="%.17g")
        stream.write(
            f"CELLS {len(volume.tetrahedra)} {5 * len(volume.tetrahedra)}\n"
        )
        cells = np.column_stack(
            (np.full(len(volume.tetrahedra), 4, dtype=np.int64), volume.tetrahedra)
        )
        np.savetxt(stream, cells, fmt="%d")
        stream.write(f"CELL_TYPES {len(volume.tetrahedra)}\n")
        np.savetxt(
            stream,
            np.full(len(volume.tetrahedra), 10, dtype=np.int64),
            fmt="%d",
        )
        stream.write(f"POINT_DATA {len(result.volume_vertices)}\n")
        stream.write("SCALARS harmonic_r double 1\nLOOKUP_TABLE default\n")
        np.savetxt(stream, volume.harmonic_r, fmt="%.17g")
        stream.write(
            "SCALARS target_correction_magnitude double 1\nLOOKUP_TABLE default\n"
        )
        np.savetxt(stream, correction, fmt="%.17g")
        stream.write(f"CELL_DATA {len(volume.tetrahedra)}\n")
        cell_scalars = (
            ("jacobian_determinant", result.jacobian_determinants),
            ("minimum_singular_value", result.jacobian_singular_values[:, 2]),
            ("maximum_singular_value", result.jacobian_singular_values[:, 0]),
            ("condition_number", result.condition_numbers),
        )
        for name, values in cell_scalars:
            stream.write(f"SCALARS {name} double 1\nLOOKUP_TABLE default\n")
            np.savetxt(stream, values, fmt="%.17g")


def _write_state(
    directory: Path,
    continuation_payload: dict[str, Any],
    continuation_vertices: np.ndarray,
    *,
    problem: InstanceVolumeProblem,
    final_payload: dict[str, Any] | None,
    final_result: InstanceVolumeOptimization | None,
    overwrite: bool,
) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent)
    )
    try:
        (staging / "continuation_state.json").write_text(
            json.dumps(continuation_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_deterministic_npz(
            staging / "continuation_state.npz",
            last_valid_volume_vertices=continuation_vertices,
        )
        artifacts = list(CONTINUATION_ARTIFACT_NAMES)
        if final_payload is not None:
            (staging / "instance_volume.json").write_text(
                json.dumps(final_payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            artifacts.append("instance_volume.json")
        if final_result is not None and final_result.status != "failed_11_b3":
            volume = problem.canonical_volume
            _write_deterministic_npz(
                staging / "instance_volume.npz",
                volume_vertices=final_result.volume_vertices,
                tetrahedra=volume.tetrahedra,
                harmonic_r=volume.harmonic_r,
                jacobian_determinants=final_result.jacobian_determinants,
                jacobian_singular_values=final_result.jacobian_singular_values,
                condition_numbers=final_result.condition_numbers,
                target_correction_vectors=final_result.target_correction_vectors,
            )
            _write_instance_vtk(
                staging / "instance_volume.vtk", problem, final_result
            )
            artifacts.extend(FINAL_ARTIFACT_NAMES[1:])
        directory.mkdir(parents=True, exist_ok=True)
        for artifact in artifacts:
            os.replace(staging / artifact, directory / artifact)
        if (
            overwrite
            and final_result is not None
            and final_result.status == "failed_11_b3"
        ):
            for artifact in FINAL_ARTIFACT_NAMES[1:]:
                stale = directory / artifact
                if stale.exists():
                    stale.unlink()
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    volume_root = args.anatomical_volume_root.expanduser().resolve(strict=True)
    surface_root = (
        args.extended_anatomical_surface_root.expanduser().resolve(strict=True)
    )
    output_root = args.output_root.expanduser().resolve()
    if not volume_root.is_dir() or not surface_root.is_dir():
        raise NotADirectoryError(
            "anatomical volume and surface roots must be directories"
        )
    names = _validated_names(volume_root, list(args.shoes), list(args.exclude))
    selected_artifacts = list(CONTINUATION_ARTIFACT_NAMES)
    if args.stop_after == "11-b3":
        selected_artifacts.extend(FINAL_ARTIFACT_NAMES)
    existing = [
        output_root / name / artifact
        for name in names
        for artifact in selected_artifacts
        if (output_root / name / artifact).exists()
    ]
    if existing and not args.overwrite and not args.resume:
        raise FileExistsError(
            "instance-volume artifacts already exist; pass --resume or "
            f"--overwrite: {existing[0]}"
        )

    started = time.perf_counter()
    canonical_volume = load_canonical_anatomical_volume(volume_root)
    extended_reference = _load_extended_reference(surface_root)
    problems = []
    for name in names:
        shoe_started = time.perf_counter()
        problems.append(
            load_instance_volume_problem(
                volume_root,
                surface_root,
                name,
                canonical_volume=canonical_volume,
                extended_reference=extended_reference,
            )
        )
        print(
            f"[11-B1] {name}: validated in "
            f"{time.perf_counter() - shoe_started:.3f}s",
            flush=True,
        )
    print(
        f"[timing] 11-B1 total={time.perf_counter() - started:.3f}s",
        flush=True,
    )
    deformation_system = _build_instance_deformation_system(canonical_volume)
    optimization_system = (
        build_instance_optimization_system(canonical_volume)
        if args.stop_after == "11-b3"
        else None
    )
    resume_states = (
        [
            _preflight_resume_state(
                output_root / problem.shoe_name,
                problem,
                optimization_system,
            )
            for problem in problems
        ]
        if args.resume
        else [_ResumeState(None, None) for _ in problems]
    )

    continuations = []
    optimization_statuses: list[str] = []
    for problem, resume_state in zip(problems, resume_states):
        if resume_state.final_status is not None:
            continuation = resume_state.continuation
            if continuation is None:
                raise RuntimeError("validated final state unexpectedly lacks B2")
            print(
                f"[resume] {problem.shoe_name}: reused validated "
                f"{resume_state.final_status}",
                flush=True,
            )
            continuations.append(continuation)
            optimization_statuses.append(resume_state.final_status)
            continue
        if resume_state.continuation is not None:
            continuation = resume_state.continuation
            print(
                f"[resume] {problem.shoe_name}: reused validated B2 "
                f"alpha={continuation.reached_alpha:.8f}",
                flush=True,
            )
            if optimization_system is None:
                continuations.append(continuation)
                continue
        else:
            print(
                f"[11-B2] {problem.shoe_name}: starting continuation", flush=True
            )
            started = time.perf_counter()
            continuation = continue_instance_volume(
                problem, deformation_system=deformation_system
            )
            print(
                f"[timing] {problem.shoe_name} 11-B2="
                f"{time.perf_counter() - started:.3f}s",
                flush=True,
            )
            print(
                f"[11-B2] {problem.shoe_name}: {continuation.status} "
                f"at alpha={continuation.reached_alpha:.8f}",
                flush=True,
            )
        directory = output_root / problem.shoe_name
        continuation_payload = _continuation_payload(
            problem, continuation, volume_root, surface_root, directory
        )
        final_result = None
        final_payload = None
        if optimization_system is not None:
            print(
                f"[11-B3] {problem.shoe_name}: starting bounded optimization",
                flush=True,
            )
            stage_timings: defaultdict[str, float] = defaultdict(float)

            def record_timing(stage: str, seconds: float) -> None:
                stage_timings[stage] += seconds

            started = time.perf_counter()
            final_result = optimize_instance_volume(
                problem,
                continuation,
                optimization_system=optimization_system,
                deformation_system=deformation_system,
                progress=lambda record: print(
                    f"[11-B3] {problem.shoe_name}: "
                    f"phase={record.get('phase', 'optimization')} "
                    f"beta={record['trial_beta']:.8f} "
                    f"accepted={record['accepted']} "
                    f"failure={record.get('failure')} "
                    f"active={record.get('active_inner_vertices', 'n/a')}",
                    flush=True,
                ),
                timing=record_timing,
            )
            stage_timings["11-b3_total"] += time.perf_counter() - started
            print(
                f"[timing] {problem.shoe_name} "
                + " ".join(
                    f"{stage}={seconds:.3f}s"
                    for stage, seconds in sorted(stage_timings.items())
                ),
                flush=True,
            )
            print(
                f"[11-B3] {problem.shoe_name}: {final_result.status} "
                f"at beta={final_result.reached_beta:.8f}",
                flush=True,
            )
            final_payload = _final_payload(
                problem,
                continuation,
                final_result,
                volume_root,
                surface_root,
                directory,
            )
            optimization_statuses.append(final_result.status)
        write_started = time.perf_counter()
        _write_state(
            directory,
            continuation_payload,
            continuation.volume_vertices,
            problem=problem,
            final_payload=final_payload,
            final_result=final_result,
            overwrite=args.overwrite or args.resume,
        )
        print(
            f"[timing] {problem.shoe_name} artifact_writing="
            f"{time.perf_counter() - write_started:.3f}s",
            flush=True,
        )
        continuations.append(continuation)

    summary = {
        "stage": (
            "instance_volume_optimization"
            if optimization_system is not None
            else "instance_volume_continuation"
        ),
        "shoe_count": len(continuations),
        "baseline_reached_target_count": sum(
            item.status == "baseline_reached_target" for item in continuations
        ),
        "needs_11_b3_count": sum(
            item.status == "needs_11_b3" for item in continuations
        ),
        "excluded": sorted(args.exclude),
        "output_root": str(output_root),
    }
    if optimization_system is not None:
        summary.update(
            {
                "final_exact_target_count": sum(
                    status == "final_exact_target"
                    for status in optimization_statuses
                ),
                "final_corrected_target_count": sum(
                    status == "final_corrected_target"
                    for status in optimization_statuses
                ),
                "failed_11_b3_count": sum(
                    status == "failed_11_b3"
                    for status in optimization_statuses
                ),
            }
        )
    return summary


def main() -> None:
    try:
        result = run(parse_args())
    except (
        FileExistsError,
        FileNotFoundError,
        NotADirectoryError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        raise SystemExit(f"instance volume deformation failed: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))
    if result.get("failed_11_b3_count", 0):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
