#!/usr/bin/env python3
"""Build validated Checkpoint 11-B2 continuation warm starts."""

from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any
import zipfile

import numpy as np

from foot_prior.anatomical_volume import (
    _build_instance_deformation_system,
    _load_extended_reference,
    continue_instance_volume,
    load_canonical_anatomical_volume,
    load_instance_volume_problem,
)
from foot_prior.anatomy import array_digest


ARTIFACT_NAMES = ("continuation_state.json", "continuation_state.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Checkpoint 11-B1 inputs and save the last valid smooth "
            "Checkpoint 11-B2 continuation state."
        )
    )
    parser.add_argument("--anatomical-volume-root", required=True, type=Path)
    parser.add_argument(
        "--extended-anatomical-surface-root", required=True, type=Path
    )
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        metavar="SHOE",
        help="Exclude one shoe name; may be supplied more than once.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("shoes", nargs="*")
    return parser.parse_args()


def _write_deterministic_npz(path: Path, **arrays: np.ndarray) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.save(buffer, np.asarray(arrays[name]), allow_pickle=False)
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
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
    return selected


def _write_state(
    directory: Path,
    payload: dict[str, Any],
    vertices: np.ndarray,
) -> None:
    directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    try:
        (staging / "continuation_state.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _write_deterministic_npz(
            staging / "continuation_state.npz",
            last_valid_volume_vertices=vertices,
        )
        directory.mkdir(parents=True, exist_ok=True)
        for artifact in ARTIFACT_NAMES:
            os.replace(staging / artifact, directory / artifact)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    volume_root = args.anatomical_volume_root.expanduser().resolve(strict=True)
    surface_root = (
        args.extended_anatomical_surface_root.expanduser().resolve(strict=True)
    )
    output_root = args.output_root.expanduser().resolve()
    if not volume_root.is_dir() or not surface_root.is_dir():
        raise NotADirectoryError("anatomical volume and surface roots must be directories")
    names = _validated_names(volume_root, list(args.shoes), list(args.exclude))

    existing = [
        output_root / name / artifact
        for name in names
        for artifact in ARTIFACT_NAMES
        if (output_root / name / artifact).exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"continuation artifacts already exist; pass --overwrite: {existing[0]}"
        )

    canonical_volume = load_canonical_anatomical_volume(volume_root)
    extended_reference = _load_extended_reference(surface_root)
    problems = [
        load_instance_volume_problem(
            volume_root,
            surface_root,
            name,
            canonical_volume=canonical_volume,
            extended_reference=extended_reference,
        )
        for name in names
    ]
    deformation_system = _build_instance_deformation_system(canonical_volume)
    completed = []
    for problem in problems:
        print(f"[11-B2] {problem.shoe_name}: starting continuation", flush=True)
        result = continue_instance_volume(
            problem,
            deformation_system=deformation_system,
        )
        print(
            f"[11-B2] {problem.shoe_name}: {result.status} "
            f"at alpha={result.reached_alpha:.8f}",
            flush=True,
        )
        completed.append((problem, result))

    for problem, result in completed:
        directory = output_root / problem.shoe_name
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
            artifact: str(directory / artifact) for artifact in ARTIFACT_NAMES
        }
        _write_state(directory, payload, result.volume_vertices)

    return {
        "stage": "instance_volume_continuation",
        "shoe_count": len(completed),
        "baseline_reached_target_count": sum(
            result.status == "baseline_reached_target" for _, result in completed
        ),
        "needs_11_b3_count": sum(
            result.status == "needs_11_b3" for _, result in completed
        ),
        "excluded": sorted(args.exclude),
        "output_root": str(output_root),
    }


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
        raise SystemExit(f"instance volume continuation failed: {error}") from error
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
