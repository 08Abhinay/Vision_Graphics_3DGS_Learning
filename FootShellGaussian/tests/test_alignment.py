"""Focused tests for articulated SUPR support fitting."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from foot_prior.alignment import (
    ANCHOR_FOOT_LENGTH_RATIO,
    ANCHOR_TOE_ALLOWANCE_MM,
    MIN_TOE_ALLOWANCE_MM,
    REFERENCE_FOOT_LENGTH_MM,
    SHOE_FUNCTIONAL_LENGTH_MM,
    identify_supr_contact_regions,
    make_supr_to_shoe_axis_remap,
    neutral_length_scale,
    transform_points,
)
from foot_prior.mesh import TriangleMesh, load_triangle_mesh
from foot_prior.supr_foot import (
    SUPR_ANKLE_PITCH_INDEX,
    SUPR_MIDFOOT_PITCH_INDEX,
    load_neutral_supr_foot,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPR_MODEL = REPOSITORY_ROOT / "baselines/SUPR/data/supr_male_right_foot.npy"
PREPARATION_ROOT = Path(
    "/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/"
    "shoe_preparation"
)
RUNNER = PROJECT_ROOT / "scripts/run_alignment.py"
ARTIFACT_NAMES = {
    "support_fit.json",
    "foot_support_fitted.ply",
    "footbed_normalized.ply",
    "support_fit_overlay.ply",
}


def _run_alignment(*arguments: object) -> subprocess.CompletedProcess[str]:
    command = [sys.executable, str(RUNNER), *(str(value) for value in arguments)]
    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_exact_axis_remap_and_contact_region_partition() -> None:
    remap = make_supr_to_shoe_axis_remap()
    basis = np.eye(3, dtype=np.float64)
    expected = np.asarray(
        [[0.0, 0.0, 1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0]]
    )
    np.testing.assert_array_equal(transform_points(basis, remap), expected)
    np.testing.assert_allclose(remap @ remap, np.eye(4), atol=0.0, rtol=0.0)

    neutral = load_neutral_supr_foot(SUPR_MODEL)
    regions = identify_supr_contact_regions(neutral)
    assert len(regions.plantar_vertex_indices) == 107
    assert len(regions.plantar_face_indices) == 150
    assert {name: len(values) for name, values in regions.vertex_regions.items()} == {
        "heel": 21,
        "arch": 28,
        "forefoot": 20,
        "toes": 38,
    }
    assert {name: len(values) for name, values in regions.face_regions.items()} == {
        "heel": 33,
        "arch": 47,
        "forefoot": 33,
        "toes": 37,
    }


def test_anchor_scale_comes_only_from_the_neutral_template() -> None:
    neutral = load_neutral_supr_foot(SUPR_MODEL)
    scale = neutral_length_scale(neutral)
    remapped = transform_points(neutral.vertices, make_supr_to_shoe_axis_remap())
    assert scale * float(np.ptp(remapped[:, 0])) == pytest.approx(
        ANCHOR_FOOT_LENGTH_RATIO, abs=1e-12
    )
    # A differently shaped foot must not move the scale; that is what makes
    # length an output of the SUPR shape parameters.
    reshaped = TriangleMesh(neutral.vertices * 0.9, neutral.faces)
    assert neutral_length_scale(reshaped) == pytest.approx(scale / 0.9)
    assert REFERENCE_FOOT_LENGTH_MM / ANCHOR_FOOT_LENGTH_RATIO == pytest.approx(
        SHOE_FUNCTIONAL_LENGTH_MM
    )


def test_runner_uses_saved_support_and_writes_reversible_fit(
    tmp_path: Path,
) -> None:
    source = PREPARATION_ROOT / "canvas_shoe"
    if not source.is_dir():
        pytest.skip(f"prepared canvas shoe is unavailable: {source}")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("articulated SUPR test requires CUDA")

    preparation = tmp_path / "prepared"
    preparation.mkdir()
    for name in ("shoe_normalized.ply", "footbed_surface.ply"):
        shutil.copy2(source / name, preparation / name)
    metadata = json.loads((source / "shoe_preparation.json").read_text())
    metadata["inputs"]["shoe_mesh"] = str(tmp_path / "missing-original-shoe.ply")
    (preparation / "shoe_preparation.json").write_text(
        json.dumps(metadata), encoding="utf-8"
    )

    output = tmp_path / "output"
    output.mkdir()
    unrelated = output / "keep.txt"
    unrelated.write_text("preserve me", encoding="utf-8")
    arguments = (
        "--preparation-dir",
        preparation,
        "--supr-model",
        SUPR_MODEL,
        "--output-dir",
        output,
    )
    completed = _run_alignment(*arguments)
    assert completed.returncode == 0, completed.stderr
    assert {path.name for path in output.iterdir()} == ARTIFACT_NAMES | {"keep.txt"}

    payload = json.loads((output / "support_fit.json").read_text())
    assert payload["schema_version"] == 2
    sizing = payload["sizing"]
    assert sizing["anchor_foot_length_ratio"] == pytest.approx(
        ANCHOR_FOOT_LENGTH_RATIO
    )
    assert sizing["anchor_toe_allowance_mm"] == pytest.approx(
        ANCHOR_TOE_ALLOWANCE_MM
    )
    assert sizing["shoe_functional_length_mm"] == pytest.approx(
        SHOE_FUNCTIONAL_LENGTH_MM
    )
    # Length is no longer dialled in: it is whatever the posed foot measures
    # under the anchored scale, so it need not equal the anchor ratio. Only the
    # hard front-allowance floor is enforced.
    ratio = sizing["foot_length_ratio"]
    assert sizing["toe_allowance_mm"] == pytest.approx(
        SHOE_FUNCTIONAL_LENGTH_MM * (1.0 - ratio)
    )
    assert sizing["toe_allowance_mm"] >= MIN_TOE_ALLOWANCE_MM - 1e-9
    assert payload["bounds"]["aligned_foot"][0][0] == pytest.approx(0.0)
    assert payload["bounds"]["aligned_foot"][1][0] == pytest.approx(ratio)
    pose = np.asarray(payload["supr"]["pose_parameters_radians"])
    inactive = np.ones(len(pose), dtype=bool)
    inactive[[SUPR_ANKLE_PITCH_INDEX, SUPR_MIDFOOT_PITCH_INDEX]] = False
    np.testing.assert_array_equal(pose[inactive], 0.0)

    contact = payload["support_contact"]["face_centroids_by_region"]
    assert contact["overall"]["projected_area_coverage"] >= 0.95
    assert contact["heel"]["projected_area_coverage"] >= 0.95
    assert contact["forefoot"]["projected_area_coverage"] >= 0.95
    assert contact["toes"]["projected_area_coverage"] >= 0.90
    # First contact is the closest of the plantar vertices and the face
    # centroids, so the invariant holds over their union, not over centroids
    # alone -- which of the two touches first depends on the selected pose.
    vertices = payload["support_contact"]["plantar_vertices"]
    closest = min(
        min(record["minimum_gap"] for record in contact.values()),
        vertices["minimum_gap"],
    )
    assert closest == pytest.approx(0.0, abs=1e-10)
    assert all(record["minimum_gap"] >= -1e-10 for record in contact.values())
    assert vertices["minimum_gap"] >= -1e-10
    lateral = payload["lateral_centerline_fit"]
    assert lateral["rms_after_translation"] <= lateral["rms_before_translation"]

    transforms = payload["transforms"]
    np.testing.assert_allclose(
        np.asarray(transforms["normalized_shoe_to_posed_supr"])
        @ np.asarray(transforms["posed_supr_to_normalized_shoe"]),
        np.eye(4),
        atol=1e-10,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(transforms["original_shoe_to_posed_supr"])
        @ np.asarray(transforms["posed_supr_to_original_shoe"]),
        np.eye(4),
        atol=1e-10,
        rtol=0.0,
    )
    foot = load_triangle_mesh(output / "foot_support_fitted.ply")
    assert foot.vertices.shape == (266, 3)
    assert foot.faces.shape == (515, 3)
    np.testing.assert_array_equal(foot.faces, load_neutral_supr_foot(SUPR_MODEL).faces)

    refused = _run_alignment(*arguments)
    assert refused.returncode != 0
    assert "pass --overwrite" in refused.stderr
    assert unrelated.read_text(encoding="utf-8") == "preserve me"


def test_runner_rejects_high_heels_before_writing(tmp_path: Path) -> None:
    preparation = PREPARATION_ROOT / "red_high_heel_shoes"
    if not preparation.is_dir():
        pytest.skip(f"prepared high heel is unavailable: {preparation}")
    output = tmp_path / "high-heel-output"
    result = _run_alignment(
        "--preparation-dir",
        preparation,
        "--supr-model",
        SUPR_MODEL,
        "--output-dir",
        output,
    )
    assert result.returncode != 0
    assert "shoe_profile='normal' only" in result.stderr
    assert not output.exists()
