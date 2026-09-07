"""Focused tests for collision-aware SUPR containment fitting."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from foot_prior.alignment import (
    ANCHOR_FOOT_LENGTH_RATIO,
    ANCHOR_TOE_ALLOWANCE_MM,
    MIN_FOOT_LENGTH_RATIO,
    REFERENCE_FOOT_LENGTH_MM,
    SHOE_FUNCTIONAL_LENGTH_MM,
    _triangle_geometry,
    build_support_placement_context,
    evaluate_support_placement,
    foot_length_ratio_to_toe_allowance,
    toe_allowance_to_foot_length_ratio,
)
from foot_prior.containment import (
    BETA_AXIS_MAGNITUDES,
    MAX_TARGET_TOE_ALLOWANCE_MM,
    MIN_TARGET_TOE_ALLOWANCE_MM,
    TARGET_MAX_RATIO,
    TARGET_MIN_RATIO,
    TARGET_TOE_ALLOWANCE_MM,
    _Parameters,
    _Search,
    _beta_templates,
    _hadamard,
    _minimum_norm_joint_update,
    affected_area_fraction,
    collision_area_fraction,
)
from foot_prior.mesh import TriangleMesh, load_triangle_mesh
from foot_prior.supr_foot import load_neutral_supr_foot


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPR_MODEL = REPOSITORY_ROOT / "baselines/SUPR/data/supr_male_right_foot.npy"
OUTPUT_ROOT = Path(
    "/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation"
)
RUNNER = PROJECT_ROOT / "scripts/run_containment_fit.py"


def _plane() -> TriangleMesh:
    return TriangleMesh(
        np.asarray(
            [
                [-0.2, 0.0, -0.6],
                [1.2, 0.0, -0.6],
                [1.2, 0.0, 0.6],
                [-0.2, 0.0, 0.6],
            ]
        ),
        np.asarray([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
    )


def _context() -> object:
    support = _plane()
    return build_support_placement_context(
        load_neutral_supr_foot(SUPR_MODEL),
        support,
        support,
        np.asarray([[-0.2, 0.0], [1.2, 0.0]]),
    )


def _place(
    vertices: np.ndarray,
    *,
    context: object | None = None,
    reference: np.ndarray | None = None,
    heel_offset_x: float = 0.0,
    lateral_offset_z: float = 0.0,
) -> tuple[object | None, str | None]:
    """Place explicit vertices, defaulting the distortion reference to itself."""

    placement_context = _context() if context is None else context
    source = vertices if reference is None else reference
    crosses, areas = _triangle_geometry(source, placement_context.faces)
    return evaluate_support_placement(
        placement_context,
        0.0,
        0.0,
        np.zeros(39, dtype=np.float64),
        np.zeros(10, dtype=np.float64),
        vertices,
        np.zeros((13, 3), dtype=np.float64),
        crosses,
        areas,
        heel_offset_x=heel_offset_x,
        lateral_offset_z=lateral_offset_z,
    )


def test_support_candidate_respects_heel_toe_and_first_contact() -> None:
    neutral = load_neutral_supr_foot(SUPR_MODEL)
    # A forward heel offset now spends real toe allowance, so it has to stay
    # inside the 2.5 mm the anchor leaves above the 10 mm floor.
    candidate, rejection = _place(
        neutral.vertices, heel_offset_x=0.005, lateral_offset_z=0.02
    )
    assert rejection is None
    assert candidate is not None
    expected_toe = 0.005 + ANCHOR_FOOT_LENGTH_RATIO
    assert candidate.aligned_vertices[:, 0].min() == pytest.approx(0.005)
    assert candidate.aligned_vertices[:, 0].max() == pytest.approx(expected_toe)
    assert candidate.lateral_fit["lateral_offset_z"] == pytest.approx(0.02)
    assert candidate.plantar_vertex_contact["minimum_gap"] == pytest.approx(
        0.0, abs=1e-10
    )
    assert candidate.plantar_vertex_contact["coverage"] >= 0.95


def test_anchor_and_allowance_are_explicit() -> None:
    assert REFERENCE_FOOT_LENGTH_MM == 250.0
    assert ANCHOR_TOE_ALLOWANCE_MM == 12.5
    assert SHOE_FUNCTIONAL_LENGTH_MM == pytest.approx(262.5)
    assert ANCHOR_FOOT_LENGTH_RATIO == pytest.approx(250.0 / 262.5)
    assert MIN_FOOT_LENGTH_RATIO == 0.80
    # One normalized unit is a fixed 262.5 mm, so the conversions are exact
    # inverses rather than assuming the foot is always 250 mm long.
    assert toe_allowance_to_foot_length_ratio(12.5) == pytest.approx(
        ANCHOR_FOOT_LENGTH_RATIO
    )
    assert TARGET_TOE_ALLOWANCE_MM == 20.0
    assert MIN_TARGET_TOE_ALLOWANCE_MM == 18.0
    assert MAX_TARGET_TOE_ALLOWANCE_MM == 22.0
    assert TARGET_MAX_RATIO == pytest.approx(1.0 - 18.0 / 262.5)
    assert TARGET_MIN_RATIO == pytest.approx(1.0 - 22.0 / 262.5)
    assert foot_length_ratio_to_toe_allowance(0.80) == pytest.approx(52.5)
    assert foot_length_ratio_to_toe_allowance(
        0.80, heel_offset_x=0.02
    ) == pytest.approx(47.25)


def test_neutral_foot_lands_exactly_on_the_anchor() -> None:
    neutral = load_neutral_supr_foot(SUPR_MODEL)
    candidate, rejection = _place(neutral.vertices)
    assert rejection is None
    assert candidate is not None
    assert candidate.foot_length_ratio == pytest.approx(
        ANCHOR_FOOT_LENGTH_RATIO, abs=1e-12
    )
    assert candidate.toe_allowance_mm == pytest.approx(
        ANCHOR_TOE_ALLOWANCE_MM, abs=1e-9
    )


def test_shape_drives_length_instead_of_being_normalized_away() -> None:
    """A shorter foot must stay shorter in the shoe.

    This is the invariant the anchored scale exists to create. Under the
    retired rule every candidate was rescaled to a target length, so a
    genuinely smaller foot was silently enlarged back to the same ratio.
    """

    neutral = load_neutral_supr_foot(SUPR_MODEL)
    context = _context()
    baseline, _ = _place(neutral.vertices, context=context)
    shorter, rejection = _place(neutral.vertices * 0.93, context=context)
    assert rejection is None
    assert baseline is not None and shorter is not None
    assert shorter.foot_length_ratio < baseline.foot_length_ratio - 1e-6
    assert shorter.foot_length_ratio == pytest.approx(
        0.93 * baseline.foot_length_ratio
    )
    assert shorter.toe_allowance_mm > baseline.toe_allowance_mm + 1.0
    assert shorter.scale == pytest.approx(baseline.scale)


def test_front_allowance_is_one_sided() -> None:
    neutral = load_neutral_supr_foot(SUPR_MODEL)
    context = _context()

    # Above 15 mm is a reporting outcome, never a rejection.
    roomy, rejection = _place(neutral.vertices * 0.93, context=context)
    assert rejection is None
    assert roomy is not None
    assert roomy.toe_allowance_mm > 15.0

    # Running past the functional toe stays hard.
    long_foot, rejection = _place(neutral.vertices * 1.05, context=context)
    assert long_foot is None
    assert rejection == "insufficient_toe_allowance"

    # And a foot that has shrunk past the diagnostic limit is still refused.
    tiny, rejection = _place(neutral.vertices * 0.80, context=context)
    assert tiny is None
    assert rejection == "foot_shorter_than_diagnostic_limit"


def test_distortion_gate_measures_pose_not_shape() -> None:
    """The gate must charge articulation, not anatomy, against its budget.

    Comparing a reshaped foot against the zero-beta template made shape change
    look like mesh damage and rejected the narrow feet the fitter now needs.
    """

    neutral = load_neutral_supr_foot(SUPR_MODEL)
    context = _context()
    reshaped = neutral.vertices * np.asarray([0.78, 0.78, 1.0])

    against_neutral, rejection = _place(
        reshaped, context=context, reference=neutral.vertices
    )
    assert against_neutral is None
    assert rejection == "triangle_area_distortion"

    against_own_shape, rejection = _place(reshaped, context=context)
    assert rejection is None
    assert against_own_shape is not None
    assert against_own_shape.distortion["symmetric_area_ratio_p99"] == pytest.approx(
        1.0, abs=1e-9
    )


def test_joint_update_can_move_parameters_together() -> None:
    update = _minimum_norm_joint_update(
        np.asarray([[1.0, 1.0], [1.0, -1.0]]),
        np.asarray([1.0, 0.0]),
    )
    np.testing.assert_allclose(update, np.asarray([0.5, 0.5]))


def test_broad_templates_cover_all_ten_betas_and_coupled_shapes() -> None:
    initial = _Parameters(0.0, 0.0, 0.0, 0.0, np.zeros(10))
    targeted, direct = _beta_templates(initial)
    assert any(abs(values[0]) == BETA_AXIS_MAGNITUDES[0] for values in direct)
    for index in range(1, 10):
        assert any(abs(values[index]) == BETA_AXIS_MAGNITUDES[0] for values in targeted)
    assert any(np.count_nonzero(values[1:]) > 1 for values in targeted)
    matrix = _hadamard(16)
    np.testing.assert_allclose(matrix @ matrix.T, 16.0 * np.eye(16))


def test_lateral_offset_has_no_fixed_grid_cell_clamp() -> None:
    initial = _Parameters(0.0, 0.0, 0.0, 0.0, np.zeros(10))
    search = object.__new__(_Search)
    search.initial = initial
    far_lateral = replace(initial, lateral_offset_z=100.0)
    assert search._parameter_rejection(far_lateral) is None


def test_collision_area_ignores_duplicate_shoe_triangle_hits() -> None:
    foot = TriangleMesh(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
    )
    sparse = np.asarray([[0, 4]], dtype=np.int64)
    dense = np.asarray([[0, index] for index in range(100)], dtype=np.int64)
    sparse_area, sparse_fraction, sparse_faces = collision_area_fraction(
        foot, sparse
    )
    dense_area, dense_fraction, dense_faces = collision_area_fraction(
        foot, dense
    )
    assert dense_area == pytest.approx(sparse_area)
    assert dense_fraction == pytest.approx(sparse_fraction)
    np.testing.assert_array_equal(dense_faces, sparse_faces)


def test_affected_area_counts_collision_and_protrusion_union_once() -> None:
    foot = TriangleMesh(
        np.asarray(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ]
        ),
        np.asarray([[0, 1, 2], [1, 3, 2]], dtype=np.int64),
    )
    area, fraction, faces = affected_area_fraction(
        foot, np.asarray([0]), np.asarray([0, 1])
    )
    np.testing.assert_array_equal(faces, np.asarray([0, 1]))
    assert area == pytest.approx(1.0)
    assert fraction == pytest.approx(1.0)


def _run(*arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), *(str(value) for value in arguments)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_real_normal_shoe_writes_schema2_containment_fit(tmp_path: Path) -> None:
    preparation = OUTPUT_ROOT / "shoe_preparation_2/aj_12_basketball_sneakers"
    support_fit = OUTPUT_ROOT / "support_fit_2/aj_12_basketball_sneakers"
    cavity = OUTPUT_ROOT / "cavity_analysis/aj_12_basketball_sneakers"
    if not all(path.is_dir() for path in (preparation, support_fit, cavity)):
        pytest.skip("reviewed AJ checkpoint artifacts are unavailable")
    support_record = support_fit / "support_fit.json"
    if (
        not support_record.is_file()
        or json.loads(support_record.read_text()).get("schema_version") != 2
    ):
        pytest.skip("regenerate the AJ support fit with schema version 2")
    cavity_record = cavity / "cavity_analysis.json"
    if (
        not cavity_record.is_file()
        or json.loads(cavity_record.read_text()).get("schema_version") != 2
    ):
        pytest.skip("regenerate the AJ cavity analysis with schema version 2")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("containment fitting requires CUDA")
    output = tmp_path / "containment"
    arguments = (
        "--preparation-dir", preparation,
        "--support-fit-dir", support_fit,
        "--cavity-analysis-dir", cavity,
        "--supr-model", SUPR_MODEL,
        "--output-dir", output,
    )
    completed = _run(*arguments)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads((output / "containment_fit.json").read_text())
    assert payload["schema_version"] == 4
    assert payload["status"] in {
        "contained_target_fit",
        "residual_target_fit",
    }
    sizing = payload["sizing"]
    assert MIN_TARGET_TOE_ALLOWANCE_MM <= sizing["toe_allowance_mm"] <= MAX_TARGET_TOE_ALLOWANCE_MM
    assert sizing["anchor_foot_length_ratio"] == pytest.approx(
        ANCHOR_FOOT_LENGTH_RATIO
    )
    # Length must be consistent with the anchored scale rather than dialled in.
    assert sizing["foot_length_ratio"] == pytest.approx(
        1.0 - sizing["toe_allowance_mm"] / SHOE_FUNCTIONAL_LENGTH_MM
        - sizing["heel_offset_x"]
    )
    assert "evaluated_size_ratios" not in payload["search"]
    result = load_triangle_mesh(output / "foot_containment_fitted.ply")
    original = load_triangle_mesh(support_fit / "foot_support_fitted.ply")
    np.testing.assert_array_equal(result.faces, original.faces)
    assert {path.name for path in output.iterdir()} == {
        "containment_fit.json",
        "foot_containment_fitted.ply",
        "foot_clearance_colored.ply",
        "containment_fit_overlay.ply",
    }

    keep = output / "keep.txt"
    keep.write_text("preserve", encoding="utf-8")
    refused = _run(*arguments)
    assert refused.returncode != 0
    assert "pass --overwrite" in refused.stderr
    replaced = _run(*arguments, "--overwrite")
    assert replaced.returncode == 0, replaced.stderr
    assert keep.read_text(encoding="utf-8") == "preserve"


def test_runner_rejects_high_heel_before_other_inputs(tmp_path: Path) -> None:
    preparation = OUTPUT_ROOT / "shoe_preparation_2/red_high_heel_shoes"
    if not preparation.is_dir():
        pytest.skip("reviewed high-heel preparation is unavailable")
    output = tmp_path / "output"
    result = _run(
        "--preparation-dir", preparation,
        "--support-fit-dir", tmp_path / "missing-support",
        "--cavity-analysis-dir", tmp_path / "missing-cavity",
        "--supr-model", SUPR_MODEL,
        "--output-dir", output,
    )
    assert result.returncode != 0
    assert "shoe_profile='normal' only" in result.stderr
    assert not output.exists()
