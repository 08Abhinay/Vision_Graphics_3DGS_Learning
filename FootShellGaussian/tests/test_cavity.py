"""Focused tests for surface-based shoe-cavity analysis."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from foot_prior.cavity import (
    CavityEvaluator,
    _build_triangle_broad_phase,
    _find_collision_pairs,
    analyze_fitted_foot_cavity,
)
from foot_prior.mesh import TriangleMesh


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREPARATION_ROOT = Path(
    "/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/"
    "shoe_preparation_2"
)
SUPPORT_FIT_ROOT = Path(
    "/home/ab5298/Outputs/FootShellGaussian/golden_set_evaluation/"
    "support_fit_2"
)
RUNNER = PROJECT_ROOT / "scripts/run_cavity_analysis.py"


@pytest.mark.parametrize(
    ("second", "expected"),
    [
        (
            ((0.0, 0.0, 1.0), (1.0, 0.0, 1.0), (0.0, 1.0, 1.0)),
            False,
        ),
        (
            ((0.25, 0.25, -1.0), (0.25, 0.25, 1.0), (0.75, 0.25, 0.0)),
            True,
        ),
        (
            ((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.0, 0.0)),
            True,
        ),
        (
            ((0.25, 0.25, 0.0), (1.25, 0.25, 0.0), (0.25, 1.25, 0.0)),
            True,
        ),
    ],
)
def test_spatial_broad_phase_preserves_exact_triangle_contacts(
    second: tuple[tuple[float, float, float], ...], expected: bool
) -> None:
    first = np.asarray(
        (((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),),
        dtype=np.float64,
    )
    obstacle = np.asarray((second,), dtype=np.float64)
    first_ids = np.asarray((7,), dtype=np.int64)
    second_ids = np.asarray((11,), dtype=np.int64)
    index = _build_triangle_broad_phase(obstacle, second_ids)
    pairs = _find_collision_pairs(
        first,
        first_ids,
        obstacle,
        second_ids,
        1.0e-14,
        obstacle_broad_phase=index,
    )
    assert bool(len(pairs)) is expected
    if expected:
        np.testing.assert_array_equal(pairs, np.asarray(((7, 11),)))
        reversed_pairs = _find_collision_pairs(
            first[:, ::-1], first_ids, obstacle[:, ::-1], second_ids, 1.0e-14
        )
        np.testing.assert_array_equal(reversed_pairs, pairs)


def test_spatial_broad_phase_deduplicates_repeated_face_ids() -> None:
    triangle = np.asarray(
        (((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),),
        dtype=np.float64,
    )
    obstacles = np.repeat(triangle, 2, axis=0)
    pairs = _find_collision_pairs(
        triangle,
        np.asarray((3,), dtype=np.int64),
        obstacles,
        np.asarray((9, 9), dtype=np.int64),
        1.0e-14,
    )
    np.testing.assert_array_equal(pairs, np.asarray(((3, 9),)))


def _quad(
    vertices: list[list[float]],
    faces: list[list[int]],
    corners: tuple[tuple[float, float, float], ...],
) -> None:
    offset = len(vertices)
    vertices.extend([list(point) for point in corners])
    faces.extend(
        [[offset, offset + 1, offset + 2], [offset, offset + 2, offset + 3]]
    )


def _shoe(
    obstacle: str | None = None, *, reverse_winding: bool = False
) -> tuple[TriangleMesh, TriangleMesh, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    _quad(
        vertices,
        faces,
        ((0.0, 0.0, -0.5), (1.0, 0.0, -0.5), (1.0, 0.0, 0.5), (0.0, 0.0, 0.5)),
    )
    surfaces = {
        "medial": (
            (0.0, 0.1, 0.1),
            (1.0, 0.1, 0.1),
            (1.0, -0.6, 0.1),
            (0.0, -0.6, 0.1),
        ),
        "upper": (
            (0.0, -0.2, -0.5),
            (1.0, -0.2, -0.5),
            (1.0, -0.2, 0.5),
            (0.0, -0.2, 0.5),
        ),
        "toe": (
            (0.7, 0.1, -0.5),
            (0.7, 0.1, 0.5),
            (0.7, -0.6, 0.5),
            (0.7, -0.6, -0.5),
        ),
        "heel": (
            (0.2, 0.1, -0.5),
            (0.2, -0.6, -0.5),
            (0.2, -0.6, 0.5),
            (0.2, 0.1, 0.5),
        ),
    }
    # Distant disconnected walls make a valid open shoe without contacting the foot.
    for corners in (
        ((0.0, 0.1, -0.5), (1.0, 0.1, -0.5), (1.0, -0.6, -0.5), (0.0, -0.6, -0.5)),
        ((0.0, 0.1, 0.5), (0.0, -0.6, 0.5), (1.0, -0.6, 0.5), (1.0, 0.1, 0.5)),
        ((1.0, 0.1, -0.5), (1.0, 0.1, 0.5), (1.0, -0.6, 0.5), (1.0, -0.6, -0.5)),
        ((0.0, 0.1, -0.5), (0.0, -0.6, -0.5), (0.0, -0.6, 0.5), (0.0, 0.1, 0.5)),
    ):
        _quad(vertices, faces, corners)
    if obstacle is not None:
        _quad(vertices, faces, surfaces[obstacle])
    face_array = np.asarray(faces, dtype=np.int64)
    if reverse_winding:
        face_array = face_array[:, ::-1]
    shoe = TriangleMesh(np.asarray(vertices), face_array)
    footbed = TriangleMesh(shoe.vertices, shoe.faces[:2])
    return shoe, footbed, np.asarray([0, 1], dtype=np.int64)


def _foot(
    *,
    bottom_y: float = 0.0,
    top_y: float = -0.2,
    z_center: float = 0.0,
    half_width: float = 0.2,
) -> TriangleMesh:
    vertices = np.asarray(
        [
            [0.2, top_y, z_center - half_width],
            [0.8, top_y, z_center - half_width],
            [0.8, top_y, z_center + half_width],
            [0.2, top_y, z_center + half_width],
            [0.2, bottom_y, z_center - half_width],
            [0.8, bottom_y, z_center - half_width],
            [0.8, bottom_y, z_center + half_width],
            [0.2, bottom_y, z_center + half_width],
        ]
    )
    faces = np.asarray(
        [
            [0, 2, 1], [0, 3, 2],
            [4, 5, 6], [4, 6, 7],
            [0, 1, 5], [0, 5, 4],
            [1, 2, 6], [1, 6, 5],
            [2, 3, 7], [2, 7, 6],
            [3, 0, 4], [3, 4, 7],
        ],
        dtype=np.int64,
    )
    return TriangleMesh(vertices, faces)


def _analyze(
    obstacle: str | None = None,
    *,
    reverse_winding: bool = False,
    bottom_y: float = 0.0,
    top_y: float = -0.2,
    z_center: float = 0.0,
    half_width: float = 0.2,
):
    shoe, footbed, source_faces = _shoe(
        obstacle, reverse_winding=reverse_winding
    )
    foot = _foot(
        bottom_y=bottom_y,
        top_y=top_y,
        z_center=z_center,
        half_width=half_width,
    )
    result = analyze_fitted_foot_cavity(
        shoe,
        footbed,
        foot,
        source_faces,
        np.asarray([4, 5, 6, 7]),
        np.asarray([2, 3]),
        np.asarray([[0.0, 0.0], [1.0, 0.0]]),
    )
    return result, foot


def test_footbed_contact_is_allowed_and_penetration_is_reported() -> None:
    contact, foot = _analyze()
    assert contact.status == "clear"
    assert len(contact.collision_pairs) == 0
    assert contact.support_contact["plantar_vertices"]["contacting_sample_count"] == 4
    assert contact.support_contact["plantar_vertices"]["penetrating_sample_count"] == 0
    assert np.all(contact.foot_vertex_colors(foot, 0.1)[:, 0] != 220)

    penetration, _ = _analyze(bottom_y=0.02, top_y=-0.18)
    assert penetration.status == "collisions_detected"
    penetration_count = penetration.support_contact["plantar_vertices"][
        "penetrating_sample_count"
    ]
    assert penetration_count == 4


@pytest.mark.parametrize("obstacle", ["medial", "upper", "toe", "heel"])
def test_non_footbed_contact_is_a_collision(obstacle: str) -> None:
    result, foot = _analyze(obstacle)
    assert result.status == "collisions_detected"
    assert len(result.collision_pairs) > 0
    colors = result.foot_vertex_colors(foot, 0.1)
    assert np.any(np.all(colors == np.asarray([220, 45, 45, 255]), axis=1))


def test_opening_is_not_treated_as_a_closed_lid() -> None:
    result, _ = _analyze(top_y=-0.8)
    assert result.status == "clear"
    assert len(result.collision_pairs) == 0


def test_upper_protrusion_is_signed_without_triangle_crossing() -> None:
    result, _ = _analyze("upper", bottom_y=-0.25, top_y=-0.35)
    assert len(result.collision_pairs) == 0
    assert result.status == "protrusion_detected"
    assert result.signed_clearance.outside_area_fraction > 0.0


def test_side_protrusion_is_signed_without_triangle_crossing() -> None:
    result, _ = _analyze("medial", z_center=0.30, half_width=0.15)
    assert len(result.collision_pairs) == 0
    assert result.status == "protrusion_detected"
    assert result.signed_clearance.outside_area_fraction > 0.0


def test_signed_exemption_does_not_disable_exact_collision() -> None:
    shoe, footbed, source_faces = _shoe("upper")
    beyond = _foot(bottom_y=-0.25, top_y=-0.35)
    evaluator = CavityEvaluator.build(
        shoe,
        footbed,
        source_faces,
        beyond,
        np.asarray([[0.0, 0.0], [1.0, 0.0]]),
    )
    all_vertices = np.arange(len(beyond.vertices), dtype=np.int64)
    all_faces = np.arange(len(beyond.faces), dtype=np.int64)
    signed_exempt = evaluator.analyze(
        beyond,
        np.asarray([4, 5, 6, 7]),
        np.asarray([2, 3]),
        all_vertices,
        all_faces,
    )
    assert signed_exempt.status == "clear"
    assert len(signed_exempt.signed_clearance.outside_face_indices) == 0
    assert np.isfinite(
        signed_exempt.signed_clearance.vertex_upper_clearances
    ).any()

    crossing = _foot(bottom_y=0.0, top_y=-0.3)
    collision_evaluator = CavityEvaluator.build(
        shoe,
        footbed,
        source_faces,
        crossing,
        np.asarray([[0.0, 0.0], [1.0, 0.0]]),
    )
    still_colliding = collision_evaluator.analyze(
        crossing,
        np.asarray([4, 5, 6, 7]),
        np.asarray([2, 3]),
        np.arange(len(crossing.vertices), dtype=np.int64),
        np.arange(len(crossing.faces), dtype=np.int64),
    )
    assert still_colliding.status == "collisions_detected"
    assert len(still_colliding.collision_pairs) > 0


def test_winding_and_repeated_evaluation_do_not_change_results() -> None:
    first, _ = _analyze("toe")
    repeated, _ = _analyze("toe")
    reversed_result, _ = _analyze("toe", reverse_winding=True)
    np.testing.assert_array_equal(first.collision_pairs, repeated.collision_pairs)
    np.testing.assert_array_equal(
        first.collision_pairs, reversed_result.collision_pairs
    )
    np.testing.assert_allclose(first.vertex_clearances, repeated.vertex_clearances)
    np.testing.assert_allclose(
        first.vertex_clearances, reversed_result.vertex_clearances
    )


def _run_runner(*arguments: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(RUNNER), *(str(value) for value in arguments)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_runner_writes_three_artifacts_and_preserves_unrelated_file(
    tmp_path: Path,
) -> None:
    preparation = PREPARATION_ROOT / "canvas_shoe"
    support_fit = SUPPORT_FIT_ROOT / "canvas_shoe"
    if not preparation.is_dir() or not support_fit.is_dir():
        pytest.skip("reviewed Canvas preparation and support fit are unavailable")
    support_record = support_fit / "support_fit.json"
    if (
        not support_record.is_file()
        or json.loads(support_record.read_text()).get("schema_version") != 2
    ):
        pytest.skip("regenerate the Canvas support fit with schema version 2")
    output = tmp_path / "output"
    arguments = (
        "--preparation-dir", preparation,
        "--support-fit-dir", support_fit,
        "--output-dir", output,
    )
    first = _run_runner(*arguments)
    assert first.returncode == 0, first.stderr
    payload = json.loads((output / "cavity_analysis.json").read_text())
    assert payload["schema_version"] == 2
    assert payload["status"] in {
        "clear",
        "protrusion_detected",
        "collisions_detected",
    }
    assert payload["contact_policy"]["allowed"].startswith("plantar contact")
    assert {path.name for path in output.iterdir()} == {
        "cavity_analysis.json",
        "foot_clearance_colored.ply",
        "cavity_overlay.ply",
    }
    keep = output / "keep.txt"
    keep.write_text("unchanged", encoding="utf-8")
    refused = _run_runner(*arguments)
    assert refused.returncode != 0
    assert "pass --overwrite" in refused.stderr
    replaced = _run_runner(*arguments, "--overwrite")
    assert replaced.returncode == 0, replaced.stderr
    assert keep.read_text(encoding="utf-8") == "unchanged"


def test_runner_rejects_high_heels_before_writing(tmp_path: Path) -> None:
    preparation = PREPARATION_ROOT / "red_high_heel_shoes"
    if not preparation.is_dir():
        pytest.skip("reviewed high-heel preparation is unavailable")
    output = tmp_path / "output"
    result = _run_runner(
        "--preparation-dir", preparation,
        "--support-fit-dir", tmp_path / "missing-support-fit",
        "--output-dir", output,
    )
    assert result.returncode != 0
    assert "shoe_profile='normal' only" in result.stderr
    assert not output.exists()
