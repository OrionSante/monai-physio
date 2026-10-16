"""Non-regression tests for the Orion ascending-aorta diameter measurement.

The measurement lives in
``experiments/Heart-GatedCT_To_USD/orion_diametre_aorte.py``. These tests feed
it synthetic oblique tubes of known cross-section, sampled on the anisotropic
grid of the Slicer-Heart-CT scan (0.7 x 0.6 x 0.6 mm), and check the three
diameters it reports (maximal and minimal Feret widths, and
area-equivalent) against the exact values. Tolerance: 0.6 mm, one voxel.

They test the GEOMETRY only. Whether a segmentation follows the aortic lumen is
a separate question (see the script's control image).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "Heart-GatedCT_To_USD"
    / "orion_diametre_aorte.py"
)
SPACING = np.array([0.7, 0.6, 0.6])
TOLERANCE_MM = 0.6


def _load():
    pytest.importorskip("totalsegmentator")
    spec = importlib.util.spec_from_file_location("orion_diametre_aorte", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _oblique_tube(
    semi_axis_a: float, semi_axis_b: float, shape=(200, 220, 220), length=120.0
):
    """Boolean mask of an elliptic tube whose axis is tilted 30 degrees."""
    z, y, x = np.indices(shape).astype(float)
    points = np.stack([z * SPACING[0], y * SPACING[1], x * SPACING[2]], axis=-1)
    origin = np.array([10.0, 40.0, 40.0])
    axis = np.array([np.cos(np.radians(30)), np.sin(np.radians(30)), 0.0])
    rel = points - origin
    along = rel @ axis
    across = rel - along[..., None] * axis
    e1 = np.array([0.0, 0.0, 1.0])
    e2 = np.cross(axis, e1)
    inside = (across @ e1) ** 2 / semi_axis_a**2 + (
        across @ e2
    ) ** 2 / semi_axis_b**2 <= 1.0
    return inside & (along > 0) & (along < length)


def _profile(module, mask):
    iso = module.isotrope(mask, SPACING, module.PAS_ISOTROPE_MM)
    return module.profil(iso, module.PAS_ISOTROPE_MM)


@pytest.mark.parametrize(
    ("semi_a", "semi_b"),
    [(12.5, 12.5), (22.5, 22.5), (20.0, 15.0)],
    ids=["circle-25mm", "circle-45mm", "oval-40x30mm"],
)
def test_diameters_of_known_tubes(semi_a, semi_b):
    module = _load()
    profile = _profile(module, _oblique_tube(semi_a, semi_b))
    assert len(profile) >= 20, "too few sections measured along the axis"

    expected_max = 2 * max(semi_a, semi_b)
    expected_min = 2 * min(semi_a, semi_b)
    expected_area = 2 * np.sqrt(semi_a * semi_b)
    for section in profile:
        assert abs(section["diametre_max_mm"] - expected_max) <= TOLERANCE_MM
        assert abs(section["diametre_min_mm"] - expected_min) <= TOLERANCE_MM
        assert abs(section["diametre_aire_mm"] - expected_area) <= TOLERANCE_MM


def test_sections_cut_by_the_field_of_view_are_discarded():
    """A tube leaving the scanned volume must not report a shrinking diameter."""
    module = _load()
    tube = _oblique_tube(16.0, 16.0, length=400.0)  # runs out of the volume
    profile = _profile(module, tube)
    assert profile, "the part inside the field of view should still be measured"
    for section in profile:
        assert abs(section["diametre_aire_mm"] - 32.0) <= TOLERANCE_MM


def _segment(shape, start_mm, end_mm, radius_mm):
    """Boolean mask of a straight tube between two points given in mm (z, y, x)."""
    z, y, x = np.indices(shape).astype(float)
    points = np.stack([z * SPACING[0], y * SPACING[1], x * SPACING[2]], axis=-1)
    start, end = np.asarray(start_mm), np.asarray(end_mm)
    direction = end - start
    t = np.clip(((points - start) @ direction) / (direction @ direction), 0.0, 1.0)
    closest = start + t[..., None] * direction
    return np.linalg.norm(points - closest, axis=-1) <= radius_mm


def test_pulmonary_bifurcation_is_found_at_the_branching_height():
    """Trunk rising to 90 mm, then left and right branches: junction at ~90 mm."""
    module = _load()
    shape = (180, 220, 220)
    trunk = _segment(shape, (20.0, 66.0, 66.0), (90.0, 66.0, 66.0), 12.0)
    left = _segment(shape, (90.0, 66.0, 66.0), (100.0, 66.0, 116.0), 9.0)
    right = _segment(shape, (90.0, 66.0, 66.0), (100.0, 66.0, 16.0), 9.0)
    labels = np.zeros(shape, dtype=np.uint8)
    labels[trunk | left | right] = module.LABEL_ARTERE_PULMONAIRE

    found = module.bifurcation_pulmonaire(labels, SPACING)

    assert abs(found["point_mm"][0] - 90.0) <= 5.0
    assert len(found["branches_mm"]) == 3
    assert min(found["branches_mm"]) >= module.BRANCHE_MIN_MM


def test_no_bifurcation_means_no_measurement_level():
    """A single straight vessel has no three-branch junction: refuse, do not guess."""
    module = _load()
    shape = (180, 120, 120)
    labels = np.zeros(shape, dtype=np.uint8)
    labels[_segment(shape, (20.0, 36.0, 36.0), (100.0, 36.0, 36.0), 12.0)] = (
        module.LABEL_ARTERE_PULMONAIRE
    )
    with pytest.raises(SystemExit):
        module.bifurcation_pulmonaire(labels, SPACING)
