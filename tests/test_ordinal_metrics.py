import math

import numpy as np

from src.utils.ordinal_metrics import (
    confusion_matrix,
    expected_calibration_error,
    mean_mass_by_dist,
    odc,
    ordinal_shape_metrics,
)


def test_perfect_diagonal():
    cm = np.diag([10, 5, 5, 3, 2])
    assert odc(cm, True) == (1.0, 1.0) and odc(cm, False) == (1.0, 1.0)
    assert mean_mass_by_dist(cm) == [1.0, 0.0, 0.0, 0.0, 0.0]


def test_always_predict_two():
    cm = np.zeros((5, 5), int)
    cm[:, 2] = 10
    # full: row0 V=1, row1 V=1 (diag 0 -> 1 at d=1), row2 V=0, row3 V=1, row4 V=1 -> ODC = 1 - 4/5
    assert math.isclose(odc(cm, True)[0], 0.2)
    # offdiag: row0 (P01=0 -> P02=1) violation 1; rows 1-3 -> 0; row4 1
    assert math.isclose(odc(cm, False)[0], 1 - 2 / 5)


def test_bump_far_from_diagonal():
    cm = np.array([[8, 0, 2, 0, 0]] + [[0] * i + [1] + [0] * (4 - i) for i in range(1, 5)])
    full, frac = odc(cm, True)
    assert math.isclose(full, 1 - 0.2 / 5) and math.isclose(frac, 0.8)


def test_ties_are_not_violations():
    cm = np.diag([1] * 5)
    cm[0] = [4, 2, 2, 2, 0]
    assert odc(cm, True)[0] == 1.0


def test_odc_skips_empty_rows():
    cm = np.diag([5, 0, 5, 5, 5]).astype(int)
    v, frac = odc(cm, True)
    assert v == 1.0 and frac == 1.0 and not math.isnan(v)
    assert all(not math.isnan(x) for x in mean_mass_by_dist(cm))


def test_odc_all_rows_empty_is_nan():
    v, frac = odc(np.zeros((5, 5), int), True)
    assert math.isnan(v) and math.isnan(frac)


def test_confusion_and_bundle():
    t = np.array([0, 1, 2, 3, 4, 4])
    p = np.array([0, 1, 2, 3, 3, 4])
    cm = confusion_matrix(t, p, 5)
    assert cm[4, 3] == 1 and cm.sum() == 6
    probs = np.eye(5)[p]
    out = ordinal_shape_metrics(t, p, probs, 5)
    assert set(out) == {"confusion_matrix", "odc_full", "odc_offdiag", "monotone_row_frac_full",
                        "monotone_row_frac_offdiag", "mean_mass_by_dist", "ece"}
    assert math.isclose(expected_calibration_error(probs, t), 1 / 6, rel_tol=1e-6)
