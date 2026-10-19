"""Ordinal shape metrics on the confusion matrix (ODC) and calibration (ECE).

ODC (Ordinal Decay Consistency): in each row of the row-normalised confusion
matrix P (row i = true grade i), mass should not rise while walking away from
the diagonal on either side. V_i sums every outward rise; ties cost nothing.
    ODC_full    — the walk starts at the diagonal (also penalises an off-diagonal
                  mode above the diagonal peak)
    ODC_offdiag — the walk starts one step off the diagonal
ODC = 1 − mean V_i over rows with test support.
"""
import numpy as np


def confusion_matrix(targets, preds, K: int) -> np.ndarray:
    """[K, K] integer counts, rows = true grade, columns = predicted grade."""
    cm = np.zeros((K, K), dtype=np.int64)
    np.add.at(cm, (np.asarray(targets, dtype=int), np.asarray(preds, dtype=int)), 1)
    return cm


def _row_probs(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cm = np.asarray(cm, dtype=np.float64)
    support = cm.sum(axis=1)
    present = support > 0
    P = np.zeros_like(cm)
    P[present] = cm[present] / support[present, None]
    return P, present


def _outward_rises(seq: np.ndarray) -> float:
    if len(seq) < 2:
        return 0.0
    return float(np.clip(seq[1:] - seq[:-1], 0.0, None).sum())


def odc(cm, include_diagonal_step: bool) -> tuple[float, float]:
    """(ODC, monotone_row_frac). include_diagonal_step=True is ODC_full, False is ODC_offdiag."""
    P, present = _row_probs(cm)
    K = P.shape[0]
    start = 0 if include_diagonal_step else 1
    violations = []
    for i in np.flatnonzero(present):
        left = P[i, i - start::-1] if i - start >= 0 else np.empty(0)   # P_{i,i-start}, …, P_{i,0}
        right = P[i, i + start:] if i + start < K else np.empty(0)       # P_{i,i+start}, …, P_{i,K-1}
        violations.append(_outward_rises(left) + _outward_rises(right))
    if not violations:
        return float("nan"), float("nan")
    v = np.asarray(violations)
    return float(1.0 - v.mean()), float((v == 0).mean())


def mean_mass_by_dist(cm) -> list[float]:
    """Average over rows with support of the mass at |pred − true| = Δ, Δ = 0 … K−1."""
    P, present = _row_probs(cm)
    K = P.shape[0]
    dist = np.abs(np.arange(K)[:, None] - np.arange(K)[None, :])
    if not present.any():
        return [float("nan")] * K
    return [float(P[present][dist[present] == d].sum() / present.sum()) for d in range(K)]


def expected_calibration_error(probs, targets, n_bins: int = 15) -> float:
    """Top-1 ECE with equal-width confidence bins."""
    probs = np.asarray(probs, dtype=np.float64)
    targets = np.asarray(targets, dtype=int)
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == targets).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        in_bin = (conf > lo) & (conf <= hi)
        if in_bin.any():
            ece += in_bin.mean() * abs(correct[in_bin].mean() - conf[in_bin].mean())
    return float(ece)


def ordinal_shape_metrics(targets, preds, probs, K: int) -> dict:
    cm = confusion_matrix(targets, preds, K)
    odc_full, mono_full = odc(cm, True)
    odc_off, mono_off = odc(cm, False)
    return {
        "confusion_matrix": cm.tolist(),
        "odc_full": odc_full,
        "odc_offdiag": odc_off,
        "monotone_row_frac_full": mono_full,
        "monotone_row_frac_offdiag": mono_off,
        "mean_mass_by_dist": mean_mass_by_dist(cm),
        "ece": expected_calibration_error(probs, targets),
    }
