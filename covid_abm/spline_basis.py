"""Cubic B-spline basis and P-spline penalty for log R(t).

Used by abm_nets.LearnableParams when R_SPLINE=1. The basis is built once on
CPU with scipy and handed to the model as a non-trainable buffer; nothing in
this module is learned.

See plans/Task_Spline_Regularization_LogR.md sections 1-2.
"""
import numpy as np
from scipy.interpolate import BSpline


def build_bspline_basis(n_days: int, n_basis: int, degree: int = 3):
    """
    Returns (B, D2):
      B  : float64 [n_days, n_basis]        -- design matrix on the daily grid
      D2 : float64 [n_basis-2, n_basis]     -- second-difference matrix

    Equally spaced interior knots with (degree+1)-fold boundary repetition,
    the standard convention for a B-spline basis on a closed interval
    (de Boor 2001 Ch. IX; Eilers & Marx 1996 Appendix). For n_basis = J and
    degree = p there are J - p - 1 interior knots and J + p + 1 knots total.
    At J = 26 over 182 days the interior knot spacing is 181/23 = 7.87 days,
    i.e. approximately one knot per week.

    Fails loud on: n_basis too small, wrong B shape, non-finite entries,
    partition-of-unity violation (tol 1e-8), negative basis values,
    wrong D2 shape.
    """
    if n_basis < degree + 1:
        raise RuntimeError(
            f"build_bspline_basis: n_basis={n_basis} must be >= "
            f"degree+1={degree + 1}"
        )
    t_eval = np.arange(n_days, dtype=np.float64)

    n_interior = n_basis - degree - 1
    interior = np.linspace(t_eval[0], t_eval[-1], n_interior + 2)[1:-1]
    knots = np.concatenate([
        np.repeat(t_eval[0],  degree + 1),
        interior,
        np.repeat(t_eval[-1], degree + 1),
    ])

    B = BSpline.design_matrix(
        t_eval, knots, degree, extrapolate=False
    ).toarray().astype(np.float64)

    if B.shape != (n_days, n_basis):
        raise RuntimeError(
            f"build_bspline_basis: B shape {B.shape} != "
            f"expected ({n_days}, {n_basis})"
        )
    if not np.isfinite(B).all():
        raise RuntimeError("build_bspline_basis: B contains NaN or Inf")
    row_sums = B.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-8):
        bad = np.where(np.abs(row_sums - 1.0) > 1e-8)[0]
        raise RuntimeError(
            f"build_bspline_basis: partition of unity violated at {len(bad)} "
            f"days, e.g. day {bad[0]} sum={row_sums[bad[0]]:.8f}"
        )
    if (B < -1e-10).any():
        raise RuntimeError(
            "build_bspline_basis: negative B-spline values detected "
            "(basis must be non-negative everywhere)"
        )

    D2 = np.diff(np.eye(n_basis, dtype=np.float64), n=2, axis=0)
    if D2.shape != (n_basis - 2, n_basis):
        raise RuntimeError(
            f"build_bspline_basis: D2 shape {D2.shape} != "
            f"expected ({n_basis - 2}, {n_basis})"
        )
    return B, D2


def weekly_midpoint_rows(n_days: int, num_weeks: int) -> np.ndarray:
    """Row indices of week midpoints: days 3, 10, ..., 7*(num_weeks-1)+3.

    Week w spans days 7w..7w+6 in transition.py (week_id = int(t/7)), so 7w+3
    is its centre. For 26 weeks over 182 days the last index is 178, safely
    inside the grid.
    """
    rows = np.arange(num_weeks) * 7 + 3
    if rows[-1] >= n_days:
        raise RuntimeError(
            f"weekly_midpoint_rows: last midpoint {rows[-1]} outside "
            f"n_days={n_days} (num_weeks={num_weeks})"
        )
    return rows


def effective_df(B_eval: np.ndarray, D2: np.ndarray, lam: float) -> float:
    """
    EDF(lam) = trace( (B'B + lam D2'D2)^{-1} B'B ).

    Report this, not n_basis, as the flexibility of the R(t) curve. At lam=0
    EDF = n_basis (if B_eval has full column rank); as lam -> inf EDF -> 2,
    the dimension of the D2 null space.
    (Hastie & Tibshirani 1990 Ch. 3; Wood 2017 Ch. 4.)

    lam is floored at 1e-9 because D2'D2 is rank-deficient: at lam = 0 with a
    square or rank-deficient B_eval the solve is singular.
    """
    lam = max(float(lam), 1e-9)
    G = B_eval.T @ B_eval
    A = G + lam * (D2.T @ D2)
    edf = float(np.trace(np.linalg.solve(A, G)))
    if not np.isfinite(edf):
        raise RuntimeError(f"effective_df({lam}): non-finite EDF={edf}")
    return edf


def lambda_for_edf(B_eval, D2, target_edf, lo=1e-4, hi=1e6, iters=80):
    """
    Bisect on log(lambda) for the lambda giving EDF(lambda) = target_edf.

    EDF is strictly decreasing in lambda, so bisection is exact. This is the
    tuning primitive for Stages 1-3: it makes lambda comparable across
    different J, which raw lambda is not.
    """
    if effective_df(B_eval, D2, hi) > target_edf:
        raise RuntimeError(
            f"lambda_for_edf: EDF at lam={hi} is "
            f"{effective_df(B_eval, D2, hi):.2f} > target {target_edf}; "
            f"target is below the penalty null-space dimension"
        )
    if effective_df(B_eval, D2, lo) < target_edf:
        raise RuntimeError(
            f"lambda_for_edf: EDF at lam={lo} is "
            f"{effective_df(B_eval, D2, lo):.2f} < target {target_edf}; "
            f"target exceeds what n_basis={B_eval.shape[1]} can supply"
        )
    for _ in range(iters):
        mid = np.sqrt(lo * hi)
        if effective_df(B_eval, D2, mid) > target_edf:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))
