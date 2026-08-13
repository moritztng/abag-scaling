"""Q1 -- sampling scales, selection does not.

For every model we build, from the 512-sample pools, the exact expected curves

    oracle(k)  = E[best DockQ among k uniformly drawn samples]
    user(k)    = E[DockQ of the sample the model's own selector picks out of those k]
    random     = oracle(1) = user(1) = the per-target mean DockQ

and read off, in the order the page states them:

    delivered saturation  k* = the smallest k at which user(k) is within 1% of user(512),
                          and the paired interval on user(512) - user(k). This is the
                          user-facing result: does the structure you receive keep getting
                          better when you draw more? (H1, and H2 for opendde-abag.)
    effective N           N_eff(N) = the k at which the ORACLE curve reaches user(N)
                          -- "drawing N and trusting confidence delivers what a perfect
                          chooser would have got from N_eff samples". Fitted against N in
                          log-log: slope b = 0 means efficiency falls exactly as 1/N. (H3)
    selection efficiency  SE(k) = (user(k) - random) / (oracle(k) - random)
    the gap               oracle(k) - user(k), its growth per doubling and its second
                          difference: a knee would show as a negative second difference. (H4)
    threshold fractions   P(delivered DockQ >= 0.23 / 0.49 / 0.80), oracle and user.

`analyse` takes an explicit target list so the same code computes the full panel and the
leave-out panel that drops the cells folded off the frozen engine tree (see robustness.py).
"""

from __future__ import annotations

import numpy as np

import core
from core import THRESHOLDS, TOP_RUNG

# k grid shipped to the site (log-ish); the internal curves are exact at every k.
KGRID = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
# Doublings used for the per-doubling gap growth and its second difference.
DOUBLINGS = [(64, 128), (128, 256), (256, 512)]
# N grid the effective-N law is fitted over. N=1 is excluded: N_eff(1) = 1 by construction
# for every model, so including it forces the fit through a point that carries no data.
LAW_GRID = [2, 4, 8, 16, 32, 64, 128, 256, 512]
# k* tolerance: "within 1% of the top" is a stated convention, not a fitted threshold.
KSTAR_TOL = 0.99


def _per_target(model, targets):
    """(nt, 512) oracle / user DockQ curves plus the threshold indicator curves."""
    pl = core.pools(model)
    oracle = np.empty((len(targets), TOP_RUNG))
    user = np.empty_like(oracle)
    thr = {name: {"oracle": np.empty_like(oracle), "user": np.empty_like(oracle)}
           for name, _ in THRESHOLDS}
    for i, t in enumerate(targets):
        p = pl[t]
        d = p.dockq.to_numpy()
        o_ord = np.argsort(d, kind="stable")
        u_ord = core.rank_order(p.selector.to_numpy(), d)
        oracle[i] = core.curve(o_ord, d)
        user[i] = core.curve(u_ord, d)
        for name, cut in THRESHOLDS:
            hit = (d >= cut).astype(float)
            thr[name]["oracle"][i] = core.curve(o_ord, hit)
            thr[name]["user"][i] = core.curve(u_ord, hit)
    return oracle, user, thr


def _effective_n(oracle_curves: np.ndarray, level: np.ndarray) -> np.ndarray:
    """Vectorised inverse of a monotone-increasing curve, one level per row."""
    reached = oracle_curves >= level[:, None]
    i = reached.argmax(axis=1)
    i = np.where(reached.any(axis=1), i, oracle_curves.shape[1] - 1)
    prev = np.maximum(i - 1, 0)
    lo = np.take_along_axis(oracle_curves, prev[:, None], 1).ravel()
    hi = np.take_along_axis(oracle_curves, i[:, None], 1).ravel()
    frac = np.where(hi > lo, (level - lo) / np.where(hi > lo, hi - lo, 1.0), 0.0)
    return np.where(i == 0, 1.0, prev + 1 + np.clip(frac, 0, 1))


def _k_star(user_curves: np.ndarray, tol: float = KSTAR_TOL) -> np.ndarray:
    """Smallest k with user(k) >= tol * user(TOP_RUNG), one per row.

    The curve is not guaranteed monotone -- for opendde-abag it turns over -- so this is
    the first crossing, which is the honest reading of "past here it stops improving".
    """
    reached = user_curves >= tol * user_curves[:, -1:]
    return reached.argmax(axis=1) + 1.0


def _law_slope(neff: np.ndarray, grid: list) -> np.ndarray:
    """OLS slope of log2(N_eff) on log2(N), one per row of `neff` (rows x len(grid)).

    b = 0 is the 1/N law: effective N does not grow with N, so the share of the draws that
    reach the user falls exactly as 1/N. b = 1 would mean efficiency is constant.
    """
    x = np.log2(np.asarray(grid, dtype=float))
    y = np.log2(np.asarray(neff, dtype=float))
    xc = x - x.mean()
    return (y - y.mean(axis=-1, keepdims=True)) @ xc / (xc @ xc)


def analyse(model: str, targets: list) -> dict:
    oracle, user, thr = _per_target(model, targets)
    b_or, b_us = core.boot_means(oracle), core.boot_means(user)
    m_or, m_us = oracle.mean(0), user.mean(0)
    rnd, b_rnd = m_or[0], b_or[:, 0]

    # SE is undefined at k=1, where oracle == user == random by construction.
    with np.errstate(divide="ignore", invalid="ignore"):
        se = (m_us - rnd) / (m_or - rnd)
        b_se = (b_us - b_rnd[:, None]) / (b_or - b_rnd[:, None])
    se[0] = np.nan
    b_se[:, 0] = np.nan
    neff = _effective_n(m_or[None, :], np.array([m_us[-1]]))[0]
    b_neff = _effective_n(b_or, b_us[:, -1])

    gi = [k - 1 for k in KGRID]
    out = {
        "model": model,
        "n_targets": len(targets),
        "k_grid": KGRID,
        "random_baseline": core.ci_of(b_rnd, rnd),
        "oracle": core.ci_of(b_or[:, gi], m_or[gi]),
        "user": core.ci_of(b_us[:, gi], m_us[gi]),
        "selection_efficiency": core.ci_of(b_se[:, gi[1:]], se[gi[1:]]),
        "selection_efficiency_k": KGRID[1:],
        "effective_n": core.ci_of(b_neff, neff),
        "gap_512": core.ci_of(b_or[:, -1] - b_us[:, -1], m_or[-1] - m_us[-1]),
        # "the gap widens" as a statistic rather than the shape of two lines: how much
        # bigger the oracle-minus-delivered gap is at 512 than at 16, paired over targets.
        "gap_widening_16_to_512": core.ci_of(
            (b_or[:, -1] - b_us[:, -1]) - (b_or[:, 15] - b_us[:, 15]),
            (m_or[-1] - m_us[-1]) - (m_or[15] - m_us[15])),
        "user_gain_16_to_512": core.ci_of(b_us[:, -1] - b_us[:, 15], m_us[-1] - m_us[15]),
        "oracle_gain_16_to_512": core.ci_of(b_or[:, -1] - b_or[:, 15], m_or[-1] - m_or[15]),
        "thresholds": {},
    }

    # Both curves measured against their own single-draw baseline. This is the pair the
    # headline is about -- "what did drawing more buy?" -- and unlike the raw levels it is a
    # PAIRED quantity, so its interval is the interval of the claim rather than the much
    # wider spread of per-target difficulty.
    out["user_gain_from_1"] = core.ci_of(b_us[:, gi] - b_us[:, :1], m_us[gi] - m_us[0])
    out["oracle_gain_from_1"] = core.ci_of(b_or[:, gi] - b_or[:, :1], m_or[gi] - m_or[0])

    # H1 -- delivered saturation. k* plus the paired interval on how much the last
    # doublings actually bought the user.
    out["k_star"] = core.ci_of(_k_star(b_us), _k_star(m_us[None, :])[0])
    out["k_star_tol"] = KSTAR_TOL
    out["delivered_delta_from_k"] = {
        str(k): core.ci_of(b_us[:, -1] - b_us[:, k - 1], m_us[-1] - m_us[k - 1])
        for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]
    }

    # H3 -- the effective-N law. N_eff(N) for each N on the grid, then the log-log slope.
    neff_grid = np.stack([_effective_n(m_or[None, :], np.array([m_us[n - 1]]))[0]
                          for n in LAW_GRID])
    b_neff_grid = np.stack([_effective_n(b_or, b_us[:, n - 1]) for n in LAW_GRID], axis=1)
    out["effective_n_law"] = {
        "n_grid": LAW_GRID,
        "n_eff": core.ci_of(b_neff_grid, neff_grid),
        "efficiency": core.ci_of(b_neff_grid / np.array(LAW_GRID, dtype=float),
                                 neff_grid / np.array(LAW_GRID, dtype=float)),
        "slope_b": core.ci_of(_law_slope(b_neff_grid, LAW_GRID),
                              _law_slope(neff_grid[None, :], LAW_GRID)[0]),
        # invert() returns exactly 1.0 when the oracle already sits at or above the
        # delivered level at k=1. Where that floor binds, N_eff is a floor and not a
        # measurement, and a slope fitted through it is a floor artifact.
        "frac_at_floor": float((neff_grid <= 1.0).mean()),
    }

    # H4 -- no knee. Gap growth per doubling and its second difference; a knee is a
    # negative second difference.
    gap = m_or - m_us
    b_gap = b_or - b_us
    out["gap_per_doubling"] = {
        f"{a}_to_{b}": core.ci_of(b_gap[:, b - 1] - b_gap[:, a - 1], gap[b - 1] - gap[a - 1])
        for a, b in DOUBLINGS
    }
    (a1, b1), (a2, b2) = DOUBLINGS[-2], DOUBLINGS[-1]
    out["gap_second_difference"] = core.ci_of(
        (b_gap[:, b2 - 1] - b_gap[:, a2 - 1]) - (b_gap[:, b1 - 1] - b_gap[:, a1 - 1]),
        (gap[b2 - 1] - gap[a2 - 1]) - (gap[b1 - 1] - gap[a1 - 1]))
    out["oracle_gain_per_doubling"] = {
        f"{a}_to_{b}": core.ci_of(b_or[:, b - 1] - b_or[:, a - 1], m_or[b - 1] - m_or[a - 1])
        for a, b in DOUBLINGS
    }
    for name, cut in THRESHOLDS:
        o, u = thr[name]["oracle"], thr[name]["user"]
        bo, bu = core.boot_means(o), core.boot_means(u)
        out["thresholds"][name] = {
            "cut": cut,
            "oracle": core.ci_of(bo[:, gi], o.mean(0)[gi]),
            "user": core.ci_of(bu[:, gi], u.mean(0)[gi]),
            "effective_n": core.ci_of(
                _effective_n(bo, bu[:, -1]),
                _effective_n(o.mean(0)[None, :], np.array([u.mean(0)[-1]]))[0],
            ),
        }
    return out


def pooled(targets: list) -> dict:
    """The lead figure's two curves: delivered and ceiling, averaged over the four models.

    The average is taken per target and bootstrapped after that, not assembled from the four
    published per-model intervals -- averaging four intervals is not an interval on their mean.
    Same estimator, same shared resample draw as everything else here.
    """
    per = {m: _per_target(m, targets)[:2] for m in core.MODELS}
    oracle = np.mean([per[m][0] for m in core.MODELS], axis=0)
    user = np.mean([per[m][1] for m in core.MODELS], axis=0)
    gap = oracle - user
    gi = [k - 1 for k in KGRID]
    out = {
        "n_targets": len(targets),
        "n_models": len(core.MODELS),
        "k_grid": KGRID,
        "oracle": core.ci_of(core.boot_means(oracle)[:, gi], oracle.mean(0)[gi]),
        "user": core.ci_of(core.boot_means(user)[:, gi], user.mean(0)[gi]),
        "gap": core.ci_of(core.boot_means(gap)[:, gi], gap.mean(0)[gi]),
    }
    # The within-range claim is paired, so it carries its own paired interval rather than
    # being read off two overlapping marginal bands.
    for k0 in (16, 32):
        for name, a in (("delivered", user), ("ceiling", oracle)):
            d = a[:, core.TOP_RUNG - 1] - a[:, k0 - 1]
            out[f"{name}_delta_{k0}_to_{core.TOP_RUNG}"] = core.paired_bootstrap(d)
    return out


def run() -> dict:
    per_model = {m: analyse(m, sorted(core.pools(m))) for m in core.MODELS}
    common = core.common_targets(core.MODELS)
    return {
        "per_model": per_model,
        "common_targets": common,
        "per_model_common": {m: analyse(m, common) for m in core.MODELS},
        "pooled_common": pooled(common),
    }


if __name__ == "__main__":
    r = run()
    for m in core.MODELS:
        a = r["per_model"][m]
        g = dict(zip(a["k_grid"], a["oracle"]["mean"]))
        u = dict(zip(a["k_grid"], a["user"]["mean"]))
        se = dict(zip(a["selection_efficiency_k"], a["selection_efficiency"]["mean"]))
        law = a["effective_n_law"]
        print(f"\n== {m}  ({a['n_targets']} targets)  random={a['random_baseline']['mean']:.4f}")
        print("  k          8      16      64     256     512")
        print(f"  oracle  {g[8]:.4f}  {g[16]:.4f}  {g[64]:.4f}  {g[256]:.4f}  {g[512]:.4f}")
        print(f"  user    {u[8]:.4f}  {u[16]:.4f}  {u[64]:.4f}  {u[256]:.4f}  {u[512]:.4f}")
        print(f"  SE      {se[8]:.3f}   {se[16]:.3f}   {se[64]:.3f}   {se[256]:.3f}   "
              f"{se[512]:.3f}")
        print(f"  H1 k*={core.fmt(a['k_star'], 1)}  "
              f"delivered(512)-delivered(8)={core.fmt(a['delivered_delta_from_k']['8'])}")
        print(f"     delivered(512)-delivered(16)={core.fmt(a['delivered_delta_from_k']['16'])}"
              f"  -(32)={core.fmt(a['delivered_delta_from_k']['32'])}")
        print(f"  H3 N_eff@512={core.fmt(a['effective_n'], 2)}  slope b="
              f"{core.fmt(law['slope_b'], 3)}  frac_at_floor={law['frac_at_floor']:.2f}")
        print("     N_eff " + "  ".join(
            f"{n}:{v:.2f}" for n, v in zip(law["n_grid"], law["n_eff"]["mean"])))
        print(f"  H4 gap@512={core.fmt(a['gap_512'])}  2nd diff="
              f"{core.fmt(a['gap_second_difference'])}")
        for k, v in a["gap_per_doubling"].items():
            print(f"     gap {k:<12}{core.fmt(v)}   oracle "
                  f"{core.fmt(a['oracle_gain_per_doubling'][k])}")
