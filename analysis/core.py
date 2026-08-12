"""Shared machinery for the AbAg-XM deep-N insights analysis.

Everything downstream is built from two primitives:

  * an EXACT selection-curve engine. No Monte Carlo: for a pool of n samples sorted
    worst-first by a ranking key, the probability that sorted position j (0-indexed) is
    the top-ranked member of a uniformly drawn k-subset is C(j, k-1) / C(n, k). Every
    curve in this analysis -- oracle best-of-k, the confidence selector's pick, threshold
    fractions, max epitope overlap, any candidate selector -- is that one weight matrix
    applied to a different value column.

  * `paired_bootstrap` -- B=20000 resamples over TARGETS with one shared resample matrix
    (seed 20260802, same convention as DATASHEET section 6), so CIs from different models
    and metrics are directly comparable and their differences are genuinely paired.

Data source: the frozen parquets under ~/abag_xm/deepn/dataset_n512/ (override with
ABAG_XM_DATASET). Nothing here folds, scores, or writes to the dataset.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln

DATASET = Path(os.environ.get("ABAG_XM_DATASET", Path.home() / "abag_xm/deepn/dataset_n512"))
MODELS = ["boltz2", "opendde-abag", "protenix-v2", "esmfold2"]
THRESHOLDS = [("acceptable", 0.23), ("medium", 0.49), ("high", 0.80)]
BOOTSTRAP_B = 20000
BOOTSTRAP_SEED = 20260802
TOP_RUNG = 512

# Confidence flavours. `selector` is the model's own shipped choice (top-plddt for
# esmfold2, top-confidence_score otherwise; DATASHEET section 4). esmfold2 carries only
# selector + ptm -- it has no interface head, so iptm/complex_plddt are absent by design.
FLAVORS = ["selector", "confidence_score", "ptm", "iptm", "complex_plddt"]
NUMERIC = FLAVORS + [
    "dockq", "irmsd", "lrmsd", "fnat", "interface_lddt",
    "cdr_h1_rmsd", "cdr_h2_rmsd", "cdr_h3_rmsd", "epitope_jaccard", "wall_s",
]


# --------------------------------------------------------------------------- loading


@functools.lru_cache(maxsize=None)
def load_samples(model: str) -> pd.DataFrame:
    df = pd.read_parquet(DATASET / f"{model}_samples.parquet")
    # All-null metric columns land as object dtype in the parquet; coerce so downstream
    # code sees a uniform float NaN rather than Python None.
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


@functools.lru_cache(maxsize=None)
def available(model: str, columns: tuple) -> list:
    df = load_samples(model)
    return [c for c in columns if df[c].notna().any()]


@functools.lru_cache(maxsize=None)
def load_curve(model: str) -> pd.DataFrame:
    return pd.read_parquet(DATASET / f"{model}_curve.parquet")


@functools.lru_cache(maxsize=None)
def pools(model: str, rung: int = TOP_RUNG, min_n: int = TOP_RUNG) -> dict:
    """Per-target sample pools at one rung, DockQ-labelled targets only.

    Rungs nest physically: rung 512's chunks 0-3 are the same inodes as rung 256's, so
    the rung-512 pool is a strict superset of every lower rung and subsampling inside it
    reproduces the ladder without mixing separately folded arms. Truncating each 512 pool
    back to its first 256 samples reproduces the published N=256 statistics exactly
    (635/635 targets, G3).

    Every cell in the completed panel is 512 samples deep by construction, so `min_n`
    no longer drops short rungs -- it is a tripwire. A target that fails it is a pooling
    bug, not a shallower measurement.
    """
    df = load_samples(model)
    df = df[(df.rung == rung) & df.dockq.notna()]
    return {
        t: g.reset_index(drop=True)
        for t, g in df.groupby("target", sort=True)
        if len(g) >= min_n
    }


def common_targets(models, **kw) -> list:
    return sorted(set.intersection(*[set(pools(m, **kw)) for m in models]))


# --------------------------------------------------------------------- panel bookkeeping

# The panel is complete: 164 targets x 4 models = 656 cells, every cell 512 samples deep.
# Two things still keep a cell out of the analysis, and neither is missing data:
#
#   unscorable      9ly2, 9ly3, 9lz2 -- the DockQ scorer resolves no antibody-antigen
#                   interface for them in any model, so there is nothing to score. Derived
#                   from the parquets below, not typed.
#   fold_artifact   opendde-abag 9sbb folded 8/8 clean on the Galaxy but the fold itself is a
#                   known p2-era pipeline artifact: galaxy samples sit in a pTM 0.668-0.697
#                   basin against ~0.91 for the same input refolded on qb1, DockQ 0.023 vs
#                   0.880 under the SAME fixed scorer, reproduced by a decisive N=4 refold.
#                   Excluded at source (GALAXY_EXCLUDE in the campaign labeller). A
#                   prevalence scan over 41 paired targets found it the only such case.
EXCLUDED_CELLS = {"opendde-abag": {"9sbb": "fold_artifact"}}

# Code provenance. The campaign ran one frozen engine tree so that every cell is
# comparable. Six cells -- the four largest targets, which needed memory fixes that were
# not in that tree -- were folded on two later trees, each cell entirely on one tree. The
# fixes are bit-exact at their own gates but the trees are not byte-identical, so every
# headline is reported with these cells in and out (the leave-out invariance test).
OFF_TREE_CELLS = {
    "opendde-abag": {"9i3p": "oomfix", "9ivj": "oomfix", "9q7y": "oomfix", "9j4c": "p35"},
    "protenix-v2": {"9j4c": "oomfix"},
    "esmfold2": {"9j4c": "p35"},
}


def homogeneous_targets(model: str) -> list:
    """Analysed targets of `model` minus the cells folded off the frozen engine tree."""
    return sorted(set(pools(model)) - set(OFF_TREE_CELLS.get(model, ())))


@functools.lru_cache(maxsize=None)
def panel() -> dict:
    """Every target and sample count the page states, derived from the parquets not typed.

    `samples_analysed` is what the analysis runs on: the 512-deep pools, one row per
    distinct structure. It is NOT the parquet row total -- rungs nest, so a structure
    carries one row per rung label it belongs to and the raw total double-counts.
    """
    per = {m: load_samples(m) for m in MODELS}
    all_targets = sorted(set().union(*[set(d.target.unique()) for d in per.values()]))
    labelled = {t for d in per.values() for t in d[d.dockq.notna()].target.unique()}
    unscorable = sorted(set(all_targets) - labelled)

    models = {}
    distinct = 0
    depths = set()
    for m, d in per.items():
        kept = pools(m)
        top = d[d.rung == TOP_RUNG]
        depths |= set(top.groupby("target").size().unique().tolist())
        dropped = sorted(set(all_targets) - set(kept) - set(unscorable))
        distinct += int(d.drop_duplicates(subset=["target", "hardware", "chunk", "rank"])
                        .dockq.notna().sum())
        models[m] = {
            "analysed_targets": len(kept),
            "excluded_unscorable": len(unscorable),
            "excluded_cells": {t: EXCLUDED_CELLS.get(m, {}).get(t, "unknown")
                               for t in dropped},
            "off_tree_cells": sorted(set(OFF_TREE_CELLS.get(m, ())) & set(kept)),
        }

    return {
        "panel_targets": len(all_targets),
        "panel_cells": len(all_targets) * len(MODELS),
        "cell_depths": sorted(depths),
        "scorable_targets": len(labelled),
        "unscorable_targets": unscorable,
        "per_model": models,
        "analysed_cells": sum(v["analysed_targets"] for v in models.values()),
        "off_tree_cell_count": sum(len(v["off_tree_cells"]) for v in models.values()),
        "samples_analysed": sum(len(p) for m in MODELS for p in pools(m).values()),
        "samples_distinct_labelled": distinct,
    }


# ------------------------------------------------------------------ selection weights


@functools.lru_cache(maxsize=64)
def topk_weights(n: int) -> np.ndarray:
    """`W[k-1, j]` = P(sorted position j is the argmax of a uniform k-subset of n).

    C(j, k-1) / C(n, k) in log space. Every row sums to 1.
    """
    lgf = gammaln(np.arange(n + 1) + 1.0)  # log j!
    j = np.arange(n)[None, :]
    k = np.arange(1, n + 1)[:, None]
    valid = j >= (k - 1)
    jj = np.where(valid, j, k - 1)  # dummy keeps every gammaln index in range
    log_num = lgf[jj] - lgf[(k - 1).ravel()][:, None] - lgf[jj - k + 1]
    log_den = (lgf[n] - lgf[k.ravel()] - lgf[n - k.ravel()])[:, None]
    return np.where(valid, np.exp(log_num - log_den), 0.0)


def rank_order(rank_key: np.ndarray, tiebreak: np.ndarray) -> np.ndarray:
    """Worst-first ordering by `rank_key`; ties resolved AGAINST the selector.

    A selector that cannot separate two samples is credited with the worse one, so every
    selector claim here is conservative. (Ties are vanishingly rare in practice -- all the
    confidence flavours are continuous float64 -- so this is a discipline, not a knob.)
    """
    return np.lexsort((-np.asarray(tiebreak, dtype=float), np.asarray(rank_key, dtype=float)))


def curve(order: np.ndarray, values: np.ndarray) -> np.ndarray:
    """E[value of the pick] for k = 1..n, given a worst-first `order`."""
    v = np.asarray(values, dtype=float)[order]
    return topk_weights(len(v)) @ v


def selector_curves(pool: pd.DataFrame, key: str, value: str = "dockq") -> np.ndarray:
    o = rank_order(pool[key].to_numpy(), pool[value].to_numpy())
    return curve(o, pool[value].to_numpy())


def oracle_curves(pool: pd.DataFrame, value: str = "dockq") -> np.ndarray:
    v = pool[value].to_numpy()
    return curve(np.argsort(v, kind="stable"), v)


# --------------------------------------------------------------------------- bootstrap


@functools.lru_cache(maxsize=8)
def resample_counts(n_targets: int, B: int = BOOTSTRAP_B) -> np.ndarray:
    """(B, n_targets) multiplicities of one shared target-resample draw.

    Bootstrapping means as `counts @ a / n` instead of fancy-indexing `a[idx]` keeps the
    whole B=20000 x 256-column curve bootstrap in tens of MB instead of gigabytes.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n_targets, size=(B, n_targets))
    counts = np.zeros((B, n_targets), dtype=np.float32)
    np.add.at(counts, (np.arange(B)[:, None], idx), 1.0)
    return counts


def boot_means(per_target: np.ndarray) -> np.ndarray:
    """(B, ...) bootstrap replicates of the per-target mean, shared resample draw."""
    a = np.asarray(per_target, dtype=float)
    n = a.shape[0]
    return (resample_counts(n).astype(np.float64) @ a.reshape(n, -1) / n).reshape(
        (BOOTSTRAP_B,) + a.shape[1:]
    )


def ci_of(boot: np.ndarray, point: float | np.ndarray) -> dict:
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)
    if np.ndim(point) == 0:
        return {"mean": float(point), "lo": float(lo), "hi": float(hi)}
    return {
        "mean": np.asarray(point, dtype=float).tolist(),
        "lo": np.asarray(lo).tolist(),
        "hi": np.asarray(hi).tolist(),
    }


def paired_bootstrap(per_target: np.ndarray) -> dict:
    """CI on the mean of a per-target quantity, resampling targets."""
    a = np.asarray(per_target, dtype=float)
    return ci_of(boot_means(a), a.mean(axis=0))


def ci(mean, lo, hi) -> dict:
    return {"mean": float(mean), "lo": float(lo), "hi": float(hi)}


def crosses_zero(d: dict) -> bool:
    return d["lo"] <= 0.0 <= d["hi"]


def fmt(d: dict, p: int = 4) -> str:
    return f"{d['mean']:+.{p}f} [{d['lo']:+.{p}f}, {d['hi']:+.{p}f}]"


# ------------------------------------------------------------------------- misc utils


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho with average ranks; NaN when either side is constant."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = pd.Series(x[ok]).rank().to_numpy()
    ry = pd.Series(y[ok]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def invert(curve_vals: np.ndarray, level: float) -> float:
    """Effective N: the (interpolated) k at which `curve_vals` first reaches `level`.

    curve_vals[i] is the curve at k = i+1. Returns 1.0 if the curve starts at or above
    `level`, and NaN if it never reaches it.
    """
    c = np.asarray(curve_vals, dtype=float)
    if c[0] >= level:
        return 1.0
    hit = np.nonzero(c >= level)[0]
    if len(hit) == 0:
        return float("nan")
    i = int(hit[0])
    lo, hi = c[i - 1], c[i]
    frac = 0.0 if hi == lo else (level - lo) / (hi - lo)
    return float(i + frac)  # k = i  ->  k = i+1
