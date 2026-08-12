"""Part 0 -- the data-integrity gates, measured from the parquets and rendered onto the page.

Every check here is one a referee would ask. Each returns numbers, not a verdict word, so
the page can state the measurement rather than the reassurance.

    G2  per-metric label depth per model. `labels.json` present is not every metric
        populated: DockQ and iRMSD are complete wherever a structure is scorable, the
        epitope and CDR metrics are not, so no epitope claim may be quoted at N=512.
    G4  pool semantics: every analysed cell holds exactly 512 rows at rung 512, so the
        delivered pick and the oracle are both re-derived over the whole pool and not
        over the 256 newly added samples.
    G5  seeds: the ladder is seed = base + 1000*chunk, one seed per (model, chunk), the
        four bases pairwise disjoint, and 512 distinct structures per pool.
    G6  the four largest targets (9i3p 980, 9q7y 853, 9ivj 891, 9j4c 1095 tokens) were the
        last to fold; they must carry the same depth and the same label coverage as the
        panel median, not a quietly shallower pool.

G1 (census), G3 (truncation) and G9 (code provenance) are gates on the fold tree rather
than on the parquets and are recorded in the state doc; G7 (the oracle cannot leak) is a
property of `core.selector_curves` / `core.oracle_curves` and is argued there.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import core

# Seed ladder the campaign dispatched (abag_xm_deepn_run.py). Asserted against the data,
# not trusted: a base collision would make two models share draws.
BASE_SEED = {"opendde-abag": 20000, "protenix-v2": 30000, "boltz2": 40000,
             "esmfold2": 50000}
CHUNK_SAMPLES = 64
# The four largest targets in the panel, in tokens. They folded last and they are exactly
# the cells whose engine tree differs (core.OFF_TREE_CELLS), so they get their own gate.
LARGEST = {"9j4c": 1095, "9i3p": 980, "9ivj": 891, "9q7y": 853}
SECONDARY = ["irmsd", "lrmsd", "fnat", "interface_lddt", "epitope_jaccard",
             "cdr_h1_rmsd", "cdr_h2_rmsd", "cdr_h3_rmsd"]


def label_depth(model: str) -> dict:
    """G2 -- non-null count per metric over the rung-512 rows, and the depth as a share.

    The share is what a claim may be quoted at. An epitope bar drawn from 31% of the rows
    is a bar at n=158 samples per target, not at 512.
    """
    d = core.load_samples(model)
    top = d[d.rung == core.TOP_RUNG]
    n = len(top)
    out = {"rows": n, "dockq": int(top.dockq.notna().sum())}
    for c in SECONDARY:
        out[c] = int(top[c].notna().sum())
    out["depth_frac"] = {c: (out[c] / out["dockq"] if out["dockq"] else None)
                         for c in ["dockq"] + SECONDARY}
    # Per-target mean depth, which is the number an epitope claim is actually quoted at.
    g = top.groupby("target")
    out["per_target_depth"] = {
        c: float(g[c].apply(lambda s: s.notna().sum()).mean()) for c in
        ["dockq", "interface_lddt", "epitope_jaccard", "cdr_h3_rmsd"]
    }
    return out


def pool_semantics(model: str) -> dict:
    """G4 -- every analysed cell is exactly TOP_RUNG rows, and the selector spans them all."""
    pl = core.pools(model)
    sizes = sorted({len(p) for p in pl.values()})
    # A selector re-derived over only the new half would leave the old half's confidence
    # column untouched but constant-ranked; check instead that both halves contribute to
    # the pool's own confidence ordering, i.e. the top-8 by selector is not confined to
    # one seed block.
    from_new = []
    for p in pl.values():
        top8 = p.nlargest(8, "selector")
        from_new.append(float((top8.chunk >= 4).mean()))
    return {
        "n_cells": len(pl),
        "cell_row_counts": sizes,
        "all_cells_full_depth": sizes == [core.TOP_RUNG],
        "mean_frac_of_top8_from_new_chunks": float(np.mean(from_new)),
        "cells_whose_top8_is_all_old": int(sum(1 for f in from_new if f == 0.0)),
    }


def seeds(model: str) -> dict:
    """G5 -- one seed per (model, chunk) on the declared ladder; 512 distinct structures."""
    d = core.load_samples(model)
    top = d[d.rung == core.TOP_RUNG]
    per_chunk = top.groupby("chunk").seed.unique()
    ladder = {int(c): sorted(int(x) for x in v) for c, v in per_chunk.items()}
    expected = {c: [BASE_SEED[model] + 1000 * c] for c in ladder}
    # Distinct structures per pool, keyed on the metric tuple rather than the CIF bytes:
    # two distinct structures colliding on DockQ and all the confidence fields at float64
    # does not happen.
    key = ["dockq", "selector", "ptm", "confidence_score", "complex_plddt", "iptm"]
    key = [c for c in key if top[c].notna().any()]
    dup = {}
    for t, g in top[top.dockq.notna()].groupby("target"):
        n_distinct = len(g.drop_duplicates(subset=key))
        if n_distinct != len(g):
            dup[t] = {"rows": len(g), "distinct": int(n_distinct)}
    return {
        "chunks": sorted(int(c) for c in ladder),
        "one_seed_per_chunk": all(len(v) == 1 for v in ladder.values()),
        "ladder_matches_declared": ladder == expected,
        "seed_block": [int(min(min(v) for v in ladder.values())),
                       int(max(max(v) for v in ladder.values()))],
        "cells_with_duplicate_structures": dup,
    }


def cross_model_seeds() -> dict:
    """G5 -- the four seed blocks are pairwise disjoint, so no two models share a draw."""
    blocks = {}
    for m in core.MODELS:
        d = core.load_samples(m)
        blocks[m] = set(int(s) for s in d[d.rung == core.TOP_RUNG].seed.dropna().unique())
    pairs = {}
    ms = list(core.MODELS)
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            pairs[f"{a}|{b}"] = sorted(blocks[a] & blocks[b])
    return {"per_model_seeds": {m: sorted(v) for m, v in blocks.items()},
            "pairwise_shared_seeds": pairs,
            "all_disjoint": all(not v for v in pairs.values())}


def largest_targets() -> dict:
    """G6 -- the four largest targets against the panel median, per model."""
    out = {}
    for m in core.MODELS:
        pl = core.pools(m)
        med = {}
        for c in ["dockq", "interface_lddt", "epitope_jaccard", "cdr_h3_rmsd"]:
            med[c] = float(np.median([p[c].notna().sum() for p in pl.values()])) if pl else None
        rows = {}
        for t in sorted(LARGEST, key=lambda x: -LARGEST[x]):
            if t not in pl:
                rows[t] = {"tokens": LARGEST[t], "in_analysis": False}
                continue
            p = pl[t]
            rows[t] = {
                "tokens": LARGEST[t], "in_analysis": True, "n": int(len(p)),
                "dockq": int(p.dockq.notna().sum()),
                "dockq_mean": float(p.dockq.mean()),
                "interface_lddt": int(p.interface_lddt.notna().sum()),
                "epitope_jaccard": int(p.epitope_jaccard.notna().sum()),
                "cdr_h3_rmsd": int(p.cdr_h3_rmsd.notna().sum()),
                "off_tree": t in core.OFF_TREE_CELLS.get(m, {}),
            }
        out[m] = {"panel_median_depth": med, "targets": rows}
    return out


def wall_time() -> dict:
    """The four chunk-folds whose fleet record landed at rung 256 carry no wall time.

    4 x 64 = 256 samples with a null `wall_s`; the cost model takes a per-target median of
    per-sample card-seconds, so three targets' cost basis rests on 7 of 8 chunks. Anything
    other than 256 means something else is missing.
    """
    out = {}
    nulls = 0
    for m in core.MODELS:
        d = core.load_samples(m)
        top = d[d.rung == core.TOP_RUNG]
        miss = top[top.wall_s.isna()]
        n = int(len(miss))
        nulls += n
        out[m] = {"null_wall_s_samples": n,
                  "cells": sorted({f"{t} c{int(c)}" for t, c in
                                   zip(miss.target, miss.chunk)})}
    out["total_null_wall_s"] = nulls
    return out


def run() -> dict:
    return {
        "g2_label_depth": {m: label_depth(m) for m in core.MODELS},
        "g4_pool_semantics": {m: pool_semantics(m) for m in core.MODELS},
        "g5_seeds": {m: seeds(m) for m in core.MODELS},
        "g5_cross_model": cross_model_seeds(),
        "g6_largest_targets": largest_targets(),
        "wall_time": wall_time(),
    }


if __name__ == "__main__":
    r = run()
    print("G2 label depth (rung-512 rows)")
    print(f"  {'model':<14}{'rows':>8}{'dockq':>8}{'irmsd':>8}{'if_lddt':>9}"
          f"{'epitope':>9}{'cdr_h3':>8}")
    for m, a in r["g2_label_depth"].items():
        print(f"  {m:<14}{a['rows']:>8}{a['dockq']:>8}{a['irmsd']:>8}"
              f"{a['interface_lddt']:>9}{a['epitope_jaccard']:>9}{a['cdr_h3_rmsd']:>8}")
        d = a["per_target_depth"]
        print(f"  {'':<14}per-target depth: dockq {d['dockq']:.1f}  if_lddt "
              f"{d['interface_lddt']:.1f}  epitope {d['epitope_jaccard']:.1f}  cdr_h3 "
              f"{d['cdr_h3_rmsd']:.1f}")

    print("\nG4 pool semantics")
    for m, a in r["g4_pool_semantics"].items():
        print(f"  {m:<14}{a['n_cells']:>4} cells  row counts {a['cell_row_counts']}  "
              f"full_depth={a['all_cells_full_depth']}  "
              f"top8 from new chunks {a['mean_frac_of_top8_from_new_chunks']:.3f}  "
              f"all-old cells {a['cells_whose_top8_is_all_old']}")

    print("\nG5 seeds")
    for m, a in r["g5_seeds"].items():
        print(f"  {m:<14}chunks {a['chunks']}  one_seed_per_chunk={a['one_seed_per_chunk']}"
              f"  ladder_ok={a['ladder_matches_declared']}  block {a['seed_block']}  "
              f"dup-structure cells {len(a['cells_with_duplicate_structures'])}")
    x = r["g5_cross_model"]
    print(f"  cross-model disjoint: {x['all_disjoint']}  "
          f"shared: { {k: v for k, v in x['pairwise_shared_seeds'].items() if v} }")

    print("\nG6 the four largest targets")
    for m, a in r["g6_largest_targets"].items():
        med = a["panel_median_depth"]
        print(f"  {m:<14}panel median depth dockq {med['dockq']:.0f} epitope "
              f"{med['epitope_jaccard']:.0f}")
        for t, d in a["targets"].items():
            if not d["in_analysis"]:
                print(f"    {t} ({d['tokens']} tok)  NOT IN ANALYSIS")
                continue
            print(f"    {t} ({d['tokens']} tok)  n={d['n']} dockq={d['dockq']} "
                  f"mean={d['dockq_mean']:.3f} epitope={d['epitope_jaccard']} "
                  f"off_tree={d['off_tree']}")

    w = r["wall_time"]
    print(f"\nwall_s nulls: total {w['total_null_wall_s']}")
    for m in core.MODELS:
        if w[m]["null_wall_s_samples"]:
            print(f"  {m:<14}{w[m]['null_wall_s_samples']:>5}  {w[m]['cells']}")
