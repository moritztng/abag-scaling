"""The three controls that decide whether the headlines survive a hostile read.

    leave_out      Every headline recomputed with and without the six cells folded off the
                   frozen engine tree (core.OFF_TREE_CELLS). The referee's question is not
                   "are the trees byte-identical" -- they are not, and we do not claim they
                   are -- but "does your conclusion depend on the cells you cannot vouch
                   for". A delta inside the full-panel interval answers it. (H5)

    tie_sensitivity  esmfold2's shipped selector is mean pLDDT stored on a 1e-4 grid, so
                   only about a third of its 512 values per pool are distinct and ties are
                   common. `core.rank_order` resolves ties AGAINST the selector, which is
                   conservative -- it credits the selector with the worse of two samples it
                   cannot separate, and therefore makes the gap as large as the tie
                   structure allows. That is the direction that flatters the headline, so
                   it has to be measured: this recomputes the delivered curve with ties
                   resolved FOR the selector and reports how far the headline moves.

    tail_range_null  The mechanism result (confidence stops ranking inside its own top
                   tail) compares rho over 512 samples with rho over the top 64 or 16. A
                   smaller sample in a narrower DockQ range has a smaller rho for reasons
                   that have nothing to do with confidence. The null draws a RANDOM subset
                   of the same size and reports its rho, so the tail effect is quoted
                   against the right baseline instead of against the whole pool.
"""

from __future__ import annotations

import numpy as np

import core
import q1_selection

# Statistics the leave-out must not move outside their own full-panel interval. Each is a
# path into q1_selection.analyse's output.
HEADLINES = [
    ("gap_512", lambda a: a["gap_512"]),
    ("effective_n", lambda a: a["effective_n"]),
    ("effective_n_law_slope_b", lambda a: a["effective_n_law"]["slope_b"]),
    ("gap_second_difference", lambda a: a["gap_second_difference"]),
    ("gap_256_to_512", lambda a: a["gap_per_doubling"]["256_to_512"]),
    ("delivered_delta_512_minus_8", lambda a: a["delivered_delta_from_k"]["8"]),
    ("delivered_512", lambda a: {"mean": a["user"]["mean"][-1],
                                 "lo": a["user"]["lo"][-1], "hi": a["user"]["hi"][-1]}),
    ("frac_acceptable_512",
     lambda a: {"mean": a["thresholds"]["acceptable"]["user"]["mean"][-1],
                "lo": a["thresholds"]["acceptable"]["user"]["lo"][-1],
                "hi": a["thresholds"]["acceptable"]["user"]["hi"][-1]}),
]
TAIL_FRACS = [1.0, 0.25, 0.125, 0.03125]
NULL_SEED = 20260812


def leave_out() -> dict:
    """H5 -- every headline, full panel vs the frozen-tree-only panel."""
    out = {}
    for m in core.MODELS:
        full_t = sorted(core.pools(m))
        homo_t = core.homogeneous_targets(m)
        dropped = sorted(set(full_t) - set(homo_t))
        if not dropped:
            out[m] = {"dropped": [], "n_full": len(full_t), "n_homogeneous": len(homo_t),
                      "identical": True, "headlines": {}}
            continue
        a_full = q1_selection.analyse(m, full_t)
        a_homo = q1_selection.analyse(m, homo_t)
        rows = {}
        for name, get in HEADLINES:
            f, h = get(a_full), get(a_homo)
            rows[name] = {
                "full": f, "homogeneous": h,
                "delta": h["mean"] - f["mean"],
                # The acceptance test: the leave-out point estimate sits inside the
                # full-panel interval, so the six cells move nothing a reader could see.
                "inside_full_ci": bool(f["lo"] <= h["mean"] <= f["hi"]),
            }
        out[m] = {"dropped": dropped, "n_full": len(full_t), "n_homogeneous": len(homo_t),
                  "identical": False, "headlines": rows}
    out["all_inside_ci"] = all(
        r["inside_full_ci"] for v in out.values() if isinstance(v, dict)
        for r in v.get("headlines", {}).values())
    return out


def tie_sensitivity() -> dict:
    """The delivered curve with selector ties resolved for, against and at random.

    `against` is what the analysis ships (conservative). If the three agree to well
    inside the interval, the tie structure carries no headline.
    """
    rng = np.random.default_rng(NULL_SEED)
    out = {}
    for m in core.MODELS:
        pl = core.pools(m)
        targets = sorted(pl)
        distinct, curves = [], {k: [] for k in ("against", "for", "random")}
        for t in targets:
            p = pl[t]
            s = p.selector.to_numpy()
            d = p.dockq.to_numpy()
            distinct.append(len(np.unique(s)) / len(s))
            # worst-first: ties broken so the WORSE dockq is picked (shipped), the BETTER
            # dockq is picked, and arbitrarily.
            curves["against"].append(core.curve(np.lexsort((-d, s)), d))
            curves["for"].append(core.curve(np.lexsort((d, s)), d))
            curves["random"].append(core.curve(np.lexsort((rng.random(len(s)), s)), d))
        res = {"mean_distinct_selector_frac": float(np.mean(distinct)),
               "n_targets": len(targets)}
        base = None
        for k, cs in curves.items():
            arr = np.array(cs)
            b = core.boot_means(arr)
            ci = core.ci_of(b[:, -1], arr.mean(0)[-1])
            res[f"delivered_512_ties_{k}"] = ci
            if k == "against":
                base = ci
        res["shift_for_minus_against"] = (
            res["delivered_512_ties_for"]["mean"] - base["mean"])
        res["shift_inside_ci"] = bool(
            base["lo"] <= res["delivered_512_ties_for"]["mean"] <= base["hi"])
        out[m] = res
    return out


def tail_range_null() -> dict:
    """Mechanism plus its control: rho in the top confidence tail vs a same-size random cut."""
    rng = np.random.default_rng(NULL_SEED)
    out = {}
    for m in core.MODELS:
        pl = core.pools(m)
        targets = sorted(pl)
        res = {"n_targets": len(targets), "fracs": {}}
        for f in TAIL_FRACS:
            tail, null = [], []
            for t in targets:
                p = pl[t]
                s = p.selector.to_numpy()
                d = p.dockq.to_numpy()
                k = max(3, int(round(len(s) * f)))
                top = np.argsort(-s, kind="stable")[:k]
                tail.append(core.spearman(s[top], d[top]))
                idx = rng.choice(len(s), k, replace=False)
                null.append(core.spearman(s[idx], d[idx]))
            tail = np.array(tail)
            null = np.array(null)
            res["fracs"][str(f)] = {
                "n_samples": int(round(512 * f)),
                "rho_tail": core.paired_bootstrap(np.nan_to_num(tail, nan=0.0)),
                "rho_random_same_size": core.paired_bootstrap(np.nan_to_num(null, nan=0.0)),
                "tail_minus_null": core.paired_bootstrap(
                    np.nan_to_num(tail, nan=0.0) - np.nan_to_num(null, nan=0.0)),
            }
        out[m] = res
    return out


def run() -> dict:
    return {"leave_out": leave_out(), "tie_sensitivity": tie_sensitivity(),
            "tail_range_null": tail_range_null()}


if __name__ == "__main__":
    r = run()
    print("H5 leave-out: the six off-frozen-tree cells in vs out")
    for m in core.MODELS:
        v = r["leave_out"][m]
        if v["identical"]:
            print(f"\n== {m}: no off-tree cells, nothing to drop")
            continue
        print(f"\n== {m}: {v['n_full']} -> {v['n_homogeneous']} targets, dropped "
              f"{v['dropped']}")
        for name, d in v["headlines"].items():
            flag = "ok" if d["inside_full_ci"] else "OUTSIDE CI"
            print(f"   {name:<30}{d['full']['mean']:+.4f} -> {d['homogeneous']['mean']:+.4f}"
                  f"  delta {d['delta']:+.4f}  full CI "
                  f"[{d['full']['lo']:+.4f}, {d['full']['hi']:+.4f}]  {flag}")
    print(f"\n  every headline inside its full-panel CI: {r['leave_out']['all_inside_ci']}")

    print("\nTie sensitivity (delivered at 512, ties against vs for the selector)")
    for m in core.MODELS:
        t = r["tie_sensitivity"][m]
        print(f"  {m:<14}distinct selector values {t['mean_distinct_selector_frac']:.3f}  "
              f"against {core.fmt(t['delivered_512_ties_against'])}")
        print(f"  {'':<14}for      {t['delivered_512_ties_for']['mean']:+.4f}   random "
              f"{t['delivered_512_ties_random']['mean']:+.4f}   shift "
              f"{t['shift_for_minus_against']:+.4f}  inside CI {t['shift_inside_ci']}")

    print("\nMechanism: rho(selector, DockQ) in the top tail, against a same-size random cut")
    for m in core.MODELS:
        print(f"  {m:<14}" + "".join(f"{'top ' + str(int(f * 100)) + '%':>16}"
                                     for f in TAIL_FRACS))
        a = r["tail_range_null"][m]["fracs"]
        print(f"  {'  tail':<14}" + "".join(
            f"{a[str(f)]['rho_tail']['mean']:>+16.3f}" for f in TAIL_FRACS))
        print(f"  {'  random':<14}" + "".join(
            f"{a[str(f)]['rho_random_same_size']['mean']:>+16.3f}" for f in TAIL_FRACS))
        print(f"  {'  diff':<14}" + "".join(
            f"{a[str(f)]['tail_minus_null']['mean']:>+16.3f}" for f in TAIL_FRACS))
        def _ci(f):
            d = a[str(f)]["tail_minus_null"]
            return f"[{d['lo']:+.3f},{d['hi']:+.3f}]"
        print(f"  {'  diff CI':<14}" + "".join(f"{_ci(f):>16}" for f in TAIL_FRACS))
