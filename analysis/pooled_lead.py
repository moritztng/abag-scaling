"""Add the lead figure's pooled curves to an existing insights.json.

`build_insights.py` already emits them as `q1_selection.pooled_common`, so this is only
for topping up a file built before that key existed. It writes that one key and touches
nothing else, which is why it exists at all: it lets the pooled curves land without
regenerating a published file.

    python3 analysis/pooled_lead.py                       # print the curve
    python3 analysis/pooled_lead.py --merge data/insights.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import core
import q1_selection


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merge", help="insights.json to add q1_selection.pooled_common to")
    args = ap.parse_args()

    r = q1_selection.pooled(core.common_targets(core.MODELS))
    for i, k in enumerate(r["k_grid"]):
        print(f"{k:4d}  delivered {r['user']['mean'][i]:.4f} "
              f"[{r['user']['lo'][i]:.4f}, {r['user']['hi'][i]:.4f}]   "
              f"ceiling {r['oracle']['mean'][i]:.4f} "
              f"[{r['oracle']['lo'][i]:.4f}, {r['oracle']['hi'][i]:.4f}]")
    for k in ("delivered_delta_32_to_512", "ceiling_delta_32_to_512",
              "delivered_delta_16_to_512", "ceiling_delta_16_to_512"):
        print(f"{k:32s} {core.fmt(r[k])}")

    if args.merge:
        p = Path(args.merge)
        d = json.loads(p.read_text())
        d["q1_selection"]["pooled_common"] = r
        p.write_text(json.dumps(d, separators=(",", ":"), allow_nan=False))
        print(f"merged into {p} ({p.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
