# AbAg-XM deep-N: what 329,216 labelled samples say about sampling and selection

Analysis of the AbAg-XM deep-N asset (four independently trained structure predictors --
boltz2, opendde-abag and protenix-v2 are AF3-style all-atom diffusion co-folders, esmfold2 is a
single-sequence folder, on the 164-target [2026ARK-AB](https://arxiv.org/abs/2607.03787) antibody-antigen benchmark,
161 of them scorable, **512 samples per target**, every sample DockQ-labelled). The target set
is OpenDDE's, not ours. The sample-level data is public at
[Tenstorrent/abag-xm](https://huggingface.co/datasets/Tenstorrent/abag-xm) under CC-BY-4.0. Source data is frozen and unmodified; this document is the
technical backing for the site.

**The panel is complete.** 164 targets x 4 models = **656 cells, every one of them 512 samples
deep by construction**. No cell is shallower, so no headline N is an average over a ragged panel.

**Sample counts.** The analysed pools hold **329,216** samples (643 analysed cells x 512). The
four packaged parquets total more rows than that, but rungs nest, so the same physical structure
carries one row per rung label it belongs to; distinct DockQ-labelled structures in the asset are
**377,456**. Any statement quoting the raw row total as a sample count is wrong.

**The analysis denominator is 643 of 656 cells**, for two reasons and neither is missing data:

* **3 unscorable targets**, 9ly2, 9ly3, 9lz2. The DockQ scorer resolves no antibody-antigen
  interface for them in any model, so 161 of 164 targets are scorable, in every model.
* **1 excluded cell**, opendde-abag/9sbb, a root-caused p2-era pipeline artifact (galaxy samples
  in a pTM 0.668-0.697 basin against ~0.91 on a qb1 refold of the same input; DockQ 0.023 against
  0.880 under the same fixed scorer; a prevalence scan over 41 paired targets found it the only
  such case).

161 + 160 + 161 + 161 = 643.

**One correction to the panel, found while checking it.** The completeness census counted
`labels.json` *present*, which is not the same as labelled. opendde-abag/9j4c carried a
schema-valid, full-length `labels.json` in all eight of its chunks in which every one of its 512
samples was a scorer-error placeholder: the campaign labeller resolves its scorers relative to its
own location, was run from a partially-populated copy of the worktree, and **records a subprocess
failure as the per-sample label value** rather than failing the fold. The cell was rescored from
the correct tree (512/512 DockQ, chain map A:A / L:C / H:B, iRMSD populated) and is included here.
The census now counts populated values: across all 656 cells, **zero partial cells, zero short
cells, zero placeholders, zero repeated structures**.

**There is no data cut-off any more.** Every chunk the first release excluded as post-cut-off, as
a tensor-lifetime failure in our own implementation, or as a short pool is folded, scored and included.

Reproduce every number:

```
python3 analysis/build_insights.py -o data/insights.json
python3 analysis/summarise.py data/insights.json
```

Inputs: `~/abag_xm/deepn/dataset_n512/{model}_samples.parquet` and
`~/abag_xm/deepn/galaxy/fleet_results.jsonl`. Nothing is fabricated, hand-entered or carried
over from a previous document.

---

## Method

**Pools.** Rungs nest physically: rung 512's chunks 0-3 are the same inodes as rung 256's, so the
512-sample pool is a strict superset of every lower rung. All curves are computed by subsampling
inside that one pool rather than by comparing separately folded arms, which removes the
differing-target-set problem that forces DATASHEET section 4 to caveat its rows.

**Pooling gate (G3), and it is an identity test rather than a statistical one.** Truncating each
512-sample pool back to its first 256 samples must reproduce the earlier 256-sample per-target
statistics exactly. It does, on **635 targets, 0 mismatches**: boltz2 161/161, esmfold2 160/160,
opendde-abag 155/155, protenix-v2 159/159. Eight analysed cells are *unchecked* rather than
failed, having no complete 256 rung to nest against (the six cells folded at rung 512 only, plus
opendde-abag/9gvn and protenix-v2/9d73). What this gates is chunk enumeration, the completeness
arithmetic, double-counting, and above all that **the selector is re-derived over the whole 512
pool rather than over the 256 newly added samples**. Directly: 48-53% of each model's top-8 by
confidence comes from the new seed block, against 100% if the selector had only ranked the new
half.

**Seeds.** `seed = base + 1000*chunk`, bases 40000 boltz2, 20000 opendde-abag, 30000 protenix-v2,
50000 esmfold2, verified against the parquets. One seed per (model, chunk); the four blocks are
pairwise disjoint, so no two models share a draw. **The seed is a pure function of model and chunk
and is therefore shared across all 164 targets**, independence across targets comes from the
inputs differing, not the draws differing. The diffusion noise tensor is target-shaped, so no two
targets receive the same noise. 512 distinct structures in every one of the 643 analysed cells.

**Exact curves, no Monte Carlo.** For a pool of n samples ordered worst-first by any ranking key,
the probability that sorted position j is the top-ranked member of a uniformly drawn k-subset is
`C(j, k-1) / C(n, k)`. Applying that one weight matrix to different value columns gives the
oracle best-of-k curve, the confidence selector's expected pick, threshold-crossing probability,
maximum epitope overlap, and any candidate selector, all exact, at every k from 1 to 512. The
engine is verified against brute-force enumeration over all k-subsets. It reproduces the
campaign's own B=200 Monte-Carlo curves (DATASHEET section 7) to ~1e-3, so it supersedes them
without contradicting them.

**Ties** in a ranking key are resolved against the selector: a selector that cannot separate two
samples is credited with the worse one, which makes every delivered figure here the conservative
one.

That is not purely a discipline. **esmfold2's shipped selector is mean pLDDT stored on a 1e-4
grid, so only 35.4% of its 512 values per pool are distinct** (99.1% boltz2, 94.9% protenix-v2,
82.6% opendde-abag), and under one tied value DockQ can span 0.005 to 0.056. Since our tie-break
is the one that makes the gap as large as the tie structure allows, the direction that flatters
the headline, it is measured rather than argued. Delivered DockQ at 512 with ties resolved FOR
the selector instead of against: boltz2 +0.0000, opendde-abag +0.0000, protenix-v2 +0.0002,
esmfold2 **+0.0046**. The largest shift is a twelfth of that model's own interval half-width, so
the tie structure carries no headline.

**Bootstrap.** B=20000 resamples over targets, one shared resample draw (seed 20260802, the same
convention as DATASHEET section 6), so CIs from different models and metrics are comparable and
their differences are genuinely paired. Implemented as a count-matrix matmul, which keeps the
full 512-column curve bootstrap in tens of MB.

**Cost.** Per-model, per-target card-seconds come from `galaxy/fleet_results.jsonl` (seconds of
one Wormhole Galaxy chip per completed 64-sample chunk), the source DATASHEET section 8 names as
the cost authority. Median cost per sample over the four-model common target set: esmfold2
14.1 s, boltz2 17.8 s, opendde-abag 31.0 s, protenix-v2 31.0 s.

256 of the 329,216 analysed samples carry no measured wall time, because their fleet record landed
under the wrong rung label (boltz2/9ua5 c2, opendde-abag/9rye c2+c3 and 9xqn c2). The cost model
takes a per-target median, so three targets' cost basis rests on seven of their eight chunks.

> **Do not cost from the `wall_s` column in the packaged sample parquets.** It is byte-identical
> across all four models for the same (target, chunk), 631 of 631 shared chunks, and takes only
> ~90 distinct minute-quantised values. It is a packaging artifact, not a per-model fold time.
> This is a defect in the packaged asset, found here; the fleet log is unaffected.

---

## Q1. Sampling scales. Selection does not.

`oracle(k)` = expected best DockQ among k samples. `user(k)` = expected DockQ of the sample the
model's own selector returns from those k. At k=1 the two coincide, and both equal the per-target
mean DockQ, the value of drawing one sample at random.

| model | targets | random | oracle@16 | oracle@512 | delivered@16 | delivered@512 | gap@512 |
|---|---|---|---|---|---|---|---|
| boltz2 | 161 | 0.2246 | 0.3370 | 0.4480 | 0.2449 | 0.2514 | +0.1966 [+0.1654, +0.2301] |
| opendde-abag | 160 | 0.4991 | 0.5613 | 0.6211 | 0.5047 | 0.4978 | +0.1234 [+0.1034, +0.1451] |
| protenix-v2 | 161 | 0.3009 | 0.4164 | 0.5673 | 0.3201 | 0.3191 | +0.2481 [+0.2169, +0.2808] |
| esmfold2 | 161 | 0.2512 | 0.3387 | 0.4431 | 0.2795 | 0.2852 | +0.1579 [+0.1377, +0.1792] |

### H1. The delivered curve is flat past k ≈ 32, in all four models

Pre-registered before the N=512 numbers existed: `delivered(512) - delivered(k)` crosses zero for
some k ≤ 32 in at least three of four models.

| model | del(512)−del(8) | del(512)−del(16) | del(512)−del(32) | oracle gain 16→512 |
|---|---|---|---|---|
| boltz2 | +0.0098 [−0.0081, +0.0280] | +0.0065 [−0.0102, +0.0231] | +0.0046 [−0.0106, +0.0195] | +0.1110 [+0.0911, +0.1332] |
| opendde-abag | −0.0072 [−0.0148, +0.0006] | −0.0070 [−0.0138, −0.0002] | −0.0058 [−0.0117, −0.00005] | +0.0598 [+0.0484, +0.0728] |
| protenix-v2 | −0.0007 [−0.0114, +0.0097] | −0.0010 [−0.0102, +0.0082] | −0.0006 [−0.0083, +0.0076] | +0.1508 [+0.1274, +0.1759] |
| esmfold2 | +0.0097 [+0.0025, +0.0179] | +0.0057 [+0.0004, +0.0116] | +0.0030 [−0.0009, +0.0072] | +0.1044 [+0.0884, +0.1218] |

**Resolved in favour of the registered branch.** At k=8 three of four cross zero, and at k=32 the
same three do; opendde-abag's k=32 interval clears zero on the low side by 5e-5, so it is a decline
rather than a gain. Sixteen-fold more compute past k=32 buys no model a delivered gain whose
interval excludes zero, while the ceiling over the same range gains 0.060 to 0.151 DockQ.

**`k*` is reported and deliberately not headlined.** Defined as the smallest k reaching 99% of
delivered(512): boltz2 448 [2, 482], opendde-abag 1 [1, 3], protenix-v2 3 [1, 410], esmfold2
34 [8, 120]. Those intervals span nearly the whole range, because a flat curve with noise on it
crosses a fraction of its own endpoint anywhere. The delta intervals above are the statistic; k*
is an illustration.

### H2. OpenDDE-abag's delivered decline does NOT survive its registered test

Registered as a one-model directional test at k=8, decided by the interval and nothing else:
**−0.0072 [−0.0148, +0.0006], which crosses zero.** Per the pre-declared branch the claim
downgrades from "delivered quality declines with N" to **"does not improve"**, and the decline is
recorded here rather than on the page.

The k=16 and k=32 intervals do exclude zero (−0.0070 [−0.0138, −0.0002] and −0.0058 [−0.0117,
−0.00005]). **They are not promoted.** The registered comparison was k=8; moving to whichever k
clears the bar is precisely the multiplicity artifact the pre-registration exists to prevent. Both
are reported, the registered test governs.

At N ≤ 256 on the earlier panel the same contrast was −0.0079 [−0.0147, −0.0013], which is what
motivated registering the test. The effect, if it is real, is about 1.5% of that model's delivered
DockQ.

### H3. Effective N does not grow with N, so efficiency falls as 1/N

`N_eff(N)` = the k at which the oracle curve reaches `delivered(N)`. Fitting
`log2(N_eff) ~ a + b·log2(N)` over N = 2..512 gives b = 0 for an exactly flat law and b = 1 for
constant efficiency.

| model | N_eff@512 | slope b | efficiency@512 | at the floor | SE@512 |
|---|---|---|---|---|---|
| boltz2 | 1.92 [1.28, 2.90] | +0.058 [−0.012, +0.128] | 0.374% | 0% | 0.120 [0.035, 0.207] |
| opendde-abag | 1.00 [1.00, 1.59] | −0.047 [−0.063, +0.030] | 0.195% | 22% | −0.011 [−0.102, +0.081] |
| protenix-v2 | 1.56 [1.02, 2.12] | +0.007 [−0.055, +0.052] | 0.305% | 0% | 0.068 [0.002, 0.135] |
| esmfold2 | 2.65 [1.83, 3.72] | +0.110 [+0.046, +0.167] | 0.518% | 0% | 0.177 [0.100, 0.257] |

Three of four slopes contain zero. esmfold2's excludes zero at +0.110, which falls in the
pre-declared `|b| < 0.2` branch, so the law is published as "constant, or very nearly", with the
fitted exponent stated. Effective N across the grid:

| model | N=2 | 8 | 32 | 128 | 512 |
|---|---|---|---|---|---|
| boltz2 | 1.26 | 1.58 | 1.76 | 1.74 | 1.92 |
| opendde-abag | 1.17 | 1.34 | 1.26 | 1.06 | 1.00 |
| protenix-v2 | 1.36 | 1.59 | 1.58 | 1.50 | 1.56 |
| esmfold2 | 1.43 | 1.96 | 2.42 | 2.66 | 2.65 |

**Effective N stays between 1.0 and 2.7 across a 256-fold range of N.** `SE(k) = (user(k) −
random) / (oracle(k) − random)` decays with k for every model; opendde-abag's is indistinguishable
from zero at 512, i.e. no better than picking at random.

**The 1.00 entries are a floor, not a measurement.** `invert()` returns exactly 1.00 wherever the
oracle already sits at or above the delivered level at k=1. opendde-abag is on that floor for 22%
of its grid, so its fitted slope is a floor artifact and is quoted as one.

By threshold, at N=512:

| model | N_eff (≥0.23) | N_eff (≥0.49) | N_eff (≥0.80) | N_eff (mean DockQ) |
|---|---|---|---|---|
| boltz2 | 2.74 [1.60, 4.72] | 1.92 [1.00, 3.82] | 3.02 [1.46, 9.87] | 1.92 [1.28, 2.90] |
| opendde-abag | 1.40 [1.00, 3.33] | 4.05 [1.54, 22.71] | 1.00 [1.00, 2.50] | 1.00 [1.00, 1.59] |
| protenix-v2 | 1.71 [1.00, 2.79] | 3.27 [1.65, 6.10] | 1.00 [1.00, 1.17] | 1.56 [1.02, 2.12] |
| esmfold2 | 3.37 [1.85, 5.74] | 8.26 [3.77, 24.73] | 1.66 [1.00, 3.91] | 2.65 [1.83, 3.72] |

### H4. The ceiling does not saturate, but "no knee in the gap" is withdrawn

| model | gap 64→128 | 128→256 | 256→512 | gap 2nd difference | oracle per doubling |
|---|---|---|---|---|---|
| boltz2 | +0.0230 [+0.0176, +0.0291] | +0.0247 [+0.0183, +0.0316] | +0.0161 [+0.0056, +0.0266] | **−0.0085 [−0.0159, −0.0013]** | +0.0220 to +0.0233 |
| opendde-abag | +0.0134 [+0.0108, +0.0162] | +0.0130 [+0.0101, +0.0164] | +0.0121 [+0.0078, +0.0174] | −0.0009 [−0.0037, +0.0022] | +0.0114 to +0.0115 |
| protenix-v2 | +0.0330 [+0.0271, +0.0393] | +0.0322 [+0.0261, +0.0388] | +0.0265 [+0.0189, +0.0346] | **−0.0057 [−0.0104, −0.0010]** | +0.0293 to +0.0315 |
| esmfold2 | +0.0199 [+0.0163, +0.0238] | +0.0212 [+0.0169, +0.0259] | +0.0227 [+0.0174, +0.0285] | +0.0015 [−0.0011, +0.0043] | +0.0212 to +0.0220 |

The first release claimed the gap keeps widening with no knee at all. **Half of that is wrong and
is corrected here.** The gap does keep widening: every per-doubling increment is positive with an
interval excluding zero. But the gap's **second difference is negative with an interval excluding
zero in boltz2 and protenix-v2**, so the widening decelerates in two of four models and "no knee"
cannot be stated.

The mechanism is measurable and it is not oracle saturation: **the oracle's gain per doubling is
flat** (boltz2 +0.0220 / +0.0233 / +0.0227, protenix-v2 +0.0315 / +0.0313 / +0.0293). The
deceleration comes from the delivered curve ticking up over the last doubling by an amount inside
its own noise. Nothing in the measured range says sampling saturates.

---

## Q2. Confidence knows which target is easy. It does not know which sample is right.

### The mechanism: confidence stops ranking inside its own top tail

The whole-pool correlation is the wrong number to lead with, because a selector never operates on
the whole pool. Restricting Spearman(selector, DockQ) to the top of the pool by confidence:

| model | whole pool (512) | top 128 | top 64 | top 16 |
|---|---|---|---|---|
| boltz2 | +0.043 [+0.007, +0.079] | +0.024 [−0.004, +0.051] | +0.025 [−0.003, +0.052] | +0.017 [−0.027, +0.060] |
| opendde-abag | +0.033 [−0.021, +0.086] | **−0.064 [−0.104, −0.023]** | **−0.068 [−0.107, −0.030]** | −0.048 [−0.096, +0.000] |
| protenix-v2 | +0.150 [+0.091, +0.209] | +0.018 [−0.030, +0.067] | +0.016 [−0.028, +0.061] | +0.013 [−0.038, +0.064] |
| esmfold2 | +0.117 [+0.069, +0.166] | +0.036 [+0.002, +0.070] | +0.031 [−0.000, +0.062] | +0.022 [−0.020, +0.064] |

On its own that table proves nothing: a smaller sample over a narrower DockQ range has a smaller
rho for reasons that have nothing to do with confidence. **So each tail is compared against a
random subset of the same size**, drawn from the same pool. The control holds its whole-pool value
at every size (protenix-v2 +0.150 whole pool, +0.149 at n=16), so the collapse is specific to the
tail. Tail minus same-size random control:

| model | top 128 | top 64 | top 16 |
|---|---|---|---|
| boltz2 | −0.020 [−0.065, +0.023] | −0.019 [−0.067, +0.029] | −0.029 [−0.098, +0.037] |
| opendde-abag | **−0.098 [−0.150, −0.045]** | **−0.093 [−0.151, −0.037]** | −0.074 [−0.156, +0.006] |
| protenix-v2 | **−0.146 [−0.206, −0.084]** | **−0.143 [−0.204, −0.080]** | **−0.121 [−0.200, −0.040]** |
| esmfold2 | **−0.079 [−0.122, −0.037]** | **−0.088 [−0.135, −0.040]** | **−0.096 [−0.171, −0.023]** |

**Confidence ranks a whole pool about as well as its headline rho says, and stops ranking inside
its own top tail.** Three of four models exclude zero at the top 128 and top 64; boltz2 crosses
zero and has the least to lose, its whole-pool rho being only +0.043. For opendde-abag the tail rho
is **negative**: inside its own top quartile, higher confidence means a worse structure. The weak
positive whole-pool correlation is carried entirely by samples no selector will ever pick.

This is the mechanism for the flat delivered curve (H1) and for opendde-abag's effective N sitting
on the floor (H3), and it is what replaced the withdrawn H2 claim. EXPLORATORY: 4 models x 3 tail
depths, each with its own control, every cell quoted with its interval.


Spearman correlation between a confidence score and DockQ, computed two ways: *within* a target
over its 256 samples, then summarised across targets; and *across* targets between target-mean
confidence and target-mean DockQ.

| model | flavour | within-target median | within-target mean | across-target |
|---|---|---|---|---|
| boltz2 | confidence_score | +0.029 | +0.051 [+0.015, +0.088] | 0.670 |
| boltz2 | iptm | +0.049 | +0.043 [+0.007, +0.079] | 0.702 |
| boltz2 | ptm | +0.039 | +0.043 [+0.008, +0.079] | 0.656 |
| boltz2 | complex_plddt | +0.024 | +0.032 [-0.016, +0.080] | 0.537 |
| opendde-abag | confidence_score | +0.069 | +0.040 [-0.015, +0.096] | 0.774 |
| opendde-abag | iptm | **-0.021** | +0.023 [-0.032, +0.078] | 0.754 |
| opendde-abag | complex_plddt | +0.025 | +0.013 [-0.040, +0.065] | 0.595 |
| protenix-v2 | confidence_score | +0.185 | +0.157 [+0.097, +0.218] | 0.723 |
| protenix-v2 | iptm | +0.182 | +0.156 [+0.096, +0.218] | 0.737 |
| protenix-v2 | complex_plddt | -0.008 | -0.011 [-0.062, +0.040] | 0.217 |
| esmfold2 | plddt (its selector) | +0.086 | +0.116 [+0.069, +0.165] | 0.681 |
| esmfold2 | ptm | +0.153 | +0.179 [+0.131, +0.228] | 0.788 |

Across-target correlation is 0.54 to 0.79 everywhere. Within-target correlation is 0.03 to 0.18.
The scores are informative about problem difficulty and near-uninformative about which of their
own samples to hand over.

`iptm` is the score the field uses for interfaces. Its within-target median is +0.049 on boltz2
and **-0.021** on opendde-abag. It does not rank samples within a target.

Protenix-v2 is the strongest within-target ranker of the four (+0.185), which corroborates the
earlier N=12 to N=23 pilot finding that protenix-v2's confidence is the best Ab-Ag trust signal
available, now at ~700x the sample scale. It is still an order of magnitude short of usable.

**Is the failure concentrated on hard targets?** No. Median within-target rho by target-difficulty
quartile (selector flavour), hardest to easiest: boltz2 +0.045 / -0.026 / +0.028 / +0.043;
opendde-abag +0.185 / -0.068 / +0.114 / -0.041. Both fail uniformly. Protenix-v2 (+0.183 / +0.179
/ +0.506 / +0.112) and esmfold2 (+0.019 / +0.049 / +0.179 / +0.295) do rank better on easier
targets, but never well enough on the hard ones, which are the ones that matter.

**Control.** The "user" pick is genuinely the model's own shipped selector: the `selector` column
equals `confidence_score` for 100% of targets on all three co-folders, and for esmfold2 holds
plddt (which the parquet stores nowhere else). Spearman(selector, file `rank`) = -0.87 to -0.99,
i.e. `rank` is just the selector ordering, correctly excluded as an independent signal.

---

## Q3. Failure is a site-discovery failure, not a pose-refinement failure.

The per-sample predicted-vs-native epitope overlap (`epitope_jaccard`, EJ) is bimodal. The trough
between the modes, searched over the interior of the range and taken as the median of the four
per-model troughs, gives EJ* = 0.558 (per-model troughs 0.208 to 0.625). Every (model, target) is
then one of three states.

| model | solved | right site, wrong pose | never finds site | share of failures that never find the site |
|---|---|---|---|---|
| boltz2 | 93 | 20 | 44 | 69% |
| opendde-abag | 123 | 11 | 22 | 67% |
| protenix-v2 | 128 | 8 | 21 | 72% |
| esmfold2 | 96 | 18 | 43 | 70% |

Median max-EJ over the whole pool, unsolved vs solved targets: boltz2 0.380 vs 0.833,
opendde-abag 0.389 vs 0.857, protenix-v2 0.400 vs 0.899, esmfold2 0.348 vs 0.891. The separation
is clean on all four, and it needs no threshold, which the counted fraction below does.

**67% to 72% of all failures are targets where no sample ever lands on the right epitope**, at
the shared cut EJ* = 0.558 (the median of the four per-model histogram troughs). That count moves
with the cut, so the sweep is published rather than hidden: across cuts of 0.458 to 0.625 the
fraction runs 55% to 76%; over the wider grid 0.30 to 0.80 it runs 39% to 79%. The threshold-free
statement above, median best-in-pool epitope overlap 0.35-0.40 on unsolved targets against
0.83-0.90 on solved ones, carries the finding without the knob, and is what the site leads with.

**Does depth buy site discovery?** Less than it buys pose quality. Over k = 1 to 256 (boltz2):
P(at least one sample finds the site) 0.417 → 0.554, a 33% relative gain, while P(at least one
acceptable pose) goes 0.292 → 0.567, a 94% relative gain. Protenix-v2 over the same range: +32% vs
+84%. Esmfold2: +27% vs +62%. Opendde-abag is the exception, +18% vs +14%, the antibody-specialised
model gains on both at the same rate.

**A sub-claim that did NOT hold.** The relative shape of the *mean* max-EJ curve and the mean
max-DockQ curve is nearly identical for every model (boltz2, normalised gain at k=2/8/32/128:
EJ 0.15/0.42/0.65/0.88 vs DockQ 0.15/0.43/0.66/0.88). Site discovery does not visibly plateau
while pose accuracy climbs. The site-vs-pose asymmetry above is real but shows up in the
threshold-crossing rates, not in a plateau of the mean. Reported as measured.

**Seven targets carry no epitope label at all, by construction.** 9kwy, 9ly2, 9ly3, 9lz2, 9ull,
9ulm and 9ynx have no resolvable native antigen chain, so there is no native epitope set to compare
a prediction against. The scorer intersected against an empty set and wrote a real 0.0, which read
as a total miss on every sample of those targets in all four models. One cell paired a max DockQ of
0.984 with an epitope overlap of 0.000, which is what exposed it. Those values were corrected to
null on the published dataset on 2026-08-14 and are excluded here as not computable, which is why
the epitope analysis runs on 156-157 targets rather than 160-161. Three of the seven (9ly2, 9ly3,
9lz2) carry no DockQ either and were already outside this analysis. Of the 16 model-target cells
that were affected, 15 were `solved` on DockQ and never entered the failure statistics; the one that
did, protenix-v2 on 9kwy, was labelled `never finds site` purely on the artifact.

**Coverage caveat.** Of the targets that have an epitope at all, EJ labels are complete for boltz2
and opendde-abag (median depth 512 per target). For protenix-v2 and esmfold2 the epitope scorer ran
on a subset of the 64-sample chunks (median depth 384 and 320), so all four are analysed at the
deepest chunk-aligned depth covering ≥100 targets, which is 256. Missingness is chunk-aligned rather
than per-sample, and DockQ means are close between labelled and unlabelled samples (protenix-v2
0.294 vs 0.300), so it is a scorer-coverage gap and not informative missingness.

**The direction of that bias, stated correctly.** Partial coverage can only under-count site
*discovery*: a sample that did find the site but carries no EJ label is unseen, so its target is
counted as "never finds the site". The fraction is therefore **inflated** for protenix-v2 and
esmfold2, not conservative, and those two carry the two highest fractions (72%, 70%), which is
exactly what the bias would produce. An earlier draft of this document called it conservative;
that was backwards.

The control bounds it. Re-scoring the two *completely* labelled models at the partial models'
depths, keeping DockQ at full depth so the asymmetry is reproduced exactly: boltz2 0.688 at depth
64, 128 and 256 alike; opendde-abag 0.727 / 0.727 / 0.697. So depth moves the fraction by at most
0.03, and protenix-v2's 0.72 is not explained by coverage.

**Do the models fail on the same targets?** Partly. Pairwise Jaccard of failure sets ranges 0.21
(opendde-abag vs esmfold2) to 0.44 (boltz2 vs esmfold2), the two generic co-folders fail most
alike; the antibody-specialised model fails most differently. On the 156 targets where all four
carry EJ labels, only **10 are failed by all four models**, while the best single model fails 29.

---

## Q4. At matched compute, spend it on different models, not on more samples.

Per target and per budget in card-hours, a strategy assigns each model
`n_m = min(256, share / cost_m(target))` samples. The union's oracle DockQ and threshold
probabilities are computed exactly from the product of the per-model best-of-k CDFs. The
pre-declared comparison is single-model-deep against an even four-way split; no subset search
feeds it. 151 targets carry pools and fleet cost records for all four models.

Oracle mean DockQ:

| card-h / target | 0.04 | 0.08 | 0.15 | 0.5 | 1.0 | 2.5 |
|---|---|---|---|---|---|---|
| boltz2 alone | 0.307 | 0.332 | 0.353 | 0.391 | 0.414 | 0.426 |
| opendde-abag alone | 0.537 | 0.554 | 0.566 | 0.588 | 0.599 | 0.610 |
| protenix-v2 alone | 0.354 | 0.385 | 0.413 | 0.467 | 0.499 | 0.529 |
| esmfold2 alone | 0.310 | 0.329 | 0.346 | 0.379 | 0.395 | 0.401 |
| **even four-way** | **0.517** | **0.598** | **0.624** | **0.660** | **0.677** | **0.698** |

Fraction of targets reaching DockQ ≥ 0.23:

| card-h / target | 0.04 | 0.08 | 0.15 | 0.5 | 1.0 | 2.5 |
|---|---|---|---|---|---|---|
| boltz2 alone | 0.411 | 0.449 | 0.479 | 0.528 | 0.555 | 0.570 |
| opendde-abag alone | 0.699 | 0.713 | 0.723 | 0.742 | 0.750 | 0.760 |
| protenix-v2 alone | 0.500 | 0.543 | 0.580 | 0.656 | 0.707 | 0.758 |
| esmfold2 alone | 0.392 | 0.415 | 0.439 | 0.500 | 0.533 | 0.543 |
| **even four-way** | **0.689** | **0.783** | **0.809** | **0.851** | **0.874** | **0.898** |

From 0.08 card-h/target upward the even four-way split beats every single-model strategy at every
budget, on both metrics. The headline comparison:

**Four models at 0.08 card-h/target, 11.9 samples in total, about 3 per model, reach DockQ ≥
0.23 on 0.783 [0.719, 0.843] of targets. The best single model at 2.5 card-h/target, 31x the
compute and 233.8 samples, reaches 0.760 [0.692, 0.826].** Oracle mean DockQ over the same
comparison: 0.598 [0.549, 0.647] vs 0.610 [0.559, 0.662].

**Those two intervals overlap, so read the paired difference, not the two intervals.** The
bootstrap is paired, so the difference is computable, and it does not support a "beats" claim at
this budget:

| union@0.08 − best-single@2.5 | paired difference |
|---|---|
| solve rate (DockQ ≥ 0.23) | +0.0233 [-0.0292, +0.0769], crosses zero |
| oracle mean DockQ | -0.0123 [-0.0430, +0.0206], crosses zero |

At 1/31 the compute the four-way split **draws level** with the deep single model; it does not
beat it. An earlier draft said "beat them on solve rate", which the paired interval does not
license. The split does overtake the deep single model, on both metrics with intervals excluding
zero, from **0.3 card-h/target**: 1/8 the compute:

| union@0.3 − best-single@2.5 | paired difference |
|---|---|
| solve rate | +0.0738 [+0.0239, +0.1261] |
| oracle mean DockQ | +0.0353 [+0.0044, +0.0685] |

**The equal-budget comparison is the strong one, and it is unambiguous.** At the same 0.08
card-h/target, the four-way split beats every single model on both metrics, every interval
excluding zero, including against the antibody specialist:

| union@0.08 − single@0.08 | solve rate | oracle mean DockQ |
|---|---|---|
| vs boltz2 | +0.3336 [+0.2606, +0.4076] | +0.2658 [+0.2186, +0.3143] |
| vs opendde-abag | +0.0699 [+0.0272, +0.1153] | +0.0444 [+0.0174, +0.0741] |
| vs protenix-v2 | +0.2403 [+0.1807, +0.3017] | +0.2127 [+0.1684, +0.2583] |
| vs esmfold2 | +0.3677 [+0.2918, +0.4449] | +0.2691 [+0.2164, +0.3224] |

Every one of these still excludes zero at 0.15, 0.5, 1.0 and 2.5 card-h/target.

**The budget ladder starts at 0.04, not 0.02.** At 0.02 card-h/target an even four-way split buys
1.4 samples in total and 49 of the 151 targets receive none, so the union strategy is undefined
there and its point estimate was a `nanmean` over 102 targets while every other line ran on 151.
That produced the visible dip where the four-way line started below opendde-abag and crossed it.
Dropping a budget at which the strategy does not exist is not a selection of the favourable
range; every strategy is defined at every remaining budget.

**The specialisation claim, verified.** opendde-abag alone at 0.08 card-h/target (9.8 samples)
scores oracle 0.554, well above boltz2 at 2.5 card-h/target (255.3 samples, 0.426) at 31x less
compute. Architecture and training domain dominate sample count.

**And the catch, reported in full.** Nobody can currently harvest the union's ceiling. Taking the
globally highest-confidence sample across the four pools (exact computation; confidence is not
calibrated across models, which is the point) delivers 0.450 at 0.08 card-h/target, then
**declines to 0.406 as the budget grows to 2.5**, worse than simply using opendde-abag alone
(0.500 to 0.509, flat). The union has by far the highest ceiling and the worst naive delivery.
This is consistent with the earlier pre-declared cross-model consensus-confidence pilot, which
was also a null result.

---

## Q5. Nothing a user already has beats the shipped selector.

Six alternative selectors, **fixed before running them**, each built only from numbers a model
already returns alongside its samples, so any of them could be applied today at inference time
with no ground truth: `pTM`, `ipTM`, `pLDDT`, and the mean within-pool rank of (ipTM, pLDDT),
of (pTM, ipTM, pLDDT), and of all available flavours. ESMFold2 carries only two flavours, so
only `pTM` applies there.

Pre-declared bar: a candidate beats the baseline only if its delivered mean DockQ exceeds the
shipped selector's, with the paired-bootstrap CI on the **difference** excluding zero, on the
same target set, for **at least 2 of the 4 models**. The majority rule is the multiplicity
guard: six candidates across four models makes a single nominal 95% interval uninformative on
its own.

Delivered mean DockQ at N=256, change against each model's shipped selector:

| model | candidate | change vs baseline |
|---|---|---|
| boltz2 | pTM | -0.0434 [-0.0745, -0.0139] |
| boltz2 | ipTM | -0.0375 [-0.0684, -0.0083] |
| boltz2 | pLDDT | +0.0026 [-0.0092, +0.0161] |
| boltz2 | rank_mean(ipTM, pLDDT) | -0.0019 [-0.0124, +0.0082] |
| boltz2 | rank_mean(pTM, ipTM, pLDDT) | -0.0163 [-0.0343, -0.0004] |
| boltz2 | rank_mean(all flavours) | -0.0011 [-0.0117, +0.0089] |
| opendde-abag | pTM | -0.0040 [-0.0128, +0.0035] |
| opendde-abag | ipTM | -0.0035 [-0.0103, +0.0025] |
| opendde-abag | pLDDT | -0.0005 [-0.0152, +0.0146] |
| opendde-abag | rank_mean(ipTM, pLDDT) | +0.0041 [-0.0028, +0.0107] |
| opendde-abag | rank_mean(pTM, ipTM, pLDDT) | +0.0041 [-0.0027, +0.0107] |
| opendde-abag | rank_mean(all flavours) | +0.0040 [-0.0028, +0.0107] |
| protenix-v2 | pTM | -0.0197 [-0.0365, -0.0050] |
| protenix-v2 | ipTM | -0.0006 [-0.0062, +0.0054] |
| protenix-v2 | pLDDT | -0.0756 [-0.1076, -0.0459] |
| protenix-v2 | rank_mean(ipTM, pLDDT) | -0.0080 [-0.0229, +0.0055] |
| protenix-v2 | rank_mean(pTM, ipTM, pLDDT) | -0.0090 [-0.0223, +0.0028] |
| protenix-v2 | rank_mean(all flavours) | -0.0061 [-0.0152, +0.0016] |
| esmfold2 | pTM | +0.0014 [-0.0075, +0.0087] |

**NULL RESULT, as pre-declared.** Not one of the six candidates beats the baseline on a single
model, let alone two. Four of the six are significantly *worse* on at least one model. The
three rank-mean ensembles on opendde-abag come closest to a win and their intervals still cross
zero.

Read with Q1 this sharpens the finding rather than softening it. The shipped selector is
already the best of what these models expose, and it is still worth about two samples out of
256. The problem is not that a better formula over the existing scores was overlooked, none of
these scores carries enough within-target information to combine.

**Arm not run.** Ranking samples by structural agreement with the pool's modal pose needs the
CIF pools and pairwise interface-RMSD. It is not implemented here and remains open. The prior
for it is the earlier cross-model consensus-confidence pilot on this panel, also a null result.

---

## Q6. Sampling alone does not get there, and here is how much of that is the fit.

Fitting the measured k = 1..512 curves with a saturating family `y = a - b·N^(-alpha)` (asymptote
bounded by the metric's own ceiling) and a log-linear family `y = c + d·log2(N)`, then solving each
for the N at which 80% of targets carry a pose at the bar.

**The saturating family is degenerate for three of the four models' oracle-DockQ curves and for
all four threshold-fraction curves**: bounded by the physical ceiling it walks its asymptote to 1.0
and collapses into the log fit, returning no finite N. Only opendde-abag's oracle-DockQ curve
admits a non-degenerate saturating fit (a = 0.867, alpha = 0.062). So the first release's statement
"at N ≤ 256 the curves carry no evidence of saturation" is superseded by a narrower and
better-supported one: **within a measured range now twice as long, the threshold-fraction curves
still admit no saturating fit.**

Because that range doubled, the fit reports **its own sensitivity to where the data stops**, the
same families fitted on 1..256 and on 1..512:

| model | bar | measured @512 | N for 80% (fit 1..512) | card-h/target | same, fit 1..256 | ratio |
|---|---|---|---|---|---|---|
| boltz2 | ≥0.23 | 60.2% | 4.34e4 | 212 | 3.55e4 | 1.22x |
| boltz2 | ≥0.49 | 46.6% | 9.79e6 | 4.78e4 | 1.03e7 | 0.95x |
| opendde-abag | ≥0.23 | 79.4% | 877 | 7.5 | 1.10e3 | 0.80x |
| opendde-abag | ≥0.49 | 68.1% | 3.98e6 | 3.4e4 | 3.11e6 | 1.28x |
| protenix-v2 | ≥0.23 | 81.4% | 420 | 3.4 | 448 | 0.94x |
| protenix-v2 | ≥0.49 | 65.8% | 5.94e3 | 48 | 8.22e3 | 0.72x |
| esmfold2 | ≥0.23 | 62.1% | 1.53e4 | 60 | 2.44e4 | 0.63x |
| esmfold2 | ≥0.49 | 40.4% | 6.44e13 | 2.52e11 | 3.74e14 | **0.17x** |

**The acceptable band survives doubling the fit range** (ratios 0.63 to 1.22) and is quotable.
protenix-v2 reaches 80% at ≥0.23 *inside* the measured range (81.4% at 512), so its 420 is a
measured crossing rather than an extrapolation; opendde-abag at 79.4% is a whisker short of it.

**The medium band does not survive it.** esmfold2 moves by a factor of 5.8 between the two fit
ranges, so **no single N is quoted for esmfold2 at ≥0.49**, the page prints "unstable" in that
cell instead of a number. That is the honest answer to "you extrapolated 10^14 samples from a curve
that stops at 256": the pre-declared rule is that where the two fit ranges disagree by more than
about 2x, no number is printed.

Every entry beyond N = 512 is an extrapolation past the measured range, under a family that assumes
no ceiling, and is labelled DERIVED wherever it appears.

---

## Q7. You cannot get interface accuracy and loop accuracy from the same sample.

Deep sampling does improve CDR-H3: best-of-k H3 RMSD falls from 1.37 Å at k=1 to 0.77 Å at k=256
on boltz2, 1.02 → 0.70 Å on opendde-abag. But within a target's pool, DockQ and H3 accuracy are
essentially uncorrelated, median Spearman(DockQ, -H3 RMSD) is +0.061 (boltz2), +0.067
(opendde-abag), +0.066 (protenix-v2), -0.007 (esmfold2), and fewer than 2.5% of targets exceed
rho = 0.5 on any model.

Taking the DockQ-best sample instead of the H3-best sample costs, in mean H3 RMSD: boltz2 +0.45
[+0.38, +0.53] Å, opendde-abag +0.30 [+0.20, +0.41] Å, protenix-v2 +0.38 [+0.27, +0.50] Å,
esmfold2 +0.46 [+0.35, +0.57] Å. If you need both the interface and the loop, you need two
different samples from the pool, and no available signal tells you which.

---

## Limitations

- **643 of 656 cells.** 9ly2 / 9ly3 / 9lz2 are 3-way Ab:Ag hetero-hexamers whose interface the
  DockQ scorer does not resolve, so they carry no DockQ labels in any model. One further cell is
  excluded. The arithmetic closes exactly, and nothing is dropped for depth any more:

  | model | analysed | 161 scorable minus |
  |---|---|---|
  | boltz2 | 161 |, |
  | opendde-abag | 160 | 1 mis-fold (9sbb) |
  | protenix-v2 | 161 |, |
  | esmfold2 | 161 |, |

  The four large-target exclusions (9i3p, 9ivj, 9j4c, 9q7y), which did not fit until we fixed how long our implementation held intermediate tensors alive, and the five short pools (9ua5, 9rye,
  9gvn, 9xqn, 9d73) that the first release carried are **gone**: every one of them folds and scores.

- **One exclusion remains, opendde-abag/9sbb**, as a p2-era pipeline mis-fold: the Galaxy samples
  sit in a pTM 0.668-0.697 basin while the same input refolded on qb1 reaches ~0.91, giving DockQ
  0.023 against 0.880 under the same fixed scorer. The model's own confidence condemns the Galaxy
  fold, so this is not a quality-based cherry-pick. A prevalence scan over 41 paired targets found
  it the only such case (next worst |delta| < 0.2).

- **Code provenance, which is the attack we would make on this ourselves.** No cell in this panel
  was folded by tt-bio as it stands on main today: the campaign deliberately ran one frozen engine
  tree so every cell is comparable, and main today cannot fold the four largest targets at all.
  **Six of the 656 cells were folded on two later trees**, opendde-abag 9i3p / 9ivj / 9q7y and
  protenix-v2 9j4c on one, esmfold2 9j4c and opendde-abag 9j4c on the other, because those targets
  needed memory fixes that were not in the frozen tree. Each cell is single-engine: all eight of its
  chunks came from one tree, so no *pool* is internally inhomogeneous.

  The six are exactly the four largest targets (853-1095 tokens), so tree is perfectly confounded
  with size there and cannot be untangled by looking at those cells. **We do not claim the trees are
  numerically equivalent.** The differing files include `esmfold2.py`, `opendde.py`, `protenix.py`
  and `tenstorrent.py`, which are the forward paths those cells ran; the individual fixes are
  bit-exact at their own gates, but a per-commit bit-exactness claim is not a whole-tree equivalence
  claim, and the same lineage records an unexplained sensitivity of opendde diffusion trajectories
  to bit-level perturbations upstream.

  What we do instead is **leave-out invariance**: every headline recomputed with and without those
  six cells, both published.

  | model | targets | largest headline delta | all inside full-panel CI |
  |---|---|---|---|
  | opendde-abag | 160 → 156 | +0.0054 (delivered@512) | yes |
  | protenix-v2 | 161 → 160 | +0.0197 (effective N) | yes |
  | esmfold2 | 161 → 160 | −0.0592 (effective N) | yes |
  | boltz2 | no off-tree cells |, |, |

  All 8 headlines x 3 affected models sit inside their own full-panel intervals. The largest
  movement in any DockQ-valued headline is 0.0054 against an interval half-width of 0.054, a tenth
  of the noise. **Concession:** the later tree's exact commit is not recoverable, because its
  `engine_commit.txt` was inherited when the tree was copied and is stale.

- **The five single-chunk refolds are not a confound.** opendde 9d73 / 9gvn / 9xqn and protenix
  9d73 / 9ssm were re-run on the **frozen** tree; what changed for them was the fold runner's
  watchdog constants, which are host-side timeouts and touch no numerics. Worth saying explicitly,
  because "refolded later" reads like a confound.

- **The oracle is an upper bound computed with ground truth.** That is what oracle-best-of-N means
  in this literature, and it cannot leak into the delivered number: the two curves share only the
  order-statistic weight matrix, differ only in the ordering key, and the single tiebreak that
  touches both resolves against the selector.

- **Epitope and CDR labels are chunk-partial** for protenix-v2 and esmfold2, and doubling N did not
  double them. Mean samples labelled per target, out of 512: interface-lDDT 499 / 499 / 368 / 341
  and CDR-H3 412 / 496 / 324 / **94** (boltz2 / opendde-abag / protenix-v2 / esmfold2). Every
  epitope and CDR claim is quoted at its own depth, never at 512; esmfold2's CDR-H3 result is a
  claim at n ≈ 94.

- **esmfold2's selector is quantised** to a 1e-4 grid, so only 35.4% of its 512 values per pool are
  distinct (see Method). Resolving ties the other way moves its delivered DockQ by +0.0046.

- **Seeds are per (model, chunk) and therefore shared across targets** (see Method). Independence
  across targets comes from the inputs differing.

- **256 of 329,216 analysed samples carry no measured wall time**, so three targets' cost basis
  rests on seven of their eight chunks. **`wall_s` in the packaged parquets is not a per-model
  cost** either (see Method); costs here come from the fleet log.

- **Four specific models at specific settings.** esmfold2 is single-sequence throughout;
  opendde-abag is antibody-specialised and its advantage here should not be read as a general
  co-folding result. A different sampler, temperature or MSA depth is a different experiment.

- **Single hardware.** All 512-sample pools were folded on the Wormhole Galaxy. Cross-hardware
  consistency was gated at N=16 and N=64; the residual Wormhole/Blackhole difference is chaotic
  amplification of reduction-order numerics, reproduced on-Galaxy by an mps 1→5 control.

- **Q6 extrapolates past the measured range**, under a family that assumes no ceiling, and reports
  its own fit-range sensitivity so a reader can see how much of the answer is where the data stops.

- **N = 512 is a decision cap, not a measured knee.** The oracle's gain per doubling is flat right
  to the cap. Nothing here says sampling saturates.

- **Multiple comparisons, declared.** Pre-registered before the N=512 numbers existed: H1 (4 models
  x 1 statistic), H2 (1 directional test, 1 model), H3 (4 fits), H4 (4 models x 4 statistics), H5
  (8 headlines x 3 models). The tail table is 4 x 3 exploratory cells plus 4 x 3 controls. Two
  checks were added during execution and are declared post-hoc: the tie-resolution sensitivity and
  the shrinking-range null. **No Bonferroni is applied and none is claimed**; every number is quoted
  with its interval, and where a registered test failed the claim came down (H2) or was corrected
  (H4).

- **Not a published-ranker benchmark**, for two different reasons that an earlier draft merged
  into one. The **PAE-derived** scores, pDockQ2, ipSAE, and AntiConf, which is built on pTM plus
  pDockQ2 and so inherits the requirement, genuinely cannot be computed from this asset: PAE was
  not written (every `labels.json` carries `pae_metrics: {"_skipped": "pae=False ptm=True"}`).
  The **learned rankers**: DeepRank-Ab, a geometric-deep-learning scoring function over the 3D
  structures, and ABAG-Rank, a learning-to-rank model, need no PAE and could have been run
  against these pools. They were not. Saying they "need PAE" is wrong. Whether a learned ranker
  closes the gap is untested here.

## Prior work, and what this adds

The headline direction is **not new and must not be presented as new.** Published work on
antibody-antigen complexes already reports both halves of it:

- Fromm et al., *Evaluating deep learning based structure prediction methods on antibody-antigen
  complexes* (Bioinformatics, 2026; vol 42 issue 4, btag136) reports that every method improves
  roughly linearly with the logarithm of sample count, AF3's best-of-N mean DockQ rising from
  below 0.3 to above 0.5 by 200 samples, and names identifying the best model among the
  generated ones as the crucial remaining bottleneck. That is Q1's direction and Q6's log-linear
  shape, already in the literature. Their benchmark is **110** targets; this panel is 161.
- The OpenDDE technical report gives ranked-vs-oracle gaps on its own benchmarks
  (FoldBench-AB 70.0% ranked vs 81.9% oracle; 2026ARK-AB 66.4% vs 80.1%), measured at its
  **default N = 5**; this panel runs the same contrast out to N = 512.
- Confidence-based ranking for these complexes has its own literature (AntiConf, pDockQ2,
  ipSAE, and learned rankers such as ABAG-Rank and DeepRank-Ab).
- Smorodina & Greiff (2026) show co-folding confidence is near-random at separating cognate
  from non-cognate pairs, a specificity result, adjacent to but distinct from the
  pose-quality question here.

Against that baseline, what this asset supports that those do not:

1. **Effective N as a scaling law rather than a number.** Inverting the oracle curve at the
   delivered accuracy turns "there is a gap" into "512 samples plus the model's own confidence is
   worth 1.0 to 2.7 samples chosen perfectly", and, because the doubled range lets us fit it,
   into the stronger statement that **effective N does not grow with N at all** (log-log slope
   +0.007 to +0.110), so selection efficiency falls as 1/N and reaches 0.2-0.5% at N=512. One N
   gives a number; a 256-fold range gives a law.
2. **Where in the confidence distribution ranking fails, with the control that makes it a
   result.** The same score ranks across targets (0.54-0.79) far better than within one
   (0.03-0.15), and the within-target signal is carried **entirely outside the region a selector
   uses**: restricted to the top quartile it falls to zero or inverts, against a same-size random
   subset that holds its whole-pool value. That is a mechanism, not an aggregate deficit.
3. **A mechanism for the failures.** 67% to 72% of all failures never place a sample on the
   right epitope, so most of the missing accuracy is not a ranking problem at all and no
   selector could recover it.
4. **Diversity priced against depth at measured compute.** Card-hour-matched, four models at
   about twelve samples total draw level with the best single model at 234 samples, and beat
   every single model at the same spend.
5. **A pre-declared null on fixing selection with what is already there.** Six candidate
   selectors, fixed in advance, none of which beats the shipped selector on any model.

Points 1-5 are what needs this asset, four independently trained generators, N to 512, 329,216
DockQ-labelled samples analysed, one panel of 161 targets throughout, every cell 512 deep. The
scale is the enabler, not the claim.

**How independent are the four, really?** Three of them (boltz2, opendde-abag, protenix-v2) are
AF3-style all-atom diffusion co-folders trained largely on the PDB; only esmfold2 is a different
kind of model. Calling them "architecturally independent", as an earlier draft did, overstates
it. The measured independence is partial and is reported as such in Q3: pairwise failure-set
Jaccard 0.21 to 0.56, and 10 of the 117 common targets are failed by all four. That partial
independence is exactly what Q4's compute-split result trades on.
