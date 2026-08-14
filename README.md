# abag-scaling

Sampling scales. Selection does not.

An interactive page on what 329,216 DockQ-labelled samples say about scaling the number of generated
structures in antibody-antigen co-folding: four independently trained predictors, three AF3-style
co-folders and one single-sequence folder, 512 samples per target, every sample scored against the
experimental structure.

Going from 32 samples to 512, 16x the compute, the structure the model hands you gains +0.000 DockQ
[-0.005, +0.005] while the best one in the pool gains +0.085 [+0.076, +0.095]. The reason is
measurable: confidence ranks a whole pool about as well as its headline correlation says, and stops
ranking, or inverts, inside its own top tail, the only region a selector ever operates in.

**Read it: https://moritztng.github.io/abag-scaling/**

Full statistics, every interval, every control and the limitations are in [FINDINGS.md](FINDINGS.md).

## The benchmark and the dataset

The targets are [2026ARK-AB](https://arxiv.org/abs/2607.03787), the antibody-antigen benchmark
released with OpenDDE: 164 PDB targets, 404 interfaces, 159 clusters. 161 of those targets are
scorable, and 643 of the 656 model-target cells are analysed here. **We did not assemble that target
set.** If you use this work, cite it as well.

The sample-level data is public: **[Tenstorrent/abag-xm](https://huggingface.co/datasets/Tenstorrent/abag-xm)**
under CC-BY-4.0. All 335,360 predicted structures, every DockQ label, the confidence values, the
reference structures and the alignments. Nothing on the page is derived from data held back.

## This repository

```
index.html          the page: one static file, no backend, no build step
data/insights.json  every number the page shows
analysis/           the pipeline that writes that file
tools/screenshot.js device-metric screenshots (desktop + Pixel 10 393px, light and dark)
```

Look at it locally:

```
python3 -m http.server 8899 --directory .
```

Rebuild the numbers (needs the source parquets, see FINDINGS.md for the input paths, and
pandas / pyarrow / scipy):

```
python3 analysis/build_insights.py -o data/insights.json
```

Nothing on the page is typed in by hand; it is all read from `data/insights.json`.

## Citation

```bibtex
@misc{abagxm2026,
  title  = {AbAg-XM: 335,360 DockQ-labelled antibody-antigen structure predictions
            from four models},
  author = {Th\"uning, Moritz},
  year   = {2026},
  url    = {https://huggingface.co/datasets/Tenstorrent/abag-xm},
  note   = {Analysis: https://moritztng.github.io/abag-scaling/}
}
```

Computed on Tenstorrent hardware with [tt-bio](https://github.com/moritztng/tt-bio).
