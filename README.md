# abag-scaling

Sampling scales. Selection does not.

An interactive page on what 383,000 DockQ-labelled samples say about scaling the number of
generated structures in antibody-antigen co-folding: four architecturally independent predictors,
161 targets, up to 256 samples per target, every sample scored against the crystal structure.

**Read it: https://moritztng.github.io/abag-scaling/**

Full statistics, every interval, every control and the limitations are in [FINDINGS.md](FINDINGS.md).

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

Rebuild the numbers (needs the source parquets, see FINDINGS.md for the input paths):

```
python3 analysis/build_insights.py -o data/insights.json
```

Nothing on the page is typed in by hand; it is all read from `data/insights.json`.

The underlying sample-level dataset is not published here.

Computed on Tenstorrent hardware with [tt-bio](https://github.com/moritztng/tt-bio).
