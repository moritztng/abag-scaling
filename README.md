# abag-scaling

Sampling scales. Selection does not.

An interactive page on what 329,000 DockQ-labelled samples say about scaling the number of
generated structures in antibody-antigen co-folding: four independently trained predictors —
three AF3-style co-folders and one single-sequence folder — on a 161-target panel, 512 samples
per target, every sample scored against the experimental structure.

Past about 32 samples the structure the model hands you stops improving, while the best one in the
pool keeps getting better at a steady rate to the cap. Effective N stays between 1 and 3 across a
256-fold range of N, so the share of your draws that reaches you falls as 1/N. The reason is
measurable: confidence ranks a whole pool about as well as its headline correlation says, and stops
ranking, or inverts, inside its own top tail — the only region a selector ever operates in.

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

Rebuild the numbers (needs the source parquets, see FINDINGS.md for the input paths, and
pandas / pyarrow / scipy):

```
python3 analysis/build_insights.py -o data/insights.json
```

Nothing on the page is typed in by hand; it is all read from `data/insights.json`.

The underlying sample-level dataset is not published here.

Computed on Tenstorrent hardware with [tt-bio](https://github.com/moritztng/tt-bio).
