# FieldLab development

The research flow is **compare two checkpoints → inspect a metric → export the evidence → understand the method**. The default pair illustrates a yield-versus-input trade-off in the archived batch.

## Run and rebuild

Open `docs/fieldlab/index.html` directly for offline use. For shareable filter URLs, serve over HTTP.

```sh
python scripts/build_fieldlab.py
python -m http.server 8000 --directory docs
```

Open http://localhost:8000/fieldlab/. The Python builder converts the archived CSV into `data.js`; it validates finite metrics and unique checkpoint/repeat pairs. The browser code keeps selection state in the URL, computes statistics from the original rows, and exports selected rows with all six metrics. There is no backend or frontend build dependency.

## Open-source design references

| Reference | Decision in FieldLab |
| --- | --- |
| [Aim](https://github.com/aimhubio/aim/blob/main/docs/source/ui/pages/explorers.md) | Start with run comparison and a metric selector, then drill into observations. |
| [Chart.js](https://github.com/chartjs/Chart.js/blob/master/docs/configuration/responsive.md) | Size the chart from its container, observe reflows, and scale the canvas for device density. |
| [RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) | Keep archived evaluation browsing distinct from the training entry point. |
| [CleanRL](https://github.com/vwxyzjn/cleanrl) | Keep setup commands and experiment lineage visible in the repository. |

## Acceptance checklist

| # | Check |
| ---: | --- |
| 1 | Page title and project identity |
| 2 | Checkpoint and row counts match the archive |
| 3 | All checkpoints selectable |
| 4 | Curated defaults restored on first open |
| 5 | Mean values match the CSV |
| 6 | Repeat table matches the selected pair |
| 7 | Six metrics selectable |
| 8 | Axis metric and units stay synchronized |
| 9 | Table updates when metric changes |
| 10 | Swap reverses A and B |
| 11 | Same checkpoint shows zero mean difference |
| 12 | URL encodes current selection on HTTP |
| 13 | Reload restores selection |
| 14 | Reset restores default pair and metric |
| 15 | CSV includes both series and full precision |
| 16 | Unknown query values fall back to defaults |
| 17 | Keyboard users can skip to the workspace |
| 18 | Interpretation and limitations can be expanded |
| 19 | Mobile content stays within viewport |
| 20 | Reduced motion is respected and browser has no uncaught errors |

## Code boundaries

- `scripts/fieldlab_data.py` is the shared CSV validation and loading layer for demos and generated assets.
- `scripts/build_fieldlab.py` creates the offline browser bundle. `scripts/plot_fieldlab.py` creates the README figure with Matplotlib.
- `docs/index.html` is the Pages root entry. `docs/fieldlab/` contains the browser UI.
- `main/` holds research implementations. `test/val_proj/` contains historical evaluation scripts and outputs. When running automated tests, target a dedicated test file instead of collecting legacy experiment scripts.

The README presents a chart, exact values, and a collapsible reproduction path, following the data and reproducibility emphasis of [Open RL Benchmark](https://github.com/openrlbenchmark/openrlbenchmark) and [CleanRL](https://github.com/vwxyzjn/cleanrl).
