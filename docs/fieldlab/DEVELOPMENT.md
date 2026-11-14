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

## Implementation map

| Layer | Entry | Responsibility |
| --- | --- | --- |
| Research | `main/llama_ppo_rnd/llama_ppo_rnd_update_glm5_v3.py` | Model, environment, training, validation, metrics |
| Arguments | `main/llama_ppo_rnd/cli.py` | Dependency-free help and input validation |
| Numerics | `main/llama_ppo_rnd/training_core.py` | PPO rollout buffer, GAE, and learning-rate schedule |
| Archive | `scripts/fieldlab_data.py` | CSV schema and numeric validation |
| Browser data | `docs/fieldlab/model.js` | Validation, means, CSV serialization |
| Browser view | `docs/fieldlab/app.js` | Selection, rendering, URL state, downloads |
| Regression | `tests/` | CPU numerical, input, CLI, and export checks |

The historical `main/` variants and `test/val_proj/` evaluation scripts retain their existing entry paths. Automated regressions use `tests/`. The active training entry accepts model and output paths from the command line.

## Demo walkthrough

1. Open the [README](../../README.md#results-demo) and explain the archived two-policy comparison.
2. Open [FieldLab](https://wuyang060826.github.io/llama_ppo_rnd_in_gymDssat/fieldlab/), switch the metric to irrigation, and inspect an individual repeat.
3. Search for an episode number to choose a different checkpoint.
4. Swap A and B to inspect direction of differences, then reset.
5. Download the selected raw observations or share the current URL.
6. Use the reproduction section to transition from the demo into development.

## Checks

```sh
python -m unittest discover -s tests -p 'test_*.py'
node --test tests/fieldlab.test.cjs
python scripts/build_fieldlab.py --check
python main/llama_ppo_rnd/llama_ppo_rnd_update_glm5_v3.py --help
```

Python tests require NumPy and CPU-compatible PyTorch. Node 20+ runs the browser data tests without packages. The [workflow](../../.github/workflows/quality.yml) runs the suite on Python 3.9 and 3.12.

The [iteration log](../review/ITERATIONS.md) records 20 implementation rounds and 20 reference repositories. `archive.json` records the source path, SHA-256 fingerprint, dimensions, and units. Update the CSV first and regenerate `data.js` and `archive.json` together. Recreate the README figure with `python scripts/plot_fieldlab.py`, which additionally requires Matplotlib.

## Research verification scope

CPU regressions verify buffer terminal masking, single-transition normalization, and the learning-rate floor. CLI tests run the normal training script's help and validate arguments. Full CUDA + DSSAT + LLaMA training remains unverified in this environment. Training now seeds its RNGs, records configuration, closes simulator resources after exceptions, and rejects unfinished episodes at the configured step limit.

Further research validation should cover the terminal observation schema, the RND decay coefficient reaching the PPO objective, exact learning-rate update counts, and checkpoint resumption. Existing checkpoints contain model and optimizer weights. A complete resume workflow still needs scheduler, RND, RNG, and environment-state handling.
