# Verification record

Reviewed on 14 November 2026.

## Automated checks

- 14 Python regression tests pass locally, covering archive integrity, build fingerprints, CLI validation, rollout normalization, terminal masking, learning-rate clamping, and intrinsic reward decay.
- 3 Node regression tests pass for browser data validation, known means, and CSV export.
- Python 3.9 and Python 3.12 CI jobs run on Ubuntu 22.04 with CPU PyTorch.
- Generated bundle verification, syntax checks, and README local links pass.

[Live CI results](https://github.com/wuyang060826/llama_ppo_rnd_in_gymDssat/actions/workflows/quality.yml)

## Live browser checks

Chrome was used against the deployed Pages site. [Recorded assertions](browser-checks.json) cover root routing, 129 checkpoint options, archive dimensions, known means, metric switching, repeat inspection and highlighting, filtered selection, no-match behavior, swap, reset, same-checkpoint feedback, URL restoration, keyboard metric selection, and invalid query fallback.

The actual downloaded CSV contains 20 observations, all six metrics, and unrounded source values. The README comparison image loaded successfully on GitHub. A second browser pass verified content-fingerprinted assets and the filtered-selection fix.

[Verified comparison view](fieldlab-comparison.jpg)

## Scope

Desktop interaction was checked at a 1363-pixel viewport with no horizontal overflow. Responsive and print styles are implemented. Phone hardware and print output were not tested.

Full CUDA, DSSAT, and LLaMA training remains unverified here. CPU tests establish numerical edge-case behavior. The modified learning behavior requires new experiments before comparing results with the archived batch.

The repository contains historical research variants alongside the active v3 entry. Those historical training implementations have not all been migrated or certified.
