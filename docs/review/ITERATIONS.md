# FieldLab and research repository review

Twenty focused review rounds, performed in order. Each entry records a new reference project, the observed gap, the implementation, and its verification. References informed design choices. Source code was implemented in this repository. Browser verification and scientific runtime limits are reported in DEVELOPMENT.md.

## Round 1

Reference [GitHub project](https://github.com/vwxyzjn/cleanrl).

Observed gap — The browser demo ended without a direct reproduction path.

Change — Add a reproduction section and navigation entry.

Verification — Checked command paths against scripts/demo_results.py and Python 3.9 syntax.

## Round 2

Reference [GitHub project](https://github.com/openrlbenchmark/openrlbenchmark).

Observed gap — Visitors cannot immediately identify the evaluation batch or checkpoint meaning.

Change — Expose archive date and simulator provenance beside the comparison.

Verification — Date checked against source CSV filename and all checkpoint repeat counts checked during archive generation.

## Round 3

Reference [GitHub project](https://github.com/mlflow/mlflow).

Observed gap — Metric switching left all visible summary cards fixed on agricultural inputs.

Change — Show selected-metric means and explicit same-checkpoint feedback.

Verification — JavaScript syntax passes; mean uses the same selected rows as the repeat table.

## Round 4

Reference [GitHub project](https://github.com/aimhubio/aim).

Observed gap — Two selectors each contained 129 entries with no search.

Change — Add checkpoint filtering while preserving selected policies.

Verification — Reviewed zero-match, swap-after-filter, and reset-after-filter paths; JavaScript syntax passes.

## Round 5

Reference [GitHub project](https://github.com/streamlit/streamlit).

Observed gap — Exports had identical filenames and no visible feedback.

Change — Name exports by checkpoint pair and acknowledge download requests.

Verification — CSV preserves raw precision and both series; all fields remain quoted.

## Round 6

Reference [GitHub project](https://github.com/gradio-app/gradio).

Observed gap — Load failure offered explanatory text without a recovery action.

Change — Provide reload and README fallback links on load failure.

Verification — Failure path hides the workspace and reveals recovery; URL write failures leave the comparison usable.

## Round 7

Reference [GitHub project](https://github.com/plotly/plotly.js).

Observed gap — Connected repeat points suggested a temporal trajectory and long tick values could collide.

Change — Use separate repeat markers with compact axis labels.

Verification — Marker values still match the full-precision data; exact values remain in the table and CSV.

## Round 8

Reference [GitHub project](https://github.com/chartjs/Chart.js).

Observed gap — Additional controls lacked mobile styling and printing clipped the table.

Change — Style search and feedback, expand CSS formatting, and add print layout.

Verification — Checked existing 500px and 710px breakpoints; print rules remove scroll clipping and keep cards together.

## Round 9

Reference [GitHub project](https://github.com/apache/echarts).

Observed gap — Charts had no way to inspect and locate a particular repeat.

Change — Link repeat selection to point emphasis and table highlighting.

Verification — Reviewed selection persistence across metric changes and reset; JavaScript syntax passes.

## Round 10

Reference [GitHub project](https://github.com/observablehq/plot).

Observed gap — Means alone concealed the spread of repeated evaluations.

Change — Expose observed metric ranges alongside selected means.

Verification — Ranges are derived from selected raw repeats; no confidence or significance claim is inferred.

## Round 11

Reference [GitHub project](https://github.com/pandas-dev/pandas).

Observed gap — Malformed CSV rows produced cryptic exceptions or ambiguous field assignments.

Change — Validate CSV shape, identifiers, finite values, and nonnegative agricultural metrics.

Verification — All 1290 archived rows pass; build output remains unchanged.

## Round 12

Reference [GitHub project](https://github.com/frictionlessdata/frictionless-py).

Observed gap — Browser trusted the generated bundle and mixed statistics with DOM rendering.

Change — Extract validated browser data and CSV operations into model.js.

Verification — Schema rejects duplicate IDs and repeats, missing or non-finite metrics; syntax checks pass.

## Round 13

Reference [GitHub project](https://github.com/iterative/dvc).

Observed gap — Published data had no content fingerprint or stale-output check.

Change — Generate an archive manifest and support deterministic verification.

Verification — Rebuild followed by --check passes against the CSV checksum and exact serialized bundle.

## Round 14

Reference [GitHub project](https://github.com/pytest-dev/pytest).

Observed gap — Schema and export changes lacked regression coverage for malformed inputs.

Change — Add focused data integrity and CSV export regression tests.

Verification — Five Python tests and three Node tests pass, including non-finite metrics, duplicate rows, counts, and export precision.

## Round 15

Reference [GitHub project](https://github.com/DLR-RM/stable-baselines3).

Observed gap — Singleton rollouts yielded NaN normalization and cosine schedules rebounded past the horizon.

Change — Extract training numerics and fix singleton normalization and learning-rate clamping.

Verification — Four CPU tests pass for terminal masking, singleton finite values, empty buffers, and schedule floor.

## Round 16

Reference [GitHub project](https://github.com/Farama-Foundation/Gymnasium).

Observed gap — Step-limited episodes could leak into the next rollout and exceptions left simulator resources open.

Change — Reject incomplete episodes and close training and validation environments in finally blocks.

Verification — Compiled the main entry and inspected exception boundaries. Full simulator execution still requires DSSAT.

## Round 17

Reference [GitHub project](https://github.com/thu-ml/tianshou).

Observed gap — Python, NumPy, and Torch RNGs were unseeded and run configuration was not saved.

Change — Seed training RNGs and persist run configuration and runtime versions.

Verification — Seed range and fresh-directory checks reviewed; metadata is written before model loading. Determinism across hardware is not claimed.

## Round 18

Reference [GitHub project](https://github.com/pytorch/examples).

Observed gap — Even --help imported CUDA and simulator dependencies on a fresh checkout.

Change — Parse and validate CLI arguments before importing training dependencies.

Verification — The normal script --help succeeds without Gym or Transformers; default episodes and seed verified through the parser.

## Round 19

Reference [GitHub project](https://github.com/actions/setup-python).

Observed gap — The repository had no automatic regression gate for incoming changes.

Change — Add Python-version CI coverage for the data, CLI, and CPU training checks.

Verification — Eleven Python tests and three Node tests pass locally. CI executes the same suite on Python 3.9 and 3.12.

## Round 20

Reference [GitHub project](https://github.com/scientific-python/cookie).

Observed gap — Contributors lacked a layer map and could confuse historical experiments with supported checks.

Change — Document implementation boundaries, demo walkthrough, regression commands, and research limits.

Verification — README links, help entry, bundle consistency, and diff checks pass. Live browser verification is recorded after publication.
