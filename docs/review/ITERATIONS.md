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
