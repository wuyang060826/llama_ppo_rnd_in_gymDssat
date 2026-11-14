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
