"""Summarize the archived checkpoint evaluations using the Python standard library."""

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean


RESULTS = (
    Path(__file__).resolve().parents[1]
    / "test/val_proj/batch_val_results_0417/batch_validation_results_20260419_053122.csv"
)


def main():
    by_checkpoint = defaultdict(list)
    with RESULTS.open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            by_checkpoint[row["checkpoint"]].append(row)

    print(f"Archived validation: {len(by_checkpoint)} checkpoints, "
          f"{sum(map(len, by_checkpoint.values()))} evaluations")
    print()
    print("| Checkpoint | Runs | Mean reward | Mean yield | Mean irrigation | Mean fertilizer |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for checkpoint, rows in by_checkpoint.items():
        metrics = [
            mean(float(row[metric]) for row in rows)
            for metric in ("reward", "yield", "irr", "fert")
        ]
        values = " | ".join(f"{value:,.2f}" for value in metrics)
        print(f"| {checkpoint} | {len(rows)} | {values} |")


if __name__ == "__main__":
    main()
