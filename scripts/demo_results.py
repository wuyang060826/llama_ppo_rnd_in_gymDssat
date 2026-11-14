"""Summarize the archived checkpoint evaluations using the Python standard library."""

from statistics import mean

from fieldlab_data import load_archive


def main():
    runs = load_archive()

    print(f"Archived validation: {len(runs)} checkpoints, "
          f"{sum(len(run['rows']) for run in runs)} evaluations")
    print()
    print("| Checkpoint | Runs | Mean reward | Mean yield | Mean irrigation | Mean fertilizer |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for run in runs:
        checkpoint, rows = run["id"], run["rows"]
        metrics = [
            mean(row[metric] for row in rows)
            for metric in ("reward", "yield", "irr", "fert")
        ]
        values = " | ".join(f"{value:,.2f}" for value in metrics)
        print(f"| {checkpoint} | {len(rows)} | {values} |")


if __name__ == "__main__":
    main()
