"""Build the FieldLab browser data bundle from the archived CSV."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "test/val_proj/batch_val_results_0417/batch_validation_results_20260419_053122.csv"
OUTPUT = ROOT / "docs/fieldlab/data.js"
METRICS = ("reward", "yield", "irr", "fert", "wue", "nue")


def main():
    runs = {}
    with SOURCE.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            checkpoint = row["checkpoint"]
            repeat = int(row["repeat"])
            values = {key: float(row[key]) for key in METRICS}
            if repeat < 1 or not all(map(math.isfinite, values.values())):
                raise ValueError(f"Invalid observation: {checkpoint}, repeat {repeat}")
            repeats = runs.setdefault(checkpoint, {})
            if repeat in repeats:
                raise ValueError(f"Duplicate observation: {checkpoint}, repeat {repeat}")
            repeats[repeat] = values

    if not runs:
        raise ValueError("The evaluation archive is empty")
    payload = [
        {"id": checkpoint, "rows": [
            {"repeat": repeat, **values} for repeat, values in sorted(repeats.items())
        ]}
        for checkpoint, repeats in runs.items()
    ]
    OUTPUT.write_text("window.FIELDLAB_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
