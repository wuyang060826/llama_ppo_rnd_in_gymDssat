"""Shared validation and loading for the archived evaluation data."""

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "test/val_proj/batch_val_results_0417/batch_validation_results_20260419_053122.csv"
METRICS = ("reward", "yield", "irr", "fert", "wue", "nue")


def load_archive(path=SOURCE):
    runs = {}
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        missing = {"checkpoint", "repeat", *METRICS} - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")
        for row in reader:
            checkpoint = row["checkpoint"]
            repeat = int(row["repeat"])
            values = {key: float(row[key]) for key in METRICS}
            if not checkpoint or repeat < 1 or not all(map(math.isfinite, values.values())):
                raise ValueError(f"Invalid observation: {checkpoint}, repeat {repeat}")
            repeats = runs.setdefault(checkpoint, {})
            if repeat in repeats:
                raise ValueError(f"Duplicate observation: {checkpoint}, repeat {repeat}")
            repeats[repeat] = values

    if not runs:
        raise ValueError("The evaluation archive is empty")
    return [
        {"id": checkpoint, "rows": [
            {"repeat": repeat, **values} for repeat, values in sorted(repeats.items())
        ]}
        for checkpoint, repeats in runs.items()
    ]
