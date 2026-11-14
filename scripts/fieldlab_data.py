"""Validate and load archived crop-policy evaluations for all demo outputs."""

import csv
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "test/val_proj/batch_val_results_0417/batch_validation_results_20260419_053122.csv"
METRICS = ("reward", "yield", "irr", "fert", "wue", "nue")


def load_archive(path=SOURCE):
    """Return checkpoints and ordered repeats, rejecting ambiguous input rows."""
    runs = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        fields = reader.fieldnames or []
        required = {"checkpoint", "repeat", *METRICS}
        if required - set(fields):
            raise ValueError(f"Missing CSV columns: {sorted(required - set(fields))}")
        if len(fields) != len(set(fields)):
            raise ValueError("Duplicate CSV column names")

        for line, row in enumerate(reader, start=2):
            if None in row or any(row[key] is None for key in required):
                raise ValueError(f"Line {line}: inconsistent column count")
            checkpoint = row["checkpoint"].strip()
            try:
                repeat = int(row["repeat"])
                values = {key: float(row[key]) for key in METRICS}
            except (ValueError, TypeError) as error:
                raise ValueError(f"Line {line}: invalid numeric field") from error
            if not re.fullmatch(r"model_ep[0-9]+", checkpoint):
                raise ValueError(f"Line {line}: invalid checkpoint ID")
            if repeat < 1 or not all(map(math.isfinite, values.values())):
                raise ValueError(f"Line {line}: invalid repeat or non-finite metric")
            if any(values[key] < 0 for key in METRICS if key != "reward"):
                raise ValueError(f"Line {line}: agricultural metrics must be nonnegative")
            repeats = runs.setdefault(checkpoint, {})
            if repeat in repeats:
                raise ValueError(f"Line {line}: duplicate {checkpoint}, repeat {repeat}")
            repeats[repeat] = values

    if not runs:
        raise ValueError("The evaluation archive is empty")
    return [
        {"id": checkpoint, "rows": [
            {"repeat": repeat, **values} for repeat, values in sorted(repeats.items())
        ]}
        for checkpoint, repeats in runs.items()
    ]
