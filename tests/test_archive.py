"""Regression coverage for the archive shared by README, CLI, and browser."""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from fieldlab_data import METRICS, load_archive


class ArchiveTests(unittest.TestCase):
    def read_rows(self, rows, fields=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.csv"
            with path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(fields or ["checkpoint", "repeat", *METRICS])
                writer.writerows(rows)
            return load_archive(path)

    def test_archive_counts_and_known_mean(self):
        runs = load_archive()
        self.assertEqual(len(runs), 129)
        self.assertEqual(sum(len(run["rows"]) for run in runs), 1290)
        rows = next(run["rows"] for run in runs if run["id"] == "model_ep949")
        self.assertAlmostEqual(sum(row["irr"] for row in rows) / len(rows), 275.4)

    def test_repeats_are_sorted(self):
        runs = self.read_rows([["model_ep1", 2, 1, 2, 3, 4, 5, 6],
                               ["model_ep1", 1, 1, 2, 3, 4, 5, 6]])
        self.assertEqual([row["repeat"] for row in runs[0]["rows"]], [1, 2])

    def test_invalid_values(self):
        valid = ["model_ep1", 1, -1, 2, 3, 4, 5, 6]
        for column, value in [(0, ""), (0, "=formula"), (1, 0), (1, "1.5"),
                              (2, "NaN"), (3, "inf"), (4, -1), (5, "")]:
            with self.subTest(column=column, value=value):
                row = valid.copy()
                row[column] = value
                with self.assertRaises(ValueError):
                    self.read_rows([row])
        self.assertEqual(self.read_rows([valid])[0]["rows"][0]["reward"], -1)

    def test_empty_duplicate_and_ragged_rows(self):
        row = ["model_ep1", 1, 1, 2, 3, 4, 5, 6]
        for rows in [[], [row, row], [row[:-1]], [row + [7]]]:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                self.read_rows(rows)

    def test_column_headers(self):
        for fields in [["checkpoint"], ["checkpoint", "repeat", *METRICS, "yield"]]:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                self.read_rows([], fields)


if __name__ == "__main__":
    unittest.main()
