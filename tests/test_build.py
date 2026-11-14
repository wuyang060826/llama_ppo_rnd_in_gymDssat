"""Verify generated artifacts and cache fingerprints stay consistent."""
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from scripts.fieldlab_data import ROOT, SOURCE


class BuildTests(unittest.TestCase):
    def test_changed_asset_requires_rebuild_and_check_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [
                "scripts/build_fieldlab.py", "scripts/fieldlab_data.py",
                "docs/fieldlab/index.html", "docs/fieldlab/app.js",
                "docs/fieldlab/model.js", "docs/fieldlab/style.css",
                str(SOURCE.relative_to(ROOT)),
            ]
            for relative in paths:
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, destination)
            command = [sys.executable, str(root / "scripts/build_fieldlab.py")]
            subprocess.run(command, check=True, capture_output=True)
            subprocess.run(command + ["--check"], check=True, capture_output=True)
            index = root / "docs/fieldlab/index.html"
            original = index.read_text()
            script = root / "docs/fieldlab/app.js"
            script.write_text(script.read_text() + "\n// changed asset\n")
            result = subprocess.run(command + ["--check"], capture_output=True)
            self.assertEqual(result.returncode, 1)
            self.assertEqual(index.read_text(), original)
            subprocess.run(command, check=True, capture_output=True)
            self.assertNotEqual(index.read_text(), original)
            subprocess.run(command + ["--check"], check=True, capture_output=True)


if __name__ == "__main__":
    unittest.main()
