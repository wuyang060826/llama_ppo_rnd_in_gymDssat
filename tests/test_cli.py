"""Check the normal training entry before any optional dependencies load."""
import contextlib
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from main.llama_ppo_rnd.cli import parse_args


class CliTests(unittest.TestCase):
    def test_help_through_script(self):
        entry = Path(__file__).resolve().parents[1] / "main/llama_ppo_rnd/llama_ppo_rnd_update_glm5_v3.py"
        result = subprocess.run([sys.executable, str(entry), "--help"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--seed", result.stdout)
        self.assertIn("--dssat-runner", result.stdout)

    def test_invalid_episodes_seed_and_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = Path(directory) / "runner"
            runner.write_text("#!/bin/sh\n")
            runner.chmod(0o700)
            base = ["--model", "local", "--output-dir", directory + "/run", "--dssat-runner", str(runner)]
            self.assertEqual(parse_args(base).episodes, 2000)
            for invalid in [["--episodes", "0"], ["--seed", "-1"], ["--seed", str(2**32)],
                            ["--dssat-runner", directory + "/missing"]]:
                with self.subTest(invalid=invalid), contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit) as error:
                        parse_args(base + invalid)
                    self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
