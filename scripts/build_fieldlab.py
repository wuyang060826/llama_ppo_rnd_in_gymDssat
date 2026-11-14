"""Rebuild or verify the browser data bundle and its provenance."""

import argparse
import hashlib
import json

from fieldlab_data import ROOT, SOURCE, load_archive

OUTPUT = ROOT / "docs/fieldlab/data.js"
MANIFEST = ROOT / "docs/fieldlab/archive.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if checked-in outputs differ")
    args = parser.parse_args()
    payload = load_archive()
    bundle = "window.FIELDLAB_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n"
    manifest = json.dumps({
        "schema_version": 1,
        "source": str(SOURCE.relative_to(ROOT)),
        "sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "checkpoints": len(payload),
        "evaluations": sum(len(run["rows"]) for run in payload),
        "units": {"reward": "score", "yield": "kg/ha", "irr": "mm",
                  "fert": "kg/ha", "wue": "kg/ha/mm", "nue": "kg/kg"},
    }, indent=2) + "\n"
    for path, content in ((OUTPUT, bundle), (MANIFEST, manifest)):
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != content:
                parser.exit(1, f"Stale generated file: {path.relative_to(ROOT)}\n")
        else:
            path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
