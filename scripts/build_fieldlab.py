"""Build the FieldLab browser data bundle from the archived CSV."""

import json

from fieldlab_data import ROOT, load_archive

OUTPUT = ROOT / "docs/fieldlab/data.js"


def main():
    payload = load_archive()
    OUTPUT.write_text("window.FIELDLAB_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n",
                      encoding="utf-8")


if __name__ == "__main__":
    main()
