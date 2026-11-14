"""Render the README's archived checkpoint comparison with Matplotlib."""

from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fieldlab_data import ROOT, load_archive


def main():
    runs = {run["id"]: run["rows"] for run in load_archive()}
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.3), layout="constrained")
    fig.set_facecolor("#f6f5ed")
    fig.suptitle("Two crop policies, different input trade-offs", fontsize=19,
                 fontweight="bold", color="#163126")
    for axis, (key, label, unit) in zip(axes, (
        ("yield", "Harvest yield", "kg/ha"),
        ("irr", "Irrigation", "mm"),
        ("fert", "Nitrogen", "kg/ha"),
    )):
        values = [mean(row[key] for row in runs[checkpoint])
                  for checkpoint in ("model_ep949", "model_ep953")]
        axis.set_facecolor("#f6f5ed")
        bars = axis.barh(["ep949", "ep953"], values, color=["#21654d", "#a64b27"], height=.5)
        axis.invert_yaxis()
        axis.bar_label(bars, labels=[f"{value:,.1f}" for value in values], padding=5)
        axis.set_xlim(0, max(values) * 1.36)
        axis.set_title(f"{label} · {unit}", loc="left", fontsize=12, color="#163126")
        axis.spines[:].set_visible(False)
        axis.tick_params(left=False, bottom=False, labelbottom=False)
    fig.savefig(ROOT / "docs/fieldlab/readme-comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
