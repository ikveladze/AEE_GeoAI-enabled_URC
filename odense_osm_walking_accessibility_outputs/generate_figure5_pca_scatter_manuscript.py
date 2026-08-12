#!/usr/bin/env python3
"""Regenerate manuscript Figure 5: PCA scatter coloured by final GMM typology.

Fixes:
  - Removes exploratory k-means (k=6) colouring that conflicted with GMM k=5
  - Colours by the four smoothed functional URC map labels
  - Axis labels include explained variance (Reviewer 4 y-axis clarity)
  - Caption states supporting heat / air / noise layers are in the PCA matrix
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
FEAT_GPKG = SCRIPT_DIR / "odense_aee_feature_space_outputs" / "odense_aee_feature_space_500m.gpkg"
TYP_GPKG = (
    SCRIPT_DIR
    / "odense_geoai_functional_urc_typology_outputs"
    / "odense_geoai_functional_urc_typology_500m.gpkg"
)
VAR_CSV = SCRIPT_DIR / "odense_aee_feature_space_outputs" / "odense_aee_pca_explained_variance.csv"
OUT_DIR = SCRIPT_DIR / "odense_aee_feature_space_outputs"

# Overwrite the manuscript scatter asset + keep a clearly named copy
OUT_PNG = OUT_DIR / "odense_aee_pca_scatter.png"
OUT_PNG_NAMED = OUT_DIR / "Figure_5_PCA_feature_space_GMM_labels.png"
OUT_PDF = OUT_DIR / "Figure_5_PCA_feature_space_GMM_labels.pdf"
OUT_CAPTION = OUT_DIR / "Figure_5_PCA_feature_space_caption.txt"

# Match typology map palette (geoai_functional_urc_typology_odense.make_typology_palette)
PALETTE = {
    "Low-access transition zone": "#9fc8c8",
    "Accessible suburban / local-centre zone": "#00b0be",
    "Compact built-up / environmental burden zone": "#1a80bb",
    "Central accessible urban core": "#ea801c",
}

# Draw order: large/background classes first, core last
DRAW_ORDER = [
    "Low-access transition zone",
    "Accessible suburban / local-centre zone",
    "Compact built-up / environmental burden zone",
    "Central accessible urban core",
]

SHORT_LABEL = {
    "Low-access transition zone": "Low-access transition",
    "Accessible suburban / local-centre zone": "Accessible suburban / local centre",
    "Compact built-up / environmental burden zone": "Compact built-up / env. burden",
    "Central accessible urban core": "Central accessible urban core",
}


def main() -> None:
    feat = gpd.read_file(FEAT_GPKG)
    typ = gpd.read_file(TYP_GPKG)
    var = pd.read_csv(VAR_CSV).set_index("component")
    # Display labels locked to audited Results figures (CSV: 0.4841 / 0.1596)
    pc1_pct, pc2_pct = 48.4, 15.9
    assert abs(100 * float(var.loc["PC1", "explained_variance_ratio"]) - pc1_pct) < 0.2
    assert abs(100 * float(var.loc["PC2", "explained_variance_ratio"]) - pc2_pct) < 0.2

    m = feat[["hex_id", "PC1", "PC2"]].merge(
        typ[["hex_id", "functional_urc_type"]],
        on="hex_id",
        how="inner",
    )
    if len(m) != 1539:
        raise SystemExit(f"Expected 1539 cells after join, got {len(m)}")

    fig, ax = plt.subplots(figsize=(9.2, 7.2))
    ax.axhline(0, color="#9aa7b5", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#9aa7b5", linewidth=0.7, zorder=0)

    for label in DRAW_ORDER:
        sub = m[m["functional_urc_type"] == label]
        ax.scatter(
            sub["PC1"],
            sub["PC2"],
            s=16,
            alpha=0.82,
            c=PALETTE[label],
            edgecolors="none",
            label=SHORT_LABEL[label],
            zorder=2,
        )

    ax.set_xlabel(f"PC1 ({pc1_pct:.1f}% of variance)", fontsize=12)
    ax.set_ylabel(f"PC2 ({pc2_pct:.1f}% of variance)", fontsize=12)
    ax.set_title(
        "Odense AEE feature space (PCA)",
        fontsize=14,
        fontweight="bold",
        loc="left",
        pad=10,
    )
    ax.tick_params(labelsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = [
        mpatches.Patch(facecolor=PALETTE[lab], edgecolor="#555555", linewidth=0.4, label=SHORT_LABEL[lab])
        for lab in DRAW_ORDER
    ]
    ax.legend(
        handles=handles,
        title="Final GMM typology (k = 5 → 4 labels)",
        loc="upper right",
        frameon=True,
        fontsize=8.5,
        title_fontsize=9,
        framealpha=0.95,
        borderpad=0.6,
    )

    fig.text(
        0.01,
        0.01,
        "Points = 500 m cells (n = 1,539). Colours = smoothed functional URC map labels "
        "(not exploratory k-means).",
        fontsize=7.5,
        color="#444444",
        ha="left",
        va="bottom",
    )

    fig.tight_layout(rect=(0, 0.03, 1, 1))
    for path in (OUT_PNG, OUT_PNG_NAMED, OUT_PDF):
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    caption = (
        "Figure 5. PCA representation of the Odense AEE feature space. Each point is a "
        "500 m hexagonal cell (n = 1,539). The PCA matrix comprises 15 median-imputed, "
        "standardised indicators: walking, cycling and public-transport stop accessibility; "
        "green-area share, environmental quality and burden, built-up share and major-road "
        "buffer exposure; supporting heat (summer land-surface temperature), air-pollution "
        "and traffic-noise surfaces; and population density and age vulnerability. "
        f"PC1 explains {pc1_pct:.1f}% and PC2 {pc2_pct:.1f}% of variance "
        f"(64.4% jointly). Colours show the final four functional URC "
        "map labels derived from the BIC-selected five-component GMM (after Queen smoothing), "
        "overlaid for interpretation; they are not a separate k-means partition of PC scores. "
        "Higher PC1 scores correspond to denser, more accessible and more environmentally "
        "burdened contexts; PC2 separates stronger road-/noise-exposed profiles from greener / "
        "higher-quality profiles."
    )
    OUT_CAPTION.write_text(caption + "\n", encoding="utf-8")

    print(f"Saved: {OUT_PNG}")
    print(f"Saved: {OUT_PNG_NAMED}")
    print(f"Saved: {OUT_PDF}")
    print(f"Saved: {OUT_CAPTION}")
    print(f"PC1={pc1_pct:.1f}%  PC2={pc2_pct:.1f}%  cells={len(m)}")
    print(m["functional_urc_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
