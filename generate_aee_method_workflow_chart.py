"""Generate a clean Figure-2 style methodological workflow (matches manuscript diagram layout).

Updated for the current Odense implementation:
- Dasymetric demography v2 (SOGN x BBR; FOLK1A; DST soft-constraint)
- Clustering on AE + density morphology (demographic vulnerability descriptive only)
- GMM k selected by BIC/AIC (this run: k = 5 → 4 map labels)
- Uncertainty: 1 - max posterior, boundary heterogeneity, scenario stability
- GMM–K-means agreement + ARI/NMI benchmarking
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PNG = SCRIPT_DIR / "odense_aee_method_workflow_chart.png"
OUT_PDF = SCRIPT_DIR / "odense_aee_method_workflow_chart.pdf"
OUT_SVG = SCRIPT_DIR / "odense_aee_method_workflow_chart.svg"
OUT_IMPL = SCRIPT_DIR / "odense_aee_method_workflow_chart_as_implemented.png"

BG = "#ECEFF1"
BOX_FC = "#CFD8DC"
BOX_EC = "#455A64"
TEXT = "#263238"
MUTED = "#546E7A"


def add_box(ax, x, y, w, h, title, subtitle=None, fontsize=9.2, title_size=10.0):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="square,pad=0.0",
        facecolor=BOX_FC,
        edgecolor=BOX_EC,
        linewidth=1.15,
        mutation_aspect=1,
    )
    # Prefer sharp rectangles like the reference screenshot
    ax.add_patch(Rectangle((x, y), w, h, facecolor=BOX_FC, edgecolor=BOX_EC, linewidth=1.15))
    cx, cy = x + w / 2, y + h / 2
    if subtitle:
        ax.text(cx, cy + 0.12, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=TEXT)
        ax.text(cx, cy - 0.18, subtitle, ha="center", va="center", fontsize=fontsize - 0.8, color=MUTED)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=title_size, fontweight="bold", color=TEXT, linespacing=1.15)
    return (x, y, w, h)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.2,
            color=BOX_EC,
            shrinkA=1.5,
            shrinkB=1.5,
        )
    )


def mid(b):
    x, y, w, h = b
    return x + w / 2, y + h / 2


def bottom(b):
    x, y, w, h = b
    return x + w / 2, y


def top(b):
    x, y, w, h = b
    return x + w / 2, y + h


def main():
    fig, ax = plt.subplots(figsize=(14.5, 18.0))
    ax.set_xlim(0, 14.5)
    ax.set_ylim(0, 18.0)
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.axis("off")

    ax.text(
        7.25,
        17.55,
        "Methodological workflow for GeoAI-enabled functional URC mapping",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        7.25,
        17.15,
        "AEE framework  |  Odense  |  500 m hex (n = 1,539)  |  EPSG:25832  |  as implemented",
        ha="center",
        va="center",
        fontsize=9.2,
        color=MUTED,
    )

    # 1. Study area
    b_study = add_box(ax, 4.35, 16.05, 5.8, 0.75, "Study area and 500 m hex grid", "EPSG:25832")

    # 2. Indicator construction
    b_ind = add_box(ax, 4.55, 14.85, 5.4, 0.7, "Indicator construction")

    # 3. Three domains
    b_acc = add_box(
        ax, 0.55, 13.25, 4.0, 1.05,
        "Accessibility",
        "Walk / bike / PT-stop network access\n(Dijkstra + exponential decay; no GTFS)",
        fontsize=8.2,
        title_size=10.0,
    )
    b_env = add_box(
        ax, 5.25, 13.25, 4.0, 1.05,
        "Environment",
        "Green share, quality, burden,\nbuilt-up, road exposure",
        fontsize=8.2,
        title_size=10.0,
    )
    b_eq = add_box(
        ax, 9.95, 13.25, 4.0, 1.05,
        "Equity",
        "Dasymetric density (SOGN×BBR→FOLK1A)\n+ demographic vulnerability (descriptive)",
        fontsize=8.0,
        title_size=10.0,
    )

    # 4. Feature matrix
    b_mat = add_box(
        ax, 3.55, 11.75, 7.4, 0.95,
        "AEE feature matrix",
        "Hex-level merge + median imputation\nClustering inputs: AE + density morphology (vuln. excluded from GMM)",
        fontsize=8.0,
        title_size=10.2,
    )

    # 5. Z-score
    b_z = add_box(ax, 4.75, 10.45, 5.0, 0.7, "Z-score standardisation")

    # 6. Four analytical branches
    col_w = 3.15
    gap = 0.28
    x0 = 0.55
    y_branch = 8.55

    b_gmm = add_box(
        ax, x0, y_branch, col_w, 1.15,
        "GMM primary clustering",
        "BIC/AIC-selected k (this run: k = 5)\nprobabilistic membership",
        fontsize=7.8,
        title_size=9.5,
    )
    b_pca = add_box(
        ax, x0 + (col_w + gap), y_branch, col_w, 1.15,
        "PCA",
        "Interpret dominant AEE gradients\n(+ UMAP exploratory only)",
        fontsize=7.8,
        title_size=9.5,
    )
    b_km = add_box(
        ax, x0 + 2 * (col_w + gap), y_branch, col_w, 1.15,
        "K-means benchmark",
        "Robustness check only\n(same feature matrix)",
        fontsize=7.8,
        title_size=9.5,
    )
    b_bench = add_box(
        ax, x0 + 3 * (col_w + gap), y_branch, col_w, 1.15,
        "Benchmarking",
        "Single-domain reference\nclassifications",
        fontsize=7.8,
        title_size=9.5,
    )

    # GMM outputs
    b_typ = add_box(
        ax, 0.35, 6.55, 3.55, 1.25,
        "Final functional URC typology",
        "4 map labels via cluster profiles\n+ Queen-contiguity smoothing",
        fontsize=7.6,
        title_size=9.2,
    )
    b_unc = add_box(
        ax, 0.35, 4.85, 3.55, 1.15,
        "Uncertainty layers",
        "1 − max posterior\n+ boundary heterogeneity\n+ scenario stability",
        fontsize=7.5,
        title_size=9.2,
    )
    b_score = add_box(
        ax, 4.15, 6.55, 2.95, 1.25,
        "Continuous URC score",
        "0 = urban / central\n100 = rural / peripheral",
        fontsize=7.6,
        title_size=9.2,
    )

    # K-means → agreement
    b_agree = add_box(
        ax, 7.45, 6.55, 3.15, 1.25,
        "Algorithmic agreement map",
        "GMM vs K-means\ndominant-class pairing",
        fontsize=7.6,
        title_size=9.2,
    )

    # Benchmarking metrics
    b_ari = add_box(
        ax, 10.95, 6.55, 3.0, 1.25,
        "ARI / NMI metrics",
        "Agreement vs single-domain\nand morphology references",
        fontsize=7.6,
        title_size=9.2,
    )

    # Typology label detail (small)
    b_labels = add_box(
        ax, 4.15, 4.85, 6.45, 1.15,
        "Interpreted functional classes",
        "Central accessible urban core · Accessible suburban / local-centre\n"
        "Compact built-up / env. burden · Low-access transition zone",
        fontsize=7.3,
        title_size=9.0,
    )

    # Final outputs
    b_out = add_box(
        ax, 2.6, 2.55, 9.3, 1.35,
        "Geovisual & tabular outputs",
        "AEE indicator maps · PCA/UMAP · typology & URC score ·\n"
        "uncertainty / boundary / stability / GMM–K-means agreement · validation tables",
        fontsize=8.0,
        title_size=10.2,
    )

    # Footer note
    ax.add_patch(Rectangle((0.55, 0.35), 13.4, 1.55, facecolor="#E0E0E0", edgecolor=BOX_EC, linewidth=1.0))
    ax.text(
        7.25,
        1.55,
        "Implementation updates reflected in this figure",
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        7.25,
        0.85,
        "Demography v2 dasymetric surface · GMM k by BIC/AIC (not fixed) · demographic vulnerability\n"
        "excluded from clustering inputs · density retained as morphology · uncertainty includes boundary\n"
        "heterogeneity and scenario stability · four interpreted map labels after Queen smoothing",
        ha="center",
        va="center",
        fontsize=7.6,
        color=MUTED,
        linespacing=1.25,
    )

    # Arrows
    arrow(ax, *bottom(b_study), *top(b_ind))
    arrow(ax, *bottom(b_ind), mid(b_acc)[0], top(b_acc)[1])
    arrow(ax, *bottom(b_ind), mid(b_env)[0], top(b_env)[1])
    arrow(ax, *bottom(b_ind), mid(b_eq)[0], top(b_eq)[1])

    arrow(ax, *bottom(b_acc), 4.2, top(b_mat)[1])
    arrow(ax, *bottom(b_env), *top(b_mat))
    arrow(ax, *bottom(b_eq), 10.3, top(b_mat)[1])

    arrow(ax, *bottom(b_mat), *top(b_z))

    # z → four branches
    zx, zy = bottom(b_z)
    for b in (b_gmm, b_pca, b_km, b_bench):
        arrow(ax, zx, zy, mid(b)[0], top(b)[1])

    # GMM → typology / uncertainty / score
    arrow(ax, *bottom(b_gmm), mid(b_typ)[0], top(b_typ)[1])
    arrow(ax, *bottom(b_gmm), mid(b_unc)[0], top(b_unc)[1])
    arrow(ax, *bottom(b_gmm), mid(b_score)[0], top(b_score)[1])
    arrow(ax, *bottom(b_typ), mid(b_labels)[0], top(b_labels)[1])
    arrow(ax, *bottom(b_score), 5.6, top(b_labels)[1])

    # K-means → agreement
    arrow(ax, *bottom(b_km), *top(b_agree))

    # Benchmark → ARI
    arrow(ax, *bottom(b_bench), *top(b_ari))

    # Into geovisual outputs
    arrow(ax, *bottom(b_unc), 4.2, top(b_out)[1])
    arrow(ax, *bottom(b_labels), *top(b_out))
    arrow(ax, *bottom(b_pca), 6.5, top(b_out)[1])
    arrow(ax, *bottom(b_agree), 9.0, top(b_out)[1])
    arrow(ax, *bottom(b_ari), 10.8, top(b_out)[1])

    # silence unused FancyBboxPatch import side-effect
    _ = FancyBboxPatch

    fig.tight_layout(pad=0.3)
    for path in (OUT_PNG, OUT_PDF, OUT_SVG, OUT_IMPL):
        if path.suffix.lower() == ".png":
            fig.savefig(path, dpi=400, bbox_inches="tight", facecolor=BG)
        else:
            fig.savefig(path, bbox_inches="tight", facecolor=BG)
        print(f"Saved: {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
