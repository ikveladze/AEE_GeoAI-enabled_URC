"""Lean methodological workflow (alternate Figure style).

Matches the older 'inputs → indicators → PCA / GMM vs K-means → uncertainty'
layout, updated for the current Odense AEE / functional URC implementation.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PNG = SCRIPT_DIR / "odense_aee_method_workflow_chart_lean.png"
OUT_PDF = SCRIPT_DIR / "odense_aee_method_workflow_chart_lean.pdf"
OUT_SVG = SCRIPT_DIR / "odense_aee_method_workflow_chart_lean.svg"

BG = "#F5F6F7"
BOX = "#E8EDF1"
EDGE = "#37474F"
TEXT = "#1C252B"
MUTED = "#546E7A"
ACCENT = "#2F5D7C"


def box(ax, x, y, w, h, title, sub=None, fs=9.0, ts=10.2):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.04",
            facecolor=BOX,
            edgecolor=EDGE,
            linewidth=1.2,
        )
    )
    cx, cy = x + w / 2, y + h / 2
    if sub:
        ax.text(cx, cy + 0.16, title, ha="center", va="center", fontsize=ts, fontweight="bold", color=TEXT)
        ax.text(cx, cy - 0.20, sub, ha="center", va="center", fontsize=fs, color=MUTED, linespacing=1.2)
    else:
        ax.text(cx, cy, title, ha="center", va="center", fontsize=ts, fontweight="bold", color=TEXT, linespacing=1.15)
    return (x, y, w, h)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.25,
            color=EDGE,
            shrinkA=2,
            shrinkB=2,
        )
    )


def bottom(b):
    x, y, w, h = b
    return x + w / 2, y


def top(b):
    x, y, w, h = b
    return x + w / 2, y + h


def mid(b):
    x, y, w, h = b
    return x + w / 2, y + h / 2


def main():
    fig, ax = plt.subplots(figsize=(13.5, 16.5))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 16.5)
    ax.set_facecolor(BG)
    fig.patch.set_facecolor(BG)
    ax.axis("off")

    ax.text(
        6.75,
        16.05,
        "Methodological workflow for GeoAI-enabled functional URC mapping",
        ha="center",
        va="center",
        fontsize=13.5,
        fontweight="bold",
        color=TEXT,
    )
    ax.text(
        6.75,
        15.65,
        "AEE framework · Odense · 500 m hex (n = 1,539) · EPSG:25832",
        ha="center",
        va="center",
        fontsize=9.5,
        color=MUTED,
    )

    # ---- Inputs ----
    b_osm = box(ax, 0.8, 14.1, 5.4, 1.0, "OpenStreetMap (OSM)", "Networks, amenities, PT stops\n(OSMnx / Overpass)")
    b_off = box(
        ax,
        7.3,
        14.1,
        5.4,
        1.0,
        "Official / statistical data",
        "GeoDanmark (Datafordeler) · BBR ·\nStatBank SOGN/FOLK · DST density classes",
        fs=8.2,
    )

    # ---- Extraction ----
    b_net = box(ax, 0.8, 12.55, 5.4, 0.95, "Network & POI extraction", "Walk / bike graphs + service points")
    b_env = box(
        ax,
        7.3,
        12.55,
        5.4,
        0.95,
        "Environment & demography",
        "Green/built-up/roads · dasymetric density\n(SOGN × BBR → FOLK1A)",
        fs=8.2,
    )

    # ---- Indicators ----
    b_ind = box(
        ax,
        2.4,
        10.85,
        8.7,
        1.1,
        "AEE indicator construction (500 m hex)",
        "Accessibility (walk / bike / PT-stop) · Environment (quality, burden, green, roads)\n"
        "Equity (density morphology; demographic vulnerability descriptive only)",
        fs=8.0,
        ts=10.5,
    )

    # ---- Matrix ----
    b_mat = box(
        ax,
        2.8,
        9.35,
        7.9,
        0.95,
        "AEE feature matrix + z-score standardisation",
        "Median imputation · clustering inputs = AE + density (vuln. excluded from GMM)",
        fs=8.2,
    )

    # ---- PCA ----
    b_pca = box(ax, 4.0, 7.95, 5.5, 0.85, "PCA (interpretive)", "Dominant AEE gradients (+ UMAP exploratory)")

    # ---- Dual path ----
    b_gmm = box(
        ax,
        0.7,
        5.85,
        5.6,
        1.35,
        "Gaussian Mixture Model (GMM)",
        "Primary typology · BIC/AIC-selected k\n(this run: k = 5) · soft membership",
        fs=8.3,
        ts=10.5,
    )
    b_km = box(
        ax,
        7.2,
        5.85,
        5.6,
        1.35,
        "K-means clustering",
        "Robustness benchmark only\n(hard class labels)",
        fs=8.3,
        ts=10.5,
    )

    b_soft = box(
        ax,
        0.7,
        4.25,
        5.6,
        1.05,
        "Functional URC typology (soft)",
        "4 interpreted labels + Queen smoothing\n+ continuous URC score (0–100)",
        fs=8.0,
    )
    b_hard = box(
        ax,
        7.2,
        4.25,
        5.6,
        1.05,
        "Hard benchmark classification",
        "K-means classes for agreement tests",
        fs=8.0,
    )

    # ---- Evaluation ----
    b_eval = box(
        ax,
        1.5,
        2.35,
        10.5,
        1.35,
        "Benchmarking and uncertainty analysis",
        "GMM–K-means agreement map  ·  GMM membership uncertainty (1 − max posterior)\n"
        "Boundary heterogeneity map  ·  Scenario stability  ·  ARI / NMI vs single-domain references",
        fs=8.2,
        ts=10.5,
    )

    # ---- Outputs ----
    b_out = box(
        ax,
        2.6,
        0.85,
        8.3,
        0.95,
        "Geovisual outputs",
        "Indicator maps · typology · URC score · uncertainty / agreement maps",
        fs=8.3,
    )

    # Arrows
    arrow(ax, *bottom(b_osm), *top(b_net))
    arrow(ax, *bottom(b_off), *top(b_env))
    arrow(ax, *bottom(b_net), 4.0, top(b_ind)[1])
    arrow(ax, *bottom(b_env), 9.5, top(b_ind)[1])
    arrow(ax, *bottom(b_ind), *top(b_mat))
    arrow(ax, *bottom(b_mat), *top(b_pca))
    arrow(ax, *bottom(b_pca), mid(b_gmm)[0], top(b_gmm)[1])
    arrow(ax, *bottom(b_pca), mid(b_km)[0], top(b_km)[1])
    # also direct from matrix to GMM/K-means (primary path)
    arrow(ax, 4.5, bottom(b_mat)[1], mid(b_gmm)[0], top(b_gmm)[1])
    arrow(ax, 9.0, bottom(b_mat)[1], mid(b_km)[0], top(b_km)[1])
    arrow(ax, *bottom(b_gmm), *top(b_soft))
    arrow(ax, *bottom(b_km), *top(b_hard))
    arrow(ax, *bottom(b_soft), 4.5, top(b_eval)[1])
    arrow(ax, *bottom(b_hard), 9.0, top(b_eval)[1])
    arrow(ax, *bottom(b_eval), *top(b_out))

    ax.text(
        6.75,
        0.35,
        "Updated vs earlier draft: full AEE inputs · BIC-selected GMM · demography v2 · boundary heterogeneity & scenario stability",
        ha="center",
        va="center",
        fontsize=7.8,
        color=ACCENT,
        style="italic",
    )

    fig.tight_layout(pad=0.4)
    for path in (OUT_PNG, OUT_PDF, OUT_SVG):
        kw = dict(bbox_inches="tight", facecolor=BG)
        if path.suffix.lower() == ".png":
            fig.savefig(path, dpi=400, **kw)
        else:
            fig.savefig(path, **kw)
        print("Saved:", path)
    plt.close(fig)


if __name__ == "__main__":
    main()
