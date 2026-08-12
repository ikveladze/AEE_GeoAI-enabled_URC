#!/usr/bin/env python3
"""Minimalist monochrome scientific / CS modelling flowchart.

Orthogonal (Manhattan) connectors, typed node shapes, critical parameters
as implemented for the Odense AEE → GeoAI functional URC pipeline.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, Circle, Rectangle
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PNG = SCRIPT_DIR / "odense_aee_cs_modelling_flowchart_mono.png"
OUT_PDF = SCRIPT_DIR / "odense_aee_cs_modelling_flowchart_mono.pdf"
OUT_SVG = SCRIPT_DIR / "odense_aee_cs_modelling_flowchart_mono.svg"

# Monochrome palette
BG = "#FFFFFF"
INK = "#111111"
MUTED = "#555555"
FILL = "#F4F4F4"
FILL_IO = "#EAEAEA"
FILL_MODEL = "#DEDEDE"
FILL_EVAL = "#D6D6D6"
BAND = "#F0F0F0"
RULE = "#BDBDBD"


class Canvas:
    def __init__(self, w=18.0, h=28.5):
        self.w, self.h = w, h
        self.fig, self.ax = plt.subplots(figsize=(w, h))
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.set_facecolor(BG)
        self.fig.patch.set_facecolor(BG)
        self.ax.axis("off")
        self.nodes = {}

    def band(self, y0, y1, label):
        self.ax.add_patch(Rectangle((0.25, y0), self.w - 0.5, y1 - y0, facecolor=BAND, edgecolor="none", zorder=0))
        self.ax.text(0.45, y1 - 0.18, label, ha="left", va="top", fontsize=8.5, fontweight="bold", color=MUTED, zorder=1)

    def _text(self, x, y, title, body=None, fs=7.6, ts=8.6):
        if body:
            self.ax.text(x, y + 0.12, title, ha="center", va="center", fontsize=ts, fontweight="bold", color=INK, zorder=3)
            self.ax.text(x, y - 0.18, body, ha="center", va="center", fontsize=fs, color=MUTED, linespacing=1.15, zorder=3)
        else:
            self.ax.text(x, y, title, ha="center", va="center", fontsize=ts, fontweight="bold", color=INK, linespacing=1.15, zorder=3)

    def process(self, key, x, y, w, h, title, body=None, fill=FILL):
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="square,pad=0",
                facecolor=fill, edgecolor=INK, linewidth=1.05, zorder=2,
            )
        )
        self._text(x + w / 2, y + h / 2, title, body)
        self.nodes[key] = (x, y, w, h)
        return self.nodes[key]

    def io(self, key, x, y, w, h, title, body=None):
        # parallelogram (lean-right)
        skew = 0.18
        pts = [
            (x + skew, y),
            (x + w, y),
            (x + w - skew, y + h),
            (x, y + h),
        ]
        self.ax.add_patch(Polygon(pts, closed=True, facecolor=FILL_IO, edgecolor=INK, linewidth=1.05, zorder=2))
        self._text(x + w / 2, y + h / 2, title, body)
        self.nodes[key] = (x, y, w, h)
        return self.nodes[key]

    def model(self, key, x, y, w, h, title, body=None):
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                facecolor=FILL_MODEL, edgecolor=INK, linewidth=1.15, zorder=2,
            )
        )
        self._text(x + w / 2, y + h / 2, title, body, fs=7.3, ts=8.4)
        self.nodes[key] = (x, y, w, h)
        return self.nodes[key]

    def decision(self, key, x, y, w, h, title, body=None):
        cx, cy = x + w / 2, y + h / 2
        pts = [(cx, y), (x + w, cy), (cx, y + h), (x, cy)]
        self.ax.add_patch(Polygon(pts, closed=True, facecolor="#FFFFFF", edgecolor=INK, linewidth=1.15, zorder=2))
        self._text(cx, cy, title, body, fs=6.8, ts=7.8)
        self.nodes[key] = (x, y, w, h)
        return self.nodes[key]

    def terminal(self, key, x, y, w, h, title, body=None):
        self.ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.35",
                facecolor=FILL, edgecolor=INK, linewidth=1.2, zorder=2,
            )
        )
        self._text(x + w / 2, y + h / 2, title, body, fs=7.4, ts=8.8)
        self.nodes[key] = (x, y, w, h)
        return self.nodes[key]

    def bus_to_targets(self, sources, y_bus, targets):
        """Drop from each source to a horizontal bus, then up into each target."""
        xs_src = []
        for k in sources:
            x, y0, w, h = self.nodes[k]
            cx = x + w / 2
            xs_src.append(cx)
            self.ax.plot([cx, cx], [y0, y_bus], color=INK, linewidth=0.95, zorder=1)
        xs_tgt = []
        for k in targets:
            x, y, w, h = self.nodes[k]
            cx = x + w / 2
            xs_tgt.append(cx)
            self.ax.plot([cx, cx], [y_bus, y + h], color=INK, linewidth=0.95, zorder=1)
            self.ax.plot(
                [cx - 0.08, cx, cx + 0.08],
                [y + h + 0.08, y + h, y + h + 0.08],
                color=INK,
                linewidth=0.95,
                zorder=1,
            )
        x0, x1 = min(xs_src + xs_tgt), max(xs_src + xs_tgt)
        self.ax.plot([x0, x1], [y_bus, y_bus], color=INK, linewidth=0.95, zorder=1)

    def down(self, a, b, label=None):
        """Vertical orthogonal link bottom(a) → top(b), with optional jog."""
        ax_, ay, aw, ah = self.nodes[a]
        bx, by, bw, bh = self.nodes[b]
        x1, y1 = ax_ + aw / 2, ay
        x2, y2 = bx + bw / 2, by + bh
        if abs(x1 - x2) < 1e-6:
            self.ax.plot([x1, x2], [y1, y2], color=INK, linewidth=0.95, zorder=1)
        else:
            midy = (y1 + y2) / 2
            self.ax.plot([x1, x1, x2, x2], [y1, midy, midy, y2], color=INK, linewidth=0.95, zorder=1)
        self.ax.plot(
            [x2 - 0.08, x2, x2 + 0.08],
            [y2 + 0.08, y2, y2 + 0.08],
            color=INK,
            linewidth=0.95,
            zorder=1,
        )
        if label:
            self.ax.text((x1 + x2) / 2 + 0.1, (y1 + y2) / 2, label, fontsize=6.5, color=MUTED)


def main():
    C = Canvas(18.0, 29.0)

    # Title
    C.ax.text(
        9.0, 28.55,
        "Detailed computational modelling pipeline (flowchart)",
        ha="center", va="center", fontsize=14, fontweight="bold", color=INK,
    )
    C.ax.text(
        9.0, 28.15,
        "AEE feature engineering  →  unsupervised GeoAI URC typology   |   "
        "Odense: 500 m hex (n = 1,539) · EPSG:25832 · random_state = 42",
        ha="center", va="center", fontsize=8.2, color=MUTED,
    )

    # Legend
    C.ax.text(0.5, 27.75, "Legend:", fontsize=7.5, fontweight="bold", color=INK)
    C.process("_L1", 1.4, 27.55, 1.5, 0.35, "process", fill=FILL)
    C.io("_L2", 3.1, 27.55, 1.5, 0.35, "I/O data")
    C.model("_L3", 4.8, 27.55, 1.6, 0.35, "model")
    C.decision("_L4", 6.6, 27.45, 1.5, 0.55, "decision")
    C.ax.text(8.5, 27.72, "orthogonal connectors · monochrome · as implemented", fontsize=7.2, color=MUTED)

    # ===================== A =====================
    C.band(25.55, 27.35, "A  ·  Spatial frame & data ingestion")
    C.terminal(
        "start", 6.4, 26.45, 5.2, 0.7,
        "START",
        "Study area = Odense Municipality",
    )
    C.process(
        "hex", 5.7, 25.7, 6.6, 0.6,
        "Spatial unit construction",
        "H: 500 m hex tessellation · n = 1,539 · EPSG:25832 · area ≈ 304.37 km² · key = hex_id",
    )
    C.io(
        "osm", 0.5, 24.55, 5.5, 0.9,
        "OSM ingestion (OSMnx / Overpass)",
        "NetworkX MultiDiGraph walk/bike · POIs · PT stops\n"
        "no GTFS · snap thresholds ≈ 200–250 m",
    )
    C.io(
        "off", 6.25, 24.55, 5.5, 0.9,
        "Official + statistical inputs",
        "GeoDanmark (Datafordeler GEODKV) · BBR BYG_BOLIG_\n"
        "StatBank SOGN/POSTNR/FOLK1A · DST 1 km density classes",
    )
    C.io(
        "aux", 12.0, 24.55, 5.5, 0.9,
        "Supporting env. layers (optional)",
        "WDPA ∪ LBST §3 · summer LST (CLIM4cities)\n"
        "state-road Lden noise · interim NO₂ (CAMS)",
    )

    # ===================== B =====================
    C.band(21.35, 24.35, "B  ·  Accessibility feature computation (network algorithms)")
    C.process(
        "walk", 0.5, 22.55, 5.5, 1.55,
        "Walking accessibility (15 min)",
        "algo: multi-source Dijkstra\n"
        "v = 1.34 m/s (4.8 km/h) · T = 15 min\n"
        "d_max ≈ 1,206 m · decay a = exp(−d/700)\n"
        "6 weighted service categories\n"
        "minmax → [0,100] · out: walk_access_15m",
    )
    C.process(
        "bike", 6.25, 22.55, 5.5, 1.55,
        "Cycling accessibility (5 & 15 min)",
        "algo: Dijkstra on bike network\n"
        "v = 15 km/h (= 4.167 m/s)\n"
        "d_max ≈ 1,250 m (5′) / 3,750 m (15′)\n"
        "decay a = exp(−d/1800)\n"
        "out: bike_access_5m, bike_access_15m",
    )
    C.process(
        "pt", 12.0, 22.55, 5.5, 1.55,
        "PT stop-access potential (10 & 20 min)",
        "first-/last-mile walk to OSM stops\n"
        "v = 1.34 m/s · β = 1,000 m\n"
        "d_max ≈ 804 m (10′) / 1,608 m (20′)\n"
        "index = 0.5·all-stops + 0.5·category\n"
        "out: pt_stop_access_10m / _20m",
    )

    # ===================== C =====================
    C.band(18.55, 21.15, "C  ·  Environment & equity feature computation")
    C.process(
        "env", 0.5, 19.0, 5.5, 1.35,
        "Environmental indicators",
        "polygon shares: green, blue, eco, built-up\n"
        "major-road buffer = 50 m → road_buf_pct\n"
        "quality / burden composites → [0,100]",
    )
    C.process(
        "demo", 6.25, 19.0, 5.5, 1.35,
        "Dasymetric demography v2",
        "SOGN × BBR BYG_BOLIG_ weights\n"
        "soft-constrain to DST density classes\n"
        "scale to FOLK1A ≈ 213,431\n"
        "ages: POSTNR1 × BBR · out: pop_density_km2",
    )
    C.process(
        "vuln", 12.0, 19.0, 5.5, 1.35,
        "Vulnerability indices (descriptive)",
        "age + socio-economic shares\n"
        "composite demo vulnerability [0,100]\n"
        "EXCLUDED from GMM inputs\n"
        "migration/origin fields excluded",
    )

    # ===================== D =====================
    C.band(15.55, 18.35, "D  ·  Feature-space assembly & quality control")
    C.process(
        "merge", 1.2, 17.15, 5.0, 0.95,
        "Merge → design matrix X_raw",
        "left-join by hex_id · n = 1,539\n"
        "drop constant / fully missing columns",
    )
    C.process(
        "select", 6.5, 17.15, 5.0, 0.95,
        "Clustering feature selection",
        "priority: AE + density morphology\n"
        "drop |r| > 0.90 · iterative VIF > 10",
    )
    C.process(
        "scale", 11.8, 17.15, 5.0, 0.95,
        "Imputation + standardisation",
        "median impute · StandardScaler (z)\n"
        "export corr / VIF tables · Z ∈ Rⁿˣᵖ",
    )
    C.process(
        "feats", 4.5, 15.85, 9.0, 0.85,
        "Clustering candidate set (priority)",
        "walk_access_15m · bike_access_15m · pt_stop_access_20m · green_share_pct · env_quality ·\n"
        "env_burden · builtup_pct · road_buf_pct · pop_density_km2   |   composites & URC score excluded",
    )

    # ===================== E =====================
    C.band(11.85, 15.35, "E  ·  Unsupervised models")
    C.model(
        "pca", 0.5, 13.55, 5.3, 1.45,
        "PCA + UMAP (exploratory)",
        "PCA: n_components ≤ 6\n"
        "UMAP: n_neighbors = 20, min_dist = 0.10\n"
        "interpret gradients only\n"
        "NOT final typology model",
    )
    C.model(
        "gmm", 6.35, 13.55, 5.3, 1.45,
        "GMM primary typology model",
        "p(z)=Σ w_k N(z; μ_k, Σ_k)\n"
        "covariance = full · n_init = 10\n"
        "sweep K ∈ {2…10} · seed = 42\n"
        "select K* by BIC (+ size≥0.03, sil≥0.10)",
    )
    C.model(
        "km", 12.2, 13.55, 5.3, 1.45,
        "K-means robustness benchmark",
        "n_clusters = K*\n"
        "n_init = 50 · seed = 42\n"
        "hard labels only\n"
        "diagnostics / agreement",
    )
    C.decision(
        "ksel", 6.7, 12.05, 4.6, 1.1,
        "Model selection",
        "this run: K* = 5 (BIC-primary)",
    )

    # ===================== F =====================
    C.band(8.55, 11.65, "F  ·  Posterior decoding, scoring & spatial regularisation")
    C.process(
        "post", 0.5, 9.55, 5.5, 1.75,
        "Posterior decoding",
        "responsibilities γ_ik = P(k|z_i)\n"
        "ŷ_i = argmax_k γ_ik\n"
        "uncertainty u_i = 1 − max_k γ_ik\n"
        "export membership matrix",
    )
    C.process(
        "smooth", 6.25, 9.55, 5.5, 1.75,
        "Queen-contiguity smoothing",
        "W = Queen adjacency\n"
        "majority-neighbour vote\n"
        "→ 4 interpreted map labels:\n"
        "central core · suburban/local-centre\n"
        "compact/burden · low-access transition",
    )
    C.process(
        "score", 12.0, 9.55, 5.5, 1.75,
        "Continuous URC score s ∈ [0,100]",
        "s ∝ 0.35·low_access\n"
        "   + 0.25·low_density\n"
        "   + 0.15·low_built-up\n"
        "   + 0.25·green/env. character\n"
        "0 = urban/central · 100 = rural",
    )

    # ===================== G =====================
    C.band(5.55, 8.35, "G  ·  Evaluation, benchmarking & uncertainty quantification")
    C.process(
        "unc", 0.5, 6.0, 5.5, 2.0,
        "Uncertainty surfaces",
        "1) GMM membership uncertainty\n"
        "   map(u_i = 1 − max γ)\n"
        "2) boundary heterogeneity\n"
        "   (Queen neighbour label mix)\n"
        "3) scenario stability\n"
        "   (feature subsets · K = 4…8)",
        fill=FILL_EVAL,
    )
    C.process(
        "bench", 6.25, 6.0, 5.5, 2.0,
        "External / single-domain benchmarks",
        "quintile references (density, access,\n"
        "green, env., vulnerability)\n"
        "single-domain K-means subsets\n"
        "agreement: ARI, NMI\n"
        "conventional vs AEE mismatch map",
        fill=FILL_EVAL,
    )
    C.process(
        "agree", 12.0, 6.0, 5.5, 2.0,
        "Algorithmic agreement",
        "dominant K-means pairing per GMM class\n"
        "binary map:\n"
        "  agree = #a6cee3\n"
        "  unstable = #1f78b4\n"
        "silhouette / CH / DB reported",
        fill=FILL_EVAL,
    )

    # ===================== H =====================
    C.band(2.55, 5.35, "H  ·  Outputs & reproducibility artefacts")
    C.io(
        "out", 2.5, 3.35, 13.0, 1.65,
        "Deliverables",
        "GPKG: indicators · typology · benchmarking/uncertainty\n"
        "PNG maps: accessibility · environment · demography · typology · URC score · uncertainty / agreement\n"
        "tables: cluster profiles · BIC/AIC · corr/VIF · validation reports\n"
        "orchestration: run_pipeline_v1.py · regenerate_all_maps_styled.py",
    )
    C.terminal(
        "end", 6.5, 2.55, 5.0, 0.55,
        "END",
        "reproducible GeoAI URC typology",
    )

    # Constants footer
    C.ax.add_patch(Rectangle((0.4, 0.25), 17.2, 2.05, facecolor="#F7F7F7", edgecolor=INK, linewidth=0.9))
    C.ax.text(0.65, 2.05, "Critical constants (as implemented)", fontsize=8.2, fontweight="bold", color=INK, va="top")
    C.ax.text(
        0.65, 1.55,
        "walk: v=1.34 m/s, β=700 m, T=15 → d_max≈1206 m   |   "
        "bike: v=15 km/h, β=1800 m, T∈{5,15}   |   "
        "PT: v=1.34 m/s, β=1000 m, T∈{10,20}   |   road buffer=50 m",
        fontsize=6.9, color=MUTED, va="top",
    )
    C.ax.text(
        0.65, 1.1,
        "GMM: full cov, n_init=10, K∈[2,10], min share≥0.03, silhouette prefer≥0.10, BIC select → K*=5, seed=42   |   "
        "K-means: n_init=50 at K*",
        fontsize=6.9, color=MUTED, va="top",
    )
    C.ax.text(
        0.65, 0.65,
        "URC weights: 0.35 / 0.25 / 0.15 / 0.25   |   "
        "corr drop |r|>0.90 · VIF>10   |   "
        "UMAP: n_neighbors=20, min_dist=0.10   |   "
        "demography: SOGN×BBR→FOLK1A; DST soft-constraint",
        fontsize=6.9, color=MUTED, va="top",
    )

    # ---- Connectors (orthogonal) ----
    C.down("start", "hex")
    C.bus_to_targets(["hex"], 25.55, ["osm", "off", "aux"])
    C.bus_to_targets(["osm", "off", "aux"], 24.2, ["walk", "bike", "pt"])
    C.bus_to_targets(["walk", "bike", "pt"], 22.25, ["env", "demo", "vuln"])
    C.bus_to_targets(["env", "demo", "vuln"], 18.75, ["merge"])
    C.down("merge", "select")
    C.down("select", "scale")
    C.down("scale", "feats")
    C.bus_to_targets(["feats"], 15.55, ["pca", "gmm", "km"])
    C.down("gmm", "ksel")
    C.bus_to_targets(["ksel"], 11.85, ["post", "smooth", "score"])
    C.bus_to_targets(["post", "smooth", "score"], 9.25, ["unc", "bench", "agree"])
    # K-means → algorithmic agreement (side path)
    kx, ky, kw, kh = C.nodes["km"]
    ax_, ay, aw, ah = C.nodes["agree"]
    C.ax.plot(
        [kx + kw / 2, 17.4, 17.4, ax_ + aw / 2],
        [ky, ky, ay + ah, ay + ah],
        color=INK,
        linewidth=0.95,
        zorder=1,
        linestyle="--",
    )
    C.ax.text(17.55, (ky + ay + ah) / 2, "benchmark", fontsize=6.2, color=MUTED, rotation=90, va="center")
    C.bus_to_targets(["unc", "bench", "agree"], 5.7, ["out"])
    C.down("out", "end")

    C.fig.tight_layout(pad=0.2)
    for path in (OUT_PNG, OUT_PDF, OUT_SVG):
        kw = dict(bbox_inches="tight", facecolor=BG)
        if path.suffix.lower() == ".png":
            C.fig.savefig(path, dpi=400, **kw)
        else:
            C.fig.savefig(path, **kw)
        print("Saved:", path)
    plt.close(C.fig)


if __name__ == "__main__":
    main()
