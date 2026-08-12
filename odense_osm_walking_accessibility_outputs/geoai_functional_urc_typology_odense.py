# ============================================================
# Odense GeoAI-derived Functional Urban–Rural Continuum (URC) Typology
# Revised for reviewer-aligned feature selection and GMM model choice
# 500 m hexagonal cells | CRS: EPSG:25832
# ============================================================
#
# Key revisions vs original submission run:
# 1. Clustering uses a nonredundant Accessibility–Environment (+ density) matrix
# 2. Derived composites and functional_urc_score are EXCLUDED from GMM inputs
# 3. demographic_vulnerability excluded (near-perfect duplicate of pop_density;
#    density enters only as morphology; vulnerability kept descriptive)
# 4. GMM component count selected with BIC/AIC (+ secondary diagnostics)
# 5. Smoothing impact and pre-/post-smoothing uncertainty are reported explicitly
# ============================================================

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from libpysal.weights import Queen
    HAS_LIBPYSAL = True
except Exception:
    HAS_LIBPYSAL = False

# ------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "Map Layers").exists() else SCRIPT_DIR
TARGET_CRS = "EPSG:25832"

INPUT_GPKG = SCRIPT_DIR / "odense_aee_feature_space_outputs" / "odense_aee_feature_space_500m.gpkg"
INPUT_LAYER = "aee_feature_space_500m_hexagons"

OUTPUT_DIR = SCRIPT_DIR / "odense_geoai_functional_urc_typology_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_SHP_DIR = OUTPUT_DIR / "odense_geoai_functional_urc_typology_500m_shp"
OUTPUT_SHP_DIR.mkdir(exist_ok=True)

OUTPUT_GPKG = OUTPUT_DIR / "odense_geoai_functional_urc_typology_500m.gpkg"
TYPOLOGY_MAP = OUTPUT_DIR / "odense_geoai_functional_urc_typology_map.png"
URC_SCORE_MAP = OUTPUT_DIR / "odense_functional_urc_score_map.png"
FEATURE_MATRIX_CSV = OUTPUT_DIR / "odense_functional_urc_feature_matrix.csv"
CLUSTER_FEATURE_CSV = OUTPUT_DIR / "odense_functional_urc_clustering_features.csv"
MODEL_SELECTION_CSV = OUTPUT_DIR / "odense_functional_urc_gmm_model_selection.csv"
CORR_CSV = OUTPUT_DIR / "odense_functional_urc_feature_correlation.csv"
VIF_CSV = OUTPUT_DIR / "odense_functional_urc_feature_vif.csv"
CLUSTER_PROFILE_CSV = OUTPUT_DIR / "odense_functional_urc_cluster_profiles.csv"
TYPOLOGY_SUMMARY_CSV = OUTPUT_DIR / "odense_functional_urc_typology_summary.csv"
VALIDATION_TXT = OUTPUT_DIR / "odense_functional_urc_validation.txt"
BIC_PLOT = OUTPUT_DIR / "odense_functional_urc_gmm_bic_aic.png"

RANDOM_STATE = 42
K_CANDIDATES = list(range(2, 11))
APPLY_SPATIAL_SMOOTHING = True
USE_SMOOTHED_CLASS_FOR_LABELS = True
CORR_DROP_THRESHOLD = 0.90
MIN_COMPONENT_SHARE = 0.03
INCLUDE_DENSITY_AS_MORPHOLOGY = True  # v1: density-class–derived pop scaled to KOMM_TOT 2021 (not full DST Kvadratnet)

# Candidate raw indicators only (no composites / URC score).
ACCESSIBILITY_FEATURES = [
    "walk_access_15m",
    "bike_access_15m",
    "pt_stop_access_20m",
]
# Nested short-threshold indicators kept for profiles only, not clustering defaults.
NESTED_THRESHOLD_FEATURES = [
    "bike_access_5m",
    "pt_stop_access_10m",
]
ENVIRONMENT_FEATURES = [
    "green_share_pct",
    "env_quality",
    "env_burden",
    "builtup_pct",
    "road_buf_pct",
]
MORPHOLOGY_FEATURES = ["pop_density_km2"]

# Explicitly excluded from clustering (reviewer R1.4 / equity audit).
EXCLUDED_FROM_CLUSTERING = {
    "demographic_vulnerability",
    "age_vulnerability",
    "socioeconomic_vulnerability",
    "functional_accessibility_norm",
    "environmental_quality_norm",
    "social_vulnerability_norm",
    "urban_density_norm",
    "functional_urc_score",
    "air_pollution_mean",
    "heat_mean",
    "noise_mean",
    "eco_health_mean",
}

FEATURE_PRIORITY = (
    ACCESSIBILITY_FEATURES
    + ["green_share_pct", "env_quality", "builtup_pct", "road_buf_pct", "env_burden"]
    + MORPHOLOGY_FEATURES
)

URC_SCORE_WEIGHTS = {
    "low_accessibility": 0.35,
    "low_density": 0.25,
    "low_builtup": 0.15,
    "green_environment": 0.25,
}

# ------------------------------------------------------------
# 2. HELPERS
# ------------------------------------------------------------

def minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    s = s.fillna(s.mean()) if s.notna().any() else s.fillna(0.0)
    mn, mx = s.min(), s.max()
    if np.isclose(mn, mx):
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


def classify_quintiles(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    s = series.fillna(series.mean()) if series.notna().any() else series.fillna(0.0)
    try:
        ranks = pd.qcut(s, q=5, labels=False, duplicates="drop")
    except ValueError:
        ranks = pd.Series(0, index=s.index)
    rank_map = {b: i + 1 for i, b in enumerate(sorted(pd.Series(ranks).dropna().unique()))}
    rank = pd.Series(ranks, index=s.index).map(rank_map).fillna(1).astype(int)
    labels = ["Very urban", "Urban", "Urban–suburban", "Peri-urban", "Rural/peripheral"]
    cls = rank.map({i + 1: labels[i] for i in range(int(rank.max()))})
    return rank, cls


def available_columns(gdf: gpd.GeoDataFrame, candidates: list[str]) -> list[str]:
    return [c for c in candidates if c in gdf.columns]


def load_aee_feature_space() -> gpd.GeoDataFrame:
    if not INPUT_GPKG.exists():
        raise FileNotFoundError(
            f"Missing AEE feature-space file: {INPUT_GPKG}\n"
            "Run aee_pca_umap_feature_space_odense.py first."
        )
    gdf = gpd.read_file(INPUT_GPKG, layer=INPUT_LAYER).to_crs(TARGET_CRS)
    if "hex_id" not in gdf.columns:
        raise ValueError("Input layer must contain a 'hex_id' field.")
    return gdf[gdf.geometry.notnull()].copy().reset_index(drop=True)


def compute_vif(X: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factors from regressing each column on the others."""
    cols = list(X.columns)
    rows = []
    for i, col in enumerate(cols):
        y = X[col].values
        Z = np.delete(X.values, i, axis=1)
        # Add intercept
        Z = np.column_stack([np.ones(len(Z)), Z])
        try:
            beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
            yhat = Z @ beta
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif = np.inf if r2 >= 1 else 1.0 / max(1e-12, (1.0 - r2))
        except Exception:
            vif = np.nan
        rows.append({"feature": col, "vif": float(vif), "r_squared_aux": float(r2) if "r2" in locals() else np.nan})
    return pd.DataFrame(rows).sort_values("vif", ascending=False)


def drop_correlated_features(X: pd.DataFrame, priority: list[str], threshold: float) -> tuple[list[str], list[dict]]:
    cols = list(X.columns)
    priority_rank = {c: i for i, c in enumerate(priority)}
    corr = X.corr().abs()
    dropped = []
    keep = set(cols)

    # Prefer deterministic pairwise drops using priority.
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            r = corr.loc[a, b]
            if pd.notna(r) and r >= threshold:
                pairs.append((r, a, b))
    pairs.sort(reverse=True)

    for r, a, b in pairs:
        if a not in keep or b not in keep:
            continue
        rank_a = priority_rank.get(a, 10_000)
        rank_b = priority_rank.get(b, 10_000)
        drop = b if rank_a <= rank_b else a
        keep.discard(drop)
        dropped.append({"feature": drop, "reason": f"|r|={r:.3f} with {a if drop == b else b} >= {threshold}"})

    kept = [c for c in priority if c in keep] + [c for c in cols if c in keep and c not in priority]
    return kept, dropped


# ------------------------------------------------------------
# 3. SUBINDICES AND FEATURE MATRIX
# ------------------------------------------------------------

def add_composite_subindices(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Descriptive composites / URC score for mapping & profiles — NOT for GMM inputs."""
    out = gdf.copy()

    acc_cols = available_columns(out, ["walk_access_15m", "bike_access_15m", "pt_stop_access_20m", "bike_access_5m", "pt_stop_access_10m"])
    env_pos_cols = available_columns(out, ["green_share_pct", "env_quality"])
    env_burden_cols = available_columns(out, ["env_burden", "builtup_pct", "road_buf_pct"])
    density_cols = available_columns(out, MORPHOLOGY_FEATURES)

    out["functional_accessibility_norm"] = (
        pd.concat([minmax(out[c]) for c in acc_cols], axis=1).mean(axis=1) if acc_cols else np.nan
    )
    env_positive = (
        pd.concat([minmax(out[c]) for c in env_pos_cols], axis=1).mean(axis=1)
        if env_pos_cols
        else pd.Series(0.0, index=out.index)
    )
    env_burden = (
        pd.concat([minmax(out[c]) for c in env_burden_cols], axis=1).mean(axis=1)
        if env_burden_cols
        else pd.Series(0.0, index=out.index)
    )
    out["environmental_quality_norm"] = minmax(env_positive - env_burden)
    # Density retained as morphology/urbanity proxy; not labelled as equity vulnerability.
    out["urban_density_norm"] = (
        pd.concat([minmax(out[c]) for c in density_cols], axis=1).mean(axis=1) if density_cols else np.nan
    )
    out["social_vulnerability_norm"] = np.nan  # unavailable / invalid in this Odense run

    low_access = 1 - out["functional_accessibility_norm"].fillna(out["functional_accessibility_norm"].mean())
    low_density = 1 - out["urban_density_norm"].fillna(out["urban_density_norm"].mean())
    builtup_norm = minmax(out["builtup_pct"]) if "builtup_pct" in out.columns else out["urban_density_norm"].fillna(0)
    low_builtup = 1 - builtup_norm
    green_cols = available_columns(out, ["green_share_pct", "env_quality"])
    green_environment = (
        pd.concat([minmax(out[c]) for c in green_cols], axis=1).mean(axis=1)
        if green_cols
        else out["environmental_quality_norm"].fillna(0)
    )

    urc_raw = (
        URC_SCORE_WEIGHTS["low_accessibility"] * low_access
        + URC_SCORE_WEIGHTS["low_density"] * low_density
        + URC_SCORE_WEIGHTS["low_builtup"] * low_builtup
        + URC_SCORE_WEIGHTS["green_environment"] * green_environment
    )
    out["functional_urc_score"] = (minmax(urc_raw) * 100).round(2)
    out["functional_urc_rank"], out["functional_urc_class"] = classify_quintiles(out["functional_urc_score"])
    return out


def prepare_cluster_matrix(gdf: gpd.GeoDataFrame):
    candidates = list(ACCESSIBILITY_FEATURES) + list(ENVIRONMENT_FEATURES)
    if INCLUDE_DENSITY_AS_MORPHOLOGY:
        candidates += list(MORPHOLOGY_FEATURES)
    candidates = available_columns(gdf, candidates)

    dropped = []
    usable = []
    for col in candidates:
        if col in EXCLUDED_FROM_CLUSTERING:
            dropped.append({"feature": col, "reason": "excluded by revision policy"})
            continue
        s = pd.to_numeric(gdf[col], errors="coerce")
        if s.notna().sum() == 0:
            dropped.append({"feature": col, "reason": "all missing"})
        elif np.isclose(s.dropna().std(), 0):
            dropped.append({"feature": col, "reason": "constant"})
        else:
            usable.append(col)

    if len(usable) < 3:
        raise ValueError("Too few usable features for GeoAI typology after filtering.")

    X_imp = pd.DataFrame(
        SimpleImputer(strategy="median").fit_transform(gdf[usable]),
        columns=usable,
        index=gdf.index,
    )
    corr = X_imp.corr()
    kept, corr_dropped = drop_correlated_features(X_imp, FEATURE_PRIORITY, CORR_DROP_THRESHOLD)
    dropped.extend(corr_dropped)
    X_imp = X_imp[kept]

    vif = compute_vif(X_imp)
    # Iteratively drop highest VIF > 10, preferring lower-priority features.
    while True:
        finite = vif.replace([np.inf], np.nan).dropna()
        if finite.empty or finite["vif"].max() <= 10:
            break
        worst = finite.iloc[0]["feature"]
        # Prefer dropping lower-priority feature among those with VIF>10
        high = set(finite.loc[finite["vif"] > 10, "feature"])
        if len(high) > 1:
            worst = sorted(high, key=lambda c: FEATURE_PRIORITY.index(c) if c in FEATURE_PRIORITY else 10_000)[-1]
        if len(X_imp.columns) <= 3:
            break
        X_imp = X_imp.drop(columns=[worst])
        dropped.append({"feature": worst, "reason": f"VIF>{10} (iterative cull)"})
        vif = compute_vif(X_imp)

    X_scaled = pd.DataFrame(StandardScaler().fit_transform(X_imp), columns=X_imp.columns, index=gdf.index)
    X_raw = gdf[["hex_id", *X_imp.columns]].copy()
    X_raw[X_imp.columns] = X_imp
    X_scaled.insert(0, "hex_id", gdf["hex_id"].values)

    corr.to_csv(CORR_CSV)
    vif.to_csv(VIF_CSV, index=False)
    pd.DataFrame({"feature": X_imp.columns}).to_csv(CLUSTER_FEATURE_CSV, index=False)

    return X_raw, X_scaled, list(X_imp.columns), pd.DataFrame(dropped), corr, vif


# ------------------------------------------------------------
# 4. GEOAI CLUSTERING AND INTERPRETATION
# ------------------------------------------------------------

def select_gmm_components(X: np.ndarray) -> tuple[int, pd.DataFrame, GaussianMixture]:
    rows = []
    models = {}
    n = len(X)
    for k in K_CANDIDATES:
        if k >= n:
            continue
        gmm = GaussianMixture(
            n_components=k,
            covariance_type="full",
            random_state=RANDOM_STATE,
            n_init=10,
        )
        gmm.fit(X)
        labels = gmm.predict(X)
        counts = pd.Series(labels).value_counts(normalize=True)
        min_share = float(counts.min()) if len(counts) else 0.0
        sil = silhouette_score(X, labels) if k > 1 and len(np.unique(labels)) > 1 else np.nan
        ch = calinski_harabasz_score(X, labels) if k > 1 and len(np.unique(labels)) > 1 else np.nan
        db = davies_bouldin_score(X, labels) if k > 1 and len(np.unique(labels)) > 1 else np.nan
        rows.append({
            "k": k,
            "bic": float(gmm.bic(X)),
            "aic": float(gmm.aic(X)),
            "log_likelihood": float(gmm.score(X) * n),
            "min_component_share": min_share,
            "n_components_fitted": int(len(counts)),
            "silhouette": sil,
            "calinski_harabasz": ch,
            "davies_bouldin": db,
            "meets_min_size": min_share >= MIN_COMPONENT_SHARE,
        })
        models[k] = gmm

    selection = pd.DataFrame(rows).sort_values("k")
    eligible = selection[selection["meets_min_size"]].copy()
    if eligible.empty:
        eligible = selection.copy()

    # Documented two-stage rule (reported in validation):
    # 1) size-eligible models (min component share >= MIN_COMPONENT_SHARE)
    # 2) prefer silhouette >= 0.10 when available (avoid over-fragmented soft mixtures)
    # 3) among remaining, minimise BIC
    # Pure BIC minimum is also recorded for transparency.
    pure_bic_k = int(eligible.loc[eligible["bic"].idxmin(), "k"])
    sil_ok = eligible[eligible["silhouette"] >= 0.10]
    pool = sil_ok if not sil_ok.empty else eligible
    best_k = int(pool.loc[pool["bic"].idxmin(), "k"])
    selection["selected"] = selection["k"] == best_k
    selection["pure_bic_selected"] = selection["k"] == pure_bic_k
    selection["selection_rule"] = (
        f"min BIC among size-eligible"
        + (" with silhouette>=0.10" if not sil_ok.empty else "")
    )
    return best_k, selection, models[best_k]


def plot_model_selection(selection: pd.DataFrame, selected_k: int, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(selection["k"], selection["bic"], marker="o", label="BIC")
    ax.plot(selection["k"], selection["aic"], marker="s", label="AIC")
    ax.axvline(selected_k, color="#c0392b", linestyle="--", linewidth=1.2, label=f"Selected k={selected_k}")
    ax.set_xlabel("Number of GMM components (k)")
    ax.set_ylabel("Information criterion")
    ax.set_title("GMM model selection (BIC / AIC)", loc="left")
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_geoai_clustering(gdf: gpd.GeoDataFrame, X_scaled: pd.DataFrame, feature_cols: list[str]):
    out = gdf.copy()
    X = X_scaled[feature_cols].values

    best_k, selection, gmm = select_gmm_components(X)
    selection.to_csv(MODEL_SELECTION_CSV, index=False)
    plot_model_selection(selection, best_k, BIC_PLOT)

    gmm_labels = gmm.predict(X) + 1
    proba = gmm.predict_proba(X)
    out["urc_type_gmm"] = gmm_labels
    out["urc_type_probability"] = proba.max(axis=1).round(4)
    out["urc_type_uncertainty"] = (1 - out["urc_type_probability"]).round(4)
    # Classification entropy (bits-normalised to [0,1] by log(k))
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.sum(np.where(proba > 0, proba * np.log(proba), 0.0), axis=1)
    out["urc_type_entropy"] = (entropy / np.log(best_k)).round(4)

    km = KMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=50)
    km_labels = km.fit_predict(X) + 1
    out["urc_type_kmeans"] = km_labels

    metrics = {
        "selected_k": best_k,
        "selected_bic": float(selection.loc[selection["k"] == best_k, "bic"].iloc[0]),
        "selected_aic": float(selection.loc[selection["k"] == best_k, "aic"].iloc[0]),
    }
    if best_k > 1 and len(out) > best_k:
        metrics["gmm_silhouette"] = silhouette_score(X, gmm_labels)
        metrics["kmeans_silhouette"] = silhouette_score(X, km_labels)
        metrics["gmm_calinski_harabasz"] = calinski_harabasz_score(X, gmm_labels)
        metrics["kmeans_calinski_harabasz"] = calinski_harabasz_score(X, km_labels)
        metrics["gmm_davies_bouldin"] = davies_bouldin_score(X, gmm_labels)
        metrics["kmeans_davies_bouldin"] = davies_bouldin_score(X, km_labels)
    return out, metrics, selection


def spatial_majority_smoothing(gdf: gpd.GeoDataFrame, source_col: str, target_col: str) -> gpd.GeoDataFrame:
    out = gdf.copy()
    if not HAS_LIBPYSAL:
        print("libpysal not installed; spatial smoothing skipped.")
        out[target_col] = out[source_col]
        out["smoothing_label_changed"] = False
        return out

    w = Queen.from_dataframe(out, use_index=True)
    smoothed = out[source_col].copy()
    for idx in out.index:
        neighbours = w.neighbors.get(idx, [])
        if not neighbours:
            continue
        vals = out.loc[neighbours, source_col].dropna()
        if vals.empty:
            continue
        maj = vals.mode().iloc[0]
        if out.loc[idx, source_col] != maj and (vals == maj).sum() >= max(2, len(vals) / 2):
            smoothed.loc[idx] = maj
    out[target_col] = smoothed.astype(int)
    out["smoothing_label_changed"] = out[source_col].astype(int) != out[target_col].astype(int)
    return out


def profile_clusters(gdf: gpd.GeoDataFrame, cluster_col: str) -> pd.DataFrame:
    profile_cols = [
        "functional_accessibility_norm",
        "environmental_quality_norm",
        "urban_density_norm",
        "functional_urc_score",
        "urc_type_probability",
        "urc_type_uncertainty",
        "urc_type_entropy",
    ]
    for col in [
        "walk_access_15m",
        "bike_access_15m",
        "pt_stop_access_20m",
        "green_share_pct",
        "env_quality",
        "env_burden",
        "builtup_pct",
        "road_buf_pct",
        "pop_density_km2",
    ]:
        if col in gdf.columns:
            profile_cols.append(col)
    profile_cols = [c for c in profile_cols if c in gdf.columns]
    return gdf.groupby(cluster_col).agg(
        n_cells=("hex_id", "count"),
        area_km2=("geometry", lambda s: s.area.sum() / 1_000_000),
        **{f"mean_{c}": (c, "mean") for c in profile_cols},
    ).reset_index()


def assign_functional_labels(gdf: gpd.GeoDataFrame, cluster_col: str):
    """Label clusters using accessibility / environment / density only (no equity claim)."""
    out = gdf.copy()
    profiles = profile_clusters(out, cluster_col)
    profiles["cell_share_pct"] = (profiles["n_cells"] / len(out) * 100).round(2)

    def pct_rank(col):
        if col not in profiles.columns:
            return pd.Series(0.5, index=profiles.index)
        return profiles[col].rank(pct=True)

    profiles["r_access"] = pct_rank("mean_functional_accessibility_norm")
    profiles["r_env"] = pct_rank("mean_environmental_quality_norm")
    profiles["r_density"] = pct_rank("mean_urban_density_norm")
    profiles["r_urc"] = pct_rank("mean_functional_urc_score")

    labels = {}
    for _, row in profiles.iterrows():
        cid = int(row[cluster_col])
        access, env, dens, urc = row["r_access"], row["r_env"], row["r_density"], row["r_urc"]
        if dens >= 0.75 and access >= 0.70:
            label = "Central accessible urban core"
        elif dens >= 0.60 and access >= 0.55 and env < 0.50:
            label = "Compact built-up / environmental burden zone"
        elif access >= 0.60 and dens < 0.70 and urc < 0.65:
            label = "Accessible suburban / local-centre zone"
        elif access < 0.55 and dens < 0.55 and urc >= 0.45:
            label = "Low-access transition zone"
        elif env >= 0.65 and urc >= 0.55:
            label = "Green peri-urban / rural fringe"
        elif urc >= 0.70 and access < 0.50:
            label = "Peripheral low-access rural zone"
        else:
            label = "Mixed functional transition zone"
        labels[cid] = label

    profiles["functional_urc_label"] = profiles[cluster_col].map(labels)
    out["functional_urc_type"] = out[cluster_col].map(labels)
    return out, profiles


# ------------------------------------------------------------
# 5. OUTPUTS AND MAPS
# ------------------------------------------------------------

def make_typology_palette(labels):
    """Fixed 4-colour manuscript palette; extras cycle through the same set."""
    # User palette: light teal, med blue, med cyan (#oobobe → #00b0be), orange
    preferred = {
        "Low-access transition zone": "#9fc8c8",  # light teal
        "Accessible suburban / local-centre zone": "#00b0be",  # med cyan
        "Compact built-up / environmental burden zone": "#1a80bb",  # med blue
        "Central accessible urban core": "#ea801c",  # orange
        # legacy / alternate labels
        "Mixed functional transition zone": "#00b0be",
        "Green peri-urban / rural fringe": "#9fc8c8",
        "Peripheral low-access rural zone": "#9fc8c8",
    }
    fallback = ["#9fc8c8", "#00b0be", "#1a80bb", "#ea801c"]
    unique = sorted(pd.Series(labels).dropna().unique())
    unused = [c for c in fallback if c not in preferred.values()]
    out = {}
    fi = 0
    for label in unique:
        if label in preferred:
            out[label] = preferred[label]
        else:
            out[label] = fallback[fi % len(fallback)]
            fi += 1
    return out


def plot_typology_map(gdf, label_col, output_path):
    palette = make_typology_palette(gdf[label_col].tolist())
    plot_gdf = gdf.copy()
    plot_gdf["map_color"] = plot_gdf[label_col].map(palette)
    fig, ax = plt.subplots(figsize=(14, 10))
    plot_gdf.plot(ax=ax, color=plot_gdf["map_color"], edgecolor="white", linewidth=0.12, alpha=0.96)
    ax.set_title(
        "GeoAI-derived Functional Urban-Rural Continuum Typology",
        fontsize=14,
        fontweight="bold",
        loc="left",
    )
    ax.set_axis_off()
    patches = [mpatches.Patch(facecolor=color, edgecolor="gray", label=label) for label, color in palette.items()]
    ax.legend(handles=patches, title="Functional URC type", loc="lower left", frameon=True, fontsize=9, title_fontsize=10)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_urc_score_map(gdf, output_path):
    fig, ax = plt.subplots(figsize=(14, 10))
    gdf.plot(ax=ax, column="functional_urc_score", cmap="viridis", legend=True, edgecolor="white", linewidth=0.12, alpha=0.96)
    ax.set_title(
        "Odense Functional Urban–Rural Continuum Score\n0 = more urban/central, 100 = more rural/peripheral",
        fontsize=16,
        fontweight="bold",
        loc="left",
    )
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_typology_summary(gdf, label_col):
    return gdf.groupby(label_col).agg(
        n_cells=("hex_id", "count"),
        area_km2=("geometry", lambda s: s.area.sum() / 1_000_000),
        mean_urc_score=("functional_urc_score", "mean"),
        mean_accessibility=("functional_accessibility_norm", "mean"),
        mean_environmental_quality=("environmental_quality_norm", "mean"),
        mean_urban_density=("urban_density_norm", "mean"),
        mean_uncertainty=("urc_type_uncertainty", "mean"),
        mean_entropy=("urc_type_entropy", "mean"),
        pct_smoothing_changed=("smoothing_label_changed", "mean"),
    ).reset_index()


def write_validation(metrics, feature_cols, dropped, profiles, selection, n_changed_smooth):
    lines = [
        "Odense GeoAI-derived Functional URC Typology validation (REVISED)",
        "=" * 70,
        f"Input: {INPUT_GPKG}",
        f"Selected GMM components (BIC-primary): {metrics.get('selected_k')}",
        f"Selected BIC: {metrics.get('selected_bic')}",
        f"Selected AIC: {metrics.get('selected_aic')}",
        f"Spatial smoothing applied: {APPLY_SPATIAL_SMOOTHING}",
        f"libpysal available: {HAS_LIBPYSAL}",
        f"Labels changed by Queen smoothing: {n_changed_smooth}",
        "",
        "IMPORTANT DATA CAVEATS",
        "- Clustering excludes demographic_vulnerability / age / socio fields.",
        "- pop_density_km2 is used only as a morphology/urbanity proxy if enabled.",
        "- Population density is dasymetric (StatBank SOGN × BBR), soft-checked vs DST classes; paid 100 m Kvadratnet not used.",
        "- Derived composites and functional_urc_score are excluded from GMM inputs.",
        "",
        "Features used in GMM clustering:",
    ]
    lines.extend([f"  - {c}" for c in feature_cols])
    if not dropped.empty:
        lines.append("")
        lines.append("Dropped / excluded features:")
        for _, row in dropped.iterrows():
            lines.append(f"  - {row['feature']}: {row['reason']}")

    lines.append("")
    lines.append("GMM candidate comparison (see also model-selection CSV/PNG):")
    for _, row in selection.iterrows():
        flag = " <-- selected" if row.get("selected") else ""
        lines.append(
            f"  k={int(row['k'])}: BIC={row['bic']:.1f}; AIC={row['aic']:.1f}; "
            f"min_share={row['min_component_share']:.3f}; sil={row['silhouette']:.4f}{flag}"
        )

    lines.append("")
    lines.append("Cluster validation metrics (selected model):")
    for k, v in metrics.items():
        if isinstance(v, (float, int, np.floating, np.integer)):
            lines.append(f"  {k}: {float(v):.4f}")
        else:
            lines.append(f"  {k}: {v}")

    lines.append("")
    lines.append("Assigned cluster labels:")
    cluster_col = profiles.columns[0]
    for _, row in profiles.iterrows():
        lines.append(f"  Cluster {row[cluster_col]}: {row['functional_urc_label']}")

    lines.append("")
    lines.append("Interpretation notes:")
    lines.append("- Typology is exploratory functional classification from AE (+ optional density) indicators.")
    lines.append("- GMM posterior uncertainty is computed before spatial smoothing.")
    lines.append("- Smoothing is cartographic; cells with smoothing_label_changed=True should be treated cautiously.")
    lines.append("- Entropy is normalised by log(k); higher values indicate softer membership.")
    VALIDATION_TXT.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------------------------------------
# 6. MAIN
# ------------------------------------------------------------

def main():
    gdf = load_aee_feature_space()
    gdf = add_composite_subindices(gdf)
    X_raw, X_scaled, feature_cols, dropped, corr, vif = prepare_cluster_matrix(gdf)
    gdf, metrics, selection = run_geoai_clustering(gdf, X_scaled, feature_cols)

    if APPLY_SPATIAL_SMOOTHING:
        gdf = spatial_majority_smoothing(gdf, "urc_type_gmm", "urc_type_gmm_smooth")
        cluster_for_labels = "urc_type_gmm_smooth" if USE_SMOOTHED_CLASS_FOR_LABELS else "urc_type_gmm"
    else:
        gdf["urc_type_gmm_smooth"] = gdf["urc_type_gmm"]
        gdf["smoothing_label_changed"] = False
        cluster_for_labels = "urc_type_gmm"

    n_changed = int(gdf["smoothing_label_changed"].sum()) if "smoothing_label_changed" in gdf.columns else 0
    # Uncertainty remains the pre-smoothing GMM posterior; flag smoothed cells explicitly.
    gdf["uncertainty_applies_to"] = "pre_smoothing_gmm_membership"

    gdf, profiles = assign_functional_labels(gdf, cluster_for_labels)
    gdf.to_file(OUTPUT_GPKG, layer="functional_urc_typology_500m_hexagons", driver="GPKG")

    shp_cols = [
        "hex_id",
        "functional_urc_score",
        "functional_urc_class",
        "functional_urc_type",
        "urc_type_gmm",
        "urc_type_gmm_smooth",
        "urc_type_probability",
        "urc_type_uncertainty",
        "urc_type_entropy",
        "smoothing_label_changed",
        "geometry",
    ]
    shp = gdf[[c for c in shp_cols if c in gdf.columns]].copy().rename(
        columns={
            "functional_urc_score": "urc_score",
            "functional_urc_class": "urc_class",
            "functional_urc_type": "urc_type",
            "urc_type_gmm": "gmm_type",
            "urc_type_gmm_smooth": "gmm_smo",
            "urc_type_probability": "type_prob",
            "urc_type_uncertainty": "type_unc",
            "urc_type_entropy": "type_ent",
            "smoothing_label_changed": "smo_chg",
        }
    )
    shp.to_file(OUTPUT_SHP_DIR / "odense_functional_urc_typology_500m.shp", driver="ESRI Shapefile", encoding="UTF-8")

    X_raw.to_csv(FEATURE_MATRIX_CSV, index=False)
    profiles.to_csv(CLUSTER_PROFILE_CSV, index=False)
    summary = create_typology_summary(gdf, "functional_urc_type")
    summary["cell_share_pct"] = (summary["n_cells"] / len(gdf) * 100).round(2)
    summary.to_csv(TYPOLOGY_SUMMARY_CSV, index=False)

    plot_typology_map(gdf, "functional_urc_type", TYPOLOGY_MAP)
    plot_urc_score_map(gdf, URC_SCORE_MAP)
    write_validation(metrics, feature_cols, dropped, profiles, selection, n_changed)

    print(f"Selected k (BIC): {metrics.get('selected_k')}")
    print(f"Features used: {feature_cols}")
    print(f"Smoothing changed labels: {n_changed} / {len(gdf)}")
    print(f"GeoPackage saved: {OUTPUT_GPKG}")
    print(f"Model selection: {MODEL_SELECTION_CSV}")
    print(f"BIC/AIC plot: {BIC_PLOT}")
    print("Finished.")


if __name__ == "__main__":
    main()
