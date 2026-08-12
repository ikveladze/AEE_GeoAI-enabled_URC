# ============================================================
# Odense AEE Feature Space Analysis
# PCA + UMAP representation of Accessibility–Environment–Equity indicators
# CRS: EPSG:25832
# ============================================================

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "Map Layers").exists() else SCRIPT_DIR
AEE_OUTPUTS_DIR = SCRIPT_DIR
TARGET_CRS = "EPSG:25832"

OUTPUT_DIR = SCRIPT_DIR / "odense_aee_feature_space_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_GPKG = OUTPUT_DIR / "odense_aee_feature_space_500m.gpkg"
FEATURE_MATRIX_CSV = OUTPUT_DIR / "odense_aee_feature_matrix.csv"
FEATURE_MATRIX_STANDARDISED_CSV = OUTPUT_DIR / "odense_aee_feature_matrix_standardised.csv"
PCA_LOADINGS_CSV = OUTPUT_DIR / "odense_aee_pca_loadings.csv"
PCA_VARIANCE_CSV = OUTPUT_DIR / "odense_aee_pca_explained_variance.csv"
CLUSTER_SUMMARY_CSV = OUTPUT_DIR / "odense_aee_cluster_summary.csv"
VALIDATION_TXT = OUTPUT_DIR / "odense_aee_feature_space_validation.txt"

PCA_SCATTER_PNG = OUTPUT_DIR / "odense_aee_pca_scatter.png"
UMAP_SCATTER_PNG = OUTPUT_DIR / "odense_aee_umap_scatter.png"
PCA_CLUSTER_MAP_PNG = OUTPUT_DIR / "odense_aee_pca_cluster_map.png"
UMAP_CLUSTER_MAP_PNG = OUTPUT_DIR / "odense_aee_umap_cluster_map.png"
FEATURE_CLUSTER_MAP_PNG = OUTPUT_DIR / "odense_aee_feature_cluster_map.png"

N_PCA_COMPONENTS = 6
N_CLUSTERS = 6
RANDOM_STATE = 42

UMAP_N_NEIGHBORS = 20
UMAP_MIN_DIST = 0.10
UMAP_METRIC = "euclidean"

# If True, negative indicators are multiplied by -1 so that all variables point
# towards more favourable conditions. Usually keep False for transparent PCA.
ORIENT_FEATURES_TO_POSITIVE = False


# ============================================================
# 2. INDICATOR CONFIGURATION
# ============================================================

INDICATOR_CONFIG = [
    # Accessibility
    {
        "indicator_id": "walk_access_15m",
        "label": "Walking access 15 min",
        "dimension": "Accessibility",
        "gpkg": AEE_OUTPUTS_DIR / "odense_walking_accessibility_services_500m.gpkg",
        "layer": "walking_accessibility_500m_hexagons",
        "value_col": "walk_access_score",
        "polarity": "positive",
    },
    {
        "indicator_id": "bike_access_5m",
        "label": "Cycling access 5 min",
        "dimension": "Accessibility",
        "gpkg": AEE_OUTPUTS_DIR / "odense_osm_cycling_accessibility_outputs" / "odense_cycling_accessibility_services_500m.gpkg",
        "layer": "cycling_accessibility_500m_hexagons",
        "value_col": "bike_access_score_5m",
        "polarity": "positive",
    },
    {
        "indicator_id": "bike_access_15m",
        "label": "Cycling access 15 min",
        "dimension": "Accessibility",
        "gpkg": AEE_OUTPUTS_DIR / "odense_osm_cycling_accessibility_outputs" / "odense_cycling_accessibility_services_500m.gpkg",
        "layer": "cycling_accessibility_500m_hexagons",
        "value_col": "bike_access_score_15m",
        "polarity": "positive",
    },
    {
        "indicator_id": "pt_stop_access_10m",
        "label": "PT stop access 10 min",
        "dimension": "Accessibility",
        "gpkg": AEE_OUTPUTS_DIR / "odense_osm_public_transport_accessibility_outputs" / "odense_public_transport_stop_accessibility_500m.gpkg",
        "layer": "pt_stop_accessibility_500m_hexagons",
        "value_col": "pt_access_score_10m",
        "polarity": "positive",
    },
    {
        "indicator_id": "pt_stop_access_20m",
        "label": "PT stop access 20 min",
        "dimension": "Accessibility",
        "gpkg": AEE_OUTPUTS_DIR / "odense_osm_public_transport_accessibility_outputs" / "odense_public_transport_stop_accessibility_500m.gpkg",
        "layer": "pt_stop_accessibility_500m_hexagons",
        "value_col": "pt_access_score_20m",
        "polarity": "positive",
    },

    # Environment
    {
        "indicator_id": "green_share_pct",
        "label": "Green-area share",
        "dimension": "Environment",
        "gpkg": AEE_OUTPUTS_DIR / "odense_osm_green_area_share_outputs" / "odense_green_area_share_500m.gpkg",
        "layer": "green_area_share_500m_hexagons",
        "value_col": "green_share_pct",
        "polarity": "positive",
    },
    {
        "indicator_id": "env_quality",
        "label": "Environmental quality",
        "dimension": "Environment",
        "gpkg": AEE_OUTPUTS_DIR / "odense_multisource_environmental_index_outputs" / "odense_multisource_environmental_index_500m.gpkg",
        "layer": "environmental_index_500m_hexagons",
        "value_col": "env_quality_score",
        "polarity": "positive",
    },
    {
        "indicator_id": "env_burden",
        "label": "Environmental burden",
        "dimension": "Environment",
        "gpkg": AEE_OUTPUTS_DIR / "odense_multisource_environmental_index_outputs" / "odense_multisource_environmental_index_500m.gpkg",
        "layer": "environmental_index_500m_hexagons",
        "value_col": "env_burden_score",
        "polarity": "negative",
    },
    {
        "indicator_id": "builtup_pct",
        "label": "Built-up share",
        "dimension": "Environment",
        "gpkg": AEE_OUTPUTS_DIR / "odense_multisource_environmental_index_outputs" / "odense_multisource_environmental_index_500m.gpkg",
        "layer": "environmental_index_500m_hexagons",
        "value_col": "builtup_pct",
        "polarity": "negative",
    },
    {
        "indicator_id": "road_buf_pct",
        "label": "Major-road exposure",
        "dimension": "Environment",
        "gpkg": AEE_OUTPUTS_DIR / "odense_multisource_environmental_index_outputs" / "odense_multisource_environmental_index_500m.gpkg",
        "layer": "environmental_index_500m_hexagons",
        "value_col": "road_buf_pct",
        "polarity": "negative",
    },

    # Optional official environmental variables
    {
        "indicator_id": "air_pollution_mean",
        "label": "Air pollution",
        "dimension": "Environment",
        "gpkg": AEE_OUTPUTS_DIR / "odense_multisource_environmental_index_outputs" / "odense_multisource_environmental_index_500m.gpkg",
        "layer": "environmental_index_500m_hexagons",
        "value_col": "air_pollution_mean",
        "polarity": "negative",
        "optional": True,
    },
    {
        "indicator_id": "heat_mean",
        "label": "Heat exposure",
        "dimension": "Environment",
        "gpkg": AEE_OUTPUTS_DIR / "odense_multisource_environmental_index_outputs" / "odense_multisource_environmental_index_500m.gpkg",
        "layer": "environmental_index_500m_hexagons",
        "value_col": "heat_mean",
        "polarity": "negative",
        "optional": True,
    },
    {
        "indicator_id": "noise_mean",
        "label": "Noise exposure",
        "dimension": "Environment",
        "gpkg": AEE_OUTPUTS_DIR / "odense_multisource_environmental_index_outputs" / "odense_multisource_environmental_index_500m.gpkg",
        "layer": "environmental_index_500m_hexagons",
        "value_col": "noise_mean",
        "polarity": "negative",
        "optional": True,
    },
    {
        "indicator_id": "eco_health_mean",
        "label": "Ecological health",
        "dimension": "Environment",
        "gpkg": AEE_OUTPUTS_DIR / "odense_multisource_environmental_index_outputs" / "odense_multisource_environmental_index_500m.gpkg",
        "layer": "environmental_index_500m_hexagons",
        "value_col": "eco_health_mean",
        "polarity": "positive",
        "optional": True,
    },

    # Morphology / density proxy (NOT treated as validated equity/vulnerability evidence).
    # Official Statistics Denmark age/income grids are required before restoring Equity claims.
    {
        "indicator_id": "pop_density_km2",
        "label": "Population-density proxy",
        "dimension": "Morphology",
        "gpkg": AEE_OUTPUTS_DIR / "odense_demographic_indicator_outputs" / "odense_demographic_indicators_500m.gpkg",
        "layer": "demographic_indicators_500m_hexagons",
        "value_col": "pop_density_km2",
        "polarity": "context",
    },
    # Kept optional for audit continuity; expected to be dropped as constant/invalid
    # or excluded from the revised clustering matrix.
    {
        "indicator_id": "age_vulnerability",
        "label": "Age vulnerability (audit only)",
        "dimension": "Equity",
        "gpkg": AEE_OUTPUTS_DIR / "odense_demographic_indicator_outputs" / "odense_demographic_indicators_500m.gpkg",
        "layer": "demographic_indicators_500m_hexagons",
        "value_col": "age_vulnerability_share",
        "polarity": "negative",
        "optional": True,
    },
    {
        "indicator_id": "socioeconomic_vulnerability",
        "label": "Socio-economic vulnerability (audit only)",
        "dimension": "Equity",
        "gpkg": AEE_OUTPUTS_DIR / "odense_demographic_indicator_outputs" / "odense_demographic_indicators_500m.gpkg",
        "layer": "demographic_indicators_500m_hexagons",
        "value_col": "socioeconomic_vulnerability_share",
        "polarity": "negative",
        "optional": True,
    },
    {
        "indicator_id": "demographic_vulnerability",
        "label": "Demographic vulnerability (excluded; density duplicate)",
        "dimension": "Equity",
        "gpkg": AEE_OUTPUTS_DIR / "odense_demographic_indicator_outputs" / "odense_demographic_indicators_500m.gpkg",
        "layer": "demographic_indicators_500m_hexagons",
        "value_col": "demo_vulnerability_score",
        "polarity": "negative",
        "optional": True,
    },
]

# Force-exclude from PCA/GMM feature matrix even if present in the GeoPackage.
EXCLUDE_FROM_FEATURE_MATRIX = {
    "demographic_vulnerability",  # r≈1 with pop_density_km2; not validated equity
}


def safe_read_layer(gpkg: Path, layer: str) -> gpd.GeoDataFrame | None:
    if not gpkg.exists():
        return None
    try:
        return gpd.read_file(gpkg, layer=layer).to_crs(TARGET_CRS)
    except Exception as e:
        print(f"Could not read {gpkg.name} / {layer}: {e}")
        return None


def assign_cell_key(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Stable cross-indicator key: indicator GeoPackages use different hex_id prefixes."""
    out = gdf.copy()
    centroid = out.geometry.centroid
    out["cell_key"] = (
        centroid.x.round(2).astype(str) + "_" + centroid.y.round(2).astype(str)
    )
    return out


def load_feature_space() -> tuple[gpd.GeoDataFrame, list[dict], list[str]]:
    merged = None
    used_configs = []
    log = ["AEE feature-space loading report", "=" * 45]

    for cfg in INDICATOR_CONFIG:
        gdf = safe_read_layer(cfg["gpkg"], cfg["layer"])

        if gdf is None:
            if not cfg.get("optional", False):
                log.append(f"SKIPPED missing file/layer: {cfg['indicator_id']}")
            continue

        value_col = cfg["value_col"]
        if value_col not in gdf.columns:
            if not cfg.get("optional", False):
                log.append(f"SKIPPED missing value field {value_col}: {cfg['indicator_id']}")
            continue

        gdf = assign_cell_key(gdf)
        tmp = gdf[["cell_key", value_col, "geometry"]].copy()
        tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
        tmp = tmp.rename(columns={value_col: cfg["indicator_id"]})

        if merged is None:
            merged = tmp
        else:
            tmp_no_geom = tmp[["cell_key", cfg["indicator_id"]]].copy()
            merged = merged.merge(tmp_no_geom, on="cell_key", how="inner")

        used_configs.append(cfg)
        log.append(f"OK: {cfg['indicator_id']}")

    if merged is None:
        raise RuntimeError("No AEE indicators could be loaded. Check output GeoPackage paths.")

    merged = gpd.GeoDataFrame(merged, geometry="geometry", crs=TARGET_CRS)
    merged = merged[merged.geometry.notnull()].copy()
    merged["hex_id"] = merged["cell_key"]

    return merged, used_configs, log


def prepare_feature_matrix(
    gdf: gpd.GeoDataFrame,
    used_configs: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    feature_cols = [cfg["indicator_id"] for cfg in used_configs if cfg["indicator_id"] in gdf.columns]
    X_raw = gdf[["hex_id", *feature_cols]].copy()

    usable_cols = []
    dropped = []

    for col in feature_cols:
        if col in EXCLUDE_FROM_FEATURE_MATRIX:
            dropped.append(f"{col}: excluded by revision policy (equity/duplicate)")
            continue
        s = pd.to_numeric(X_raw[col], errors="coerce")
        if s.notna().sum() == 0:
            dropped.append(f"{col}: all missing")
            continue
        if np.isclose(s.dropna().std(), 0):
            dropped.append(f"{col}: constant")
            continue
        usable_cols.append(col)

    X = X_raw[usable_cols].copy()

    if ORIENT_FEATURES_TO_POSITIVE:
        polarity_map = {cfg["indicator_id"]: cfg.get("polarity", "context") for cfg in used_configs}
        for col in usable_cols:
            if polarity_map.get(col) == "negative":
                X[col] = -1 * pd.to_numeric(X[col], errors="coerce")

    imputer = SimpleImputer(strategy="median")
    X_imputed_array = imputer.fit_transform(X)
    X_imputed = pd.DataFrame(X_imputed_array, columns=usable_cols, index=X.index)

    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_imputed)
    X_scaled = pd.DataFrame(X_scaled_array, columns=usable_cols, index=X.index)
    X_scaled.insert(0, "hex_id", X_raw["hex_id"].values)

    imputation_table = pd.DataFrame({
        "feature": usable_cols,
        "median_used_for_imputation": imputer.statistics_,
        "mean_after_imputation": X_imputed.mean(axis=0).values,
        "std_after_imputation": X_imputed.std(axis=0).values,
    })

    X_imputed.insert(0, "hex_id", X_raw["hex_id"].values)
    return X_imputed, X_scaled, imputation_table, dropped


def run_pca(X_scaled: pd.DataFrame):
    feature_cols = [c for c in X_scaled.columns if c != "hex_id"]
    n_components = min(N_PCA_COMPONENTS, len(feature_cols), len(X_scaled))

    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X_scaled[feature_cols])

    pca_cols = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(coords, columns=pca_cols)
    pca_df.insert(0, "hex_id", X_scaled["hex_id"].values)

    loadings = pd.DataFrame(
        pca.components_.T,
        index=feature_cols,
        columns=pca_cols,
    ).reset_index().rename(columns={"index": "feature"})

    variance = pd.DataFrame({
        "component": pca_cols,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_explained_variance": np.cumsum(pca.explained_variance_ratio_),
    })

    return pca_df, loadings, variance, pca


def run_umap(X_scaled: pd.DataFrame) -> pd.DataFrame | None:
    if not HAS_UMAP:
        print("umap-learn is not installed; UMAP will be skipped.")
        return None

    feature_cols = [c for c in X_scaled.columns if c != "hex_id"]

    reducer = umap.UMAP(
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
    )
    coords = reducer.fit_transform(X_scaled[feature_cols])

    return pd.DataFrame({
        "hex_id": X_scaled["hex_id"].values,
        "UMAP1": coords[:, 0],
        "UMAP2": coords[:, 1],
    })


def add_kmeans_clusters(df: pd.DataFrame, cols: list[str], cluster_col: str):
    out = df.copy()
    valid = out[cols].notna().all(axis=1)

    if valid.sum() < N_CLUSTERS:
        out[cluster_col] = np.nan
        return out, None

    km = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=20)
    labels = km.fit_predict(out.loc[valid, cols])
    out.loc[valid, cluster_col] = labels + 1
    out[cluster_col] = out[cluster_col].astype("Int64")

    sil = None
    if N_CLUSTERS > 1 and valid.sum() > N_CLUSTERS:
        sil = silhouette_score(out.loc[valid, cols], labels)

    return out, sil


def plot_scatter(df, x_col, y_col, cluster_col, title, output_path):
    fig, ax = plt.subplots(figsize=(10, 8))

    if cluster_col in df.columns and df[cluster_col].notna().any():
        scatter = ax.scatter(df[x_col], df[y_col], c=df[cluster_col], s=18, alpha=0.75)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(cluster_col)
    else:
        ax.scatter(df[x_col], df[y_col], s=18, alpha=0.75)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(title, fontsize=14, fontweight="bold", loc="left")
    ax.axhline(0, linewidth=0.6)
    ax.axvline(0, linewidth=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_cluster_map(gdf, cluster_col, title, output_path):
    if cluster_col not in gdf.columns or not gdf[cluster_col].notna().any():
        print(f"Cannot plot map; missing/empty cluster field {cluster_col}")
        return

    fig, ax = plt.subplots(figsize=(12, 10))
    gdf.plot(
        ax=ax,
        column=cluster_col,
        categorical=True,
        legend=True,
        edgecolor="white",
        linewidth=0.15,
        alpha=0.95,
    )
    ax.set_title(title, fontsize=16, fontweight="bold", loc="left")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def create_cluster_summary(gdf, cluster_col, feature_cols):
    if cluster_col not in gdf.columns or not gdf[cluster_col].notna().any():
        return pd.DataFrame()

    base = gdf.groupby(cluster_col).agg(n_cells=("hex_id", "count")).reset_index()
    means = gdf.groupby(cluster_col)[feature_cols].mean().reset_index()
    return base.merge(means, on=cluster_col, how="left")


def main():
    validation = ["Odense AEE PCA/UMAP feature-space validation", "=" * 55]
    validation.append(f"HAS_UMAP: {HAS_UMAP}")
    validation.append(f"ORIENT_FEATURES_TO_POSITIVE: {ORIENT_FEATURES_TO_POSITIVE}")

    gdf, used_configs, loading_log = load_feature_space()
    validation.extend(loading_log)
    validation.append("")

    X_imputed, X_scaled, imputation_table, dropped = prepare_feature_matrix(gdf, used_configs)
    feature_cols = [c for c in X_scaled.columns if c != "hex_id"]

    validation.append(f"Cells in merged feature space: {len(gdf)}")
    validation.append(f"Features used: {len(feature_cols)}")
    validation.append("Features:")
    for col in feature_cols:
        validation.append(f"  - {col}")

    if dropped:
        validation.append("")
        validation.append("Dropped features:")
        for d in dropped:
            validation.append(f"  - {d}")

    pca_df, loadings, variance, pca_model = run_pca(X_scaled)
    umap_df = run_umap(X_scaled)

    out_gdf = gdf.copy()
    out_gdf = out_gdf.merge(pca_df, on="hex_id", how="left")

    if umap_df is not None:
        out_gdf = out_gdf.merge(umap_df, on="hex_id", how="left")

    pca_cols_for_cluster = [c for c in ["PC1", "PC2", "PC3"] if c in out_gdf.columns]
    out_gdf, pca_silhouette = add_kmeans_clusters(out_gdf, pca_cols_for_cluster, "cluster_pca")
    validation.append(f"PCA cluster silhouette: {pca_silhouette}")

    if umap_df is not None:
        out_gdf, umap_silhouette = add_kmeans_clusters(out_gdf, ["UMAP1", "UMAP2"], "cluster_umap")
        validation.append(f"UMAP cluster silhouette: {umap_silhouette}")

    feature_cluster_input = X_scaled.copy()
    feature_cluster_input, feature_silhouette = add_kmeans_clusters(feature_cluster_input, feature_cols, "cluster_feature")
    out_gdf = out_gdf.merge(feature_cluster_input[["hex_id", "cluster_feature"]], on="hex_id", how="left")
    validation.append(f"Full feature-space cluster silhouette: {feature_silhouette}")

    out_gdf.to_file(OUTPUT_GPKG, layer="aee_feature_space_500m_hexagons", driver="GPKG")
    X_imputed.to_csv(FEATURE_MATRIX_CSV, index=False)
    X_scaled.to_csv(FEATURE_MATRIX_STANDARDISED_CSV, index=False)
    loadings.to_csv(PCA_LOADINGS_CSV, index=False)
    variance.to_csv(PCA_VARIANCE_CSV, index=False)
    imputation_table.to_csv(OUTPUT_DIR / "odense_aee_imputation_table.csv", index=False)

    cluster_summary = create_cluster_summary(out_gdf, "cluster_feature", feature_cols)
    if not cluster_summary.empty:
        cluster_summary.to_csv(CLUSTER_SUMMARY_CSV, index=False)

    if {"PC1", "PC2"}.issubset(out_gdf.columns):
        plot_scatter(out_gdf, "PC1", "PC2", "cluster_feature", "Odense AEE feature space: PCA", PCA_SCATTER_PNG)
        plot_cluster_map(out_gdf, "cluster_pca", "Odense AEE PCA clusters by 500 m cell", PCA_CLUSTER_MAP_PNG)

    if umap_df is not None and {"UMAP1", "UMAP2"}.issubset(out_gdf.columns):
        plot_scatter(out_gdf, "UMAP1", "UMAP2", "cluster_umap", "Odense AEE feature space: UMAP", UMAP_SCATTER_PNG)
        plot_cluster_map(out_gdf, "cluster_umap", "Odense AEE UMAP clusters by 500 m cell", UMAP_CLUSTER_MAP_PNG)

    plot_cluster_map(out_gdf, "cluster_feature", "Odense AEE clusters in full standardised feature space", FEATURE_CLUSTER_MAP_PNG)

    validation.append("")
    validation.append("PCA explained variance:")
    for _, row in variance.iterrows():
        validation.append(
            f"  {row['component']}: {row['explained_variance_ratio']:.4f}; "
            f"cumulative={row['cumulative_explained_variance']:.4f}"
        )

    validation.append("")
    validation.append("Outputs:")
    validation.append(f"  GeoPackage: {OUTPUT_GPKG}")
    validation.append(f"  Feature matrix: {FEATURE_MATRIX_CSV}")
    validation.append(f"  PCA loadings: {PCA_LOADINGS_CSV}")
    validation.append(f"  PCA variance: {PCA_VARIANCE_CSV}")

    VALIDATION_TXT.write_text("\n".join(validation), encoding="utf-8")

    print(f"GeoPackage saved: {OUTPUT_GPKG}")
    print(f"Feature matrix saved: {FEATURE_MATRIX_CSV}")
    print(f"PCA loadings saved: {PCA_LOADINGS_CSV}")
    print(f"PCA variance saved: {PCA_VARIANCE_CSV}")
    print(f"Validation report saved: {VALIDATION_TXT}")
    print("Finished.")


if __name__ == "__main__":
    main()
