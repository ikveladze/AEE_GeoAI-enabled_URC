# ============================================================
# Odense Functional URC Benchmarking and Uncertainty Assessment
# 500 m hexagonal cells | GeoAI-derived AEE typology
# CRS: EPSG:25832
# ============================================================
#
# PURPOSE
# -------
# This script evaluates the robustness, benchmark agreement, and uncertainty
# of the GeoAI-derived functional Urban–Rural Continuum (URC) typology.
#
# It assumes that you have already run:
#   1. aee_pca_umap_feature_space_odense.py
#   2. geoai_functional_urc_typology_odense.py
#
# Main outputs:
#   - Benchmark classifications:
#       population-density benchmark
#       built-up intensity benchmark
#       green-area benchmark
#       accessibility-only benchmark
#       environment-only benchmark
#       equity-only benchmark
#   - Agreement metrics:
#       Adjusted Rand Index (ARI)
#       Normalised Mutual Information (NMI)
#       cluster-profile cross-tabulations
#   - Uncertainty layers:
#       GMM membership uncertainty
#       neighbour-class boundary heterogeneity
#       scenario-based stability / class-change frequency
#       algorithmic agreement between GMM and K-means
#   - Maps:
#       GMM uncertainty map
#       boundary heterogeneity map
#       scenario stability map
#       GMM–K-means agreement map
#   - Tables:
#       benchmark agreement table
#       typology profile table
#       scenario stability summary
#
# Required packages:
#   pip install geopandas pandas numpy scikit-learn matplotlib libpysal openpyxl
#
# Optional:
#   pip install mapclassify
#
# ============================================================

from __future__ import annotations

import itertools
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from libpysal.weights import Queen
    HAS_LIBPYSAL = True
except Exception:
    HAS_LIBPYSAL = False


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "Map Layers").exists() else SCRIPT_DIR

TARGET_CRS = "EPSG:25832"

URC_GPKG = (
    SCRIPT_DIR
    / "odense_geoai_functional_urc_typology_outputs"
    / "odense_geoai_functional_urc_typology_500m.gpkg"
)
URC_LAYER = "functional_urc_typology_500m_hexagons"

OUTPUT_DIR = SCRIPT_DIR / "odense_urc_benchmarking_uncertainty_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_GPKG = OUTPUT_DIR / "odense_urc_benchmarking_uncertainty_500m.gpkg"
OUTPUT_XLSX = OUTPUT_DIR / "odense_urc_benchmarking_uncertainty_tables.xlsx"
BENCHMARK_METRICS_CSV = OUTPUT_DIR / "odense_urc_benchmark_agreement_metrics.csv"
TYPOLOGY_PROFILE_CSV = OUTPUT_DIR / "odense_urc_typology_profiles.csv"
SCENARIO_STABILITY_CSV = OUTPUT_DIR / "odense_urc_scenario_stability_summary.csv"
VALIDATION_TXT = OUTPUT_DIR / "odense_urc_benchmarking_uncertainty_validation.txt"

MAP_GMM_UNCERTAINTY = OUTPUT_DIR / "odense_urc_gmm_membership_uncertainty_map.png"
MAP_BOUNDARY_HETEROGENEITY = OUTPUT_DIR / "odense_urc_boundary_heterogeneity_map.png"
MAP_SCENARIO_STABILITY = OUTPUT_DIR / "odense_urc_scenario_stability_map.png"
MAP_ALGORITHM_AGREEMENT = OUTPUT_DIR / "odense_urc_gmm_kmeans_agreement_map.png"
MAP_BENCHMARK_DENSITY = OUTPUT_DIR / "odense_urc_density_benchmark_map.png"

RANDOM_STATE = 42
N_TYPES_MAIN = 6
CLUSTER_RANGE = [4, 5, 6, 7, 8]

# Choose the main typology field from the GeoAI URC script.
# If smoothed field is available, it is preferred for cartographic comparison.
MAIN_TYPE_FIELD_CANDIDATES = [
    "functional_urc_type",
    "urc_type_gmm_smooth",
    "urc_type_gmm",
]

GMM_TYPE_FIELD_CANDIDATES = [
    "urc_type_gmm_smooth",
    "urc_type_gmm",
]

KMEANS_TYPE_FIELD_CANDIDATES = [
    "urc_type_kmeans",
]

GMM_PROB_FIELD = "urc_type_probability"
GMM_UNCERTAINTY_FIELD = "urc_type_uncertainty"

# Sensitivity scenarios. Only features available in your GeoPackage will be used.
SCENARIOS = {
    "revised_primary": [
        "walk_access_15m", "bike_access_15m", "pt_stop_access_20m",
        "green_share_pct", "builtup_pct", "pop_density_km2",
    ],
    "accessibility_only": [
        "walk_access_15m", "bike_access_15m", "pt_stop_access_20m",
    ],
    "environment_only": [
        "green_share_pct", "env_quality", "env_burden", "builtup_pct", "road_buf_pct",
    ],
    "morphology_density_only": [
        "pop_density_km2", "builtup_pct", "green_share_pct",
    ],
    "access_environment_only": [
        "walk_access_15m", "bike_access_15m", "pt_stop_access_20m",
        "green_share_pct", "env_quality", "builtup_pct", "road_buf_pct",
    ],
}

# Benchmark variables used to create simple reference classifications.
BENCHMARK_VARIABLES = {
    "population_density_quintile": "pop_density_km2",
    "builtup_intensity_quintile": "builtup_pct",
    "green_share_quintile": "green_share_pct",
    "functional_urc_score_quintile": "functional_urc_score",
    "accessibility_quintile": "functional_accessibility_norm",
    "environmental_quality_quintile": "environmental_quality_norm",
    "social_vulnerability_quintile": "social_vulnerability_norm",
    "urban_density_quintile": "urban_density_norm",
}


# ============================================================
# 2. BASIC HELPERS
# ============================================================

def first_existing_column(gdf: gpd.GeoDataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in gdf.columns:
            return c
    return None


def minmax(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").astype(float)
    if s.notna().any():
        s = s.fillna(s.mean())
    else:
        s = s.fillna(0.0)
    mn, mx = s.min(), s.max()
    if np.isclose(mn, mx):
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


def classify_quantiles(series: pd.Series, q: int = 5, prefix: str = "Q") -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() == 0 or np.isclose(s.dropna().std(), 0):
        return pd.Series("Not available", index=series.index)

    labels = [f"{prefix}{i}" for i in range(1, q + 1)]
    try:
        return pd.qcut(s, q=q, labels=labels, duplicates="drop").astype(str)
    except ValueError:
        return pd.Series("Not available", index=series.index)


def prepare_feature_matrix(gdf: gpd.GeoDataFrame, feature_candidates: list[str]) -> tuple[pd.DataFrame, list[str]]:
    features = []
    for col in feature_candidates:
        if col in gdf.columns:
            s = pd.to_numeric(gdf[col], errors="coerce")
            if s.notna().sum() > 0 and not np.isclose(s.dropna().std(), 0):
                features.append(col)

    if len(features) < 2:
        raise ValueError(f"Too few usable features. Available: {features}")

    X_raw = gdf[features].copy()

    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imp)

    return pd.DataFrame(X_scaled, columns=features, index=gdf.index), features


def run_kmeans_labels(X: pd.DataFrame, n_clusters: int) -> np.ndarray:
    km = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=30)
    return km.fit_predict(X) + 1


def run_gmm_labels_and_uncertainty(X: pd.DataFrame, n_clusters: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type="full",
        random_state=RANDOM_STATE,
        n_init=10,
    )
    labels = gmm.fit_predict(X) + 1
    probs = gmm.predict_proba(X)
    max_prob = probs.max(axis=1)
    uncertainty = 1 - max_prob
    return labels, max_prob, uncertainty


def cluster_metrics(X: pd.DataFrame, labels: np.ndarray) -> dict:
    if len(np.unique(labels)) < 2:
        return {
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan,
        }

    return {
        "silhouette": silhouette_score(X, labels),
        "calinski_harabasz": calinski_harabasz_score(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels),
    }


def contingency_table(gdf: gpd.GeoDataFrame, type_col: str, benchmark_col: str) -> pd.DataFrame:
    tab = pd.crosstab(gdf[type_col], gdf[benchmark_col], dropna=False)
    tab.index.name = type_col
    tab.columns.name = benchmark_col
    return tab.reset_index()


# ============================================================
# 3. LOAD INPUT
# ============================================================

def load_urc_typology() -> gpd.GeoDataFrame:
    if not URC_GPKG.exists():
        raise FileNotFoundError(
            f"Missing URC typology file: {URC_GPKG}\n"
            "Run geoai_functional_urc_typology_odense.py first."
        )

    gdf = gpd.read_file(URC_GPKG, layer=URC_LAYER).to_crs(TARGET_CRS)

    if "hex_id" not in gdf.columns:
        raise ValueError("Input URC layer must contain hex_id.")

    gdf = gdf[gdf.geometry.notnull()].copy().reset_index(drop=True)
    print(f"Loaded URC typology cells: {len(gdf)}")
    return gdf


# ============================================================
# 4. BENCHMARK CLASSIFICATIONS
# ============================================================

def create_benchmark_classifications(gdf: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, list[str]]:
    out = gdf.copy()
    created = []

    for benchmark_name, col in BENCHMARK_VARIABLES.items():
        if col not in out.columns:
            continue

        out[benchmark_name] = classify_quantiles(out[col], q=5, prefix="Q")
        if out[benchmark_name].nunique(dropna=True) > 1:
            created.append(benchmark_name)

    # Add accessibility-only, environment-only, and equity-only k-means benchmarks.
    for scenario_name in ["accessibility_only", "environment_only", "morphology_density_only", "access_environment_only"]:
        try:
            X, features = prepare_feature_matrix(out, SCENARIOS[scenario_name])
            labels = run_kmeans_labels(X, min(N_TYPES_MAIN, max(2, len(out) - 1)))
            colname = f"benchmark_{scenario_name}"
            out[colname] = labels.astype(str)
            created.append(colname)
        except Exception as e:
            print(f"Could not create benchmark {scenario_name}: {e}")

    return out, created


def compute_benchmark_agreement(gdf: gpd.GeoDataFrame, main_type_col: str, benchmark_cols: list[str]) -> pd.DataFrame:
    rows = []

    main_labels = gdf[main_type_col].astype(str)

    for bcol in benchmark_cols:
        valid = main_labels.notna() & gdf[bcol].notna()
        if valid.sum() == 0:
            continue

        benchmark_labels = gdf.loc[valid, bcol].astype(str)

        rows.append({
            "benchmark": bcol,
            "n_cells": int(valid.sum()),
            "adjusted_rand_index": adjusted_rand_score(main_labels.loc[valid], benchmark_labels),
            "normalised_mutual_information": normalized_mutual_info_score(main_labels.loc[valid], benchmark_labels),
            "main_typology_classes": int(main_labels.loc[valid].nunique()),
            "benchmark_classes": int(benchmark_labels.nunique()),
        })

    return pd.DataFrame(rows)


# ============================================================
# 5. CLUSTER PROFILE BENCHMARKING
# ============================================================

def create_typology_profiles(gdf: gpd.GeoDataFrame, main_type_col: str) -> pd.DataFrame:
    profile_vars = [
        "walk_access_15m",
        "bike_access_15m",
        "pt_stop_access_20m",
        "functional_accessibility_norm",
        "green_share_pct",
        "env_quality",
        "env_burden",
        "builtup_pct",
        "road_buf_pct",
        "environmental_quality_norm",
        "pop_density_km2",
        "age_vulnerability",
        "socioeconomic_vulnerability",
        "demographic_vulnerability",
        "social_vulnerability_norm",
        "urban_density_norm",
        "functional_urc_score",
        "urc_type_uncertainty",
    ]
    profile_vars = [c for c in profile_vars if c in gdf.columns]

    agg = {
        "hex_id": "count",
        "geometry": lambda s: s.area.sum() / 1_000_000,
    }
    for c in profile_vars:
        agg[c] = "mean"

    profiles = gdf.groupby(main_type_col).agg(agg).reset_index()
    profiles = profiles.rename(columns={"hex_id": "n_cells", "geometry": "area_km2"})
    profiles["cell_share_pct"] = (profiles["n_cells"] / len(gdf) * 100).round(2)

    return profiles


# ============================================================
# 6. UNCERTAINTY ASSESSMENT
# ============================================================

def add_gmm_uncertainty_classes(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()

    if GMM_UNCERTAINTY_FIELD not in out.columns:
        if GMM_PROB_FIELD in out.columns:
            out[GMM_UNCERTAINTY_FIELD] = 1 - pd.to_numeric(out[GMM_PROB_FIELD], errors="coerce")
        else:
            out[GMM_UNCERTAINTY_FIELD] = np.nan

    u = pd.to_numeric(out[GMM_UNCERTAINTY_FIELD], errors="coerce")

    bins = [-0.001, 0.20, 0.40, 0.60, 1.001]
    labels = [
        "Low uncertainty",
        "Moderate uncertainty",
        "High uncertainty",
        "Very high uncertainty",
    ]
    out["gmm_uncertainty_class"] = pd.cut(u, bins=bins, labels=labels).astype(str)
    out.loc[u.isna(), "gmm_uncertainty_class"] = "Not available"

    return out


def add_algorithmic_agreement(gdf: gpd.GeoDataFrame, gmm_col: str) -> gpd.GeoDataFrame:
    out = gdf.copy()

    kmeans_col = first_existing_column(out, KMEANS_TYPE_FIELD_CANDIDATES)

    if kmeans_col is None:
        out["gmm_kmeans_agreement"] = np.nan
        out["gmm_kmeans_agreement_class"] = "Not available"
        return out

    # Because cluster IDs are arbitrary, direct numeric equality is not meaningful.
    # We calculate agreement as whether each GMM type has a dominant K-means type
    # and whether the cell belongs to that dominant pairing.
    dominant_map = (
        out.groupby(gmm_col)[kmeans_col]
        .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else np.nan)
        .to_dict()
    )

    out["dominant_kmeans_for_gmm"] = out[gmm_col].map(dominant_map)
    out["gmm_kmeans_agreement"] = (out[kmeans_col] == out["dominant_kmeans_for_gmm"]).astype(int)
    out["gmm_kmeans_agreement_class"] = np.where(
        out["gmm_kmeans_agreement"] == 1,
        "Agreement with dominant K-means pairing",
        "Algorithmically unstable",
    )

    return out


def add_boundary_heterogeneity(gdf: gpd.GeoDataFrame, type_col: str) -> gpd.GeoDataFrame:
    out = gdf.copy()

    if not HAS_LIBPYSAL:
        print("libpysal not installed; boundary heterogeneity cannot be calculated.")
        out["boundary_heterogeneity"] = np.nan
        out["boundary_heterogeneity_class"] = "Not available"
        return out

    print("Calculating queen-contiguity boundary heterogeneity...")
    w = Queen.from_dataframe(out, use_index=True)

    heterogeneity_values = []

    for idx in out.index:
        neighbours = w.neighbors.get(idx, [])
        if not neighbours:
            heterogeneity_values.append(np.nan)
            continue

        neighbour_classes = out.loc[neighbours, type_col].astype(str)
        own_class = str(out.loc[idx, type_col])

        different_share = (neighbour_classes != own_class).mean()
        heterogeneity_values.append(float(different_share))

    out["boundary_heterogeneity"] = heterogeneity_values

    bins = [-0.001, 0.25, 0.50, 0.75, 1.001]
    labels = [
        "Low boundary heterogeneity",
        "Moderate boundary heterogeneity",
        "High boundary heterogeneity",
        "Very high boundary heterogeneity",
    ]
    out["boundary_heterogeneity_class"] = pd.cut(
        out["boundary_heterogeneity"],
        bins=bins,
        labels=labels,
    ).astype(str)
    out.loc[out["boundary_heterogeneity"].isna(), "boundary_heterogeneity_class"] = "Not available"

    return out


def run_scenario_sensitivity(gdf: gpd.GeoDataFrame, main_type_col: str) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    out = gdf.copy()
    summary_rows = []

    # Generate labels for each scenario and cluster count.
    scenario_label_cols = []

    for scenario_name, features in SCENARIOS.items():
        try:
            X, used_features = prepare_feature_matrix(out, features)
        except Exception as e:
            summary_rows.append({
                "scenario": scenario_name,
                "k": np.nan,
                "status": f"skipped: {e}",
                "features_used": "",
                "silhouette": np.nan,
                "calinski_harabasz": np.nan,
                "davies_bouldin": np.nan,
                "ari_vs_main_typology": np.nan,
                "nmi_vs_main_typology": np.nan,
            })
            continue

        for k in CLUSTER_RANGE:
            if len(out) <= k:
                continue

            try:
                labels, max_prob, uncertainty = run_gmm_labels_and_uncertainty(X, k)
                col = f"scenario_{scenario_name}_k{k}"
                out[col] = labels.astype(str)
                scenario_label_cols.append(col)

                metrics = cluster_metrics(X, labels)

                main_labels = out[main_type_col].astype(str)
                scenario_labels = out[col].astype(str)

                summary_rows.append({
                    "scenario": scenario_name,
                    "k": k,
                    "status": "ok",
                    "features_used": ", ".join(used_features),
                    "silhouette": metrics["silhouette"],
                    "calinski_harabasz": metrics["calinski_harabasz"],
                    "davies_bouldin": metrics["davies_bouldin"],
                    "ari_vs_main_typology": adjusted_rand_score(main_labels, scenario_labels),
                    "nmi_vs_main_typology": normalized_mutual_info_score(main_labels, scenario_labels),
                    "mean_membership_uncertainty": float(np.mean(uncertainty)),
                })

            except Exception as e:
                summary_rows.append({
                    "scenario": scenario_name,
                    "k": k,
                    "status": f"error: {e}",
                    "features_used": ", ".join(used_features),
                    "silhouette": np.nan,
                    "calinski_harabasz": np.nan,
                    "davies_bouldin": np.nan,
                    "ari_vs_main_typology": np.nan,
                    "nmi_vs_main_typology": np.nan,
                })

    scenario_summary = pd.DataFrame(summary_rows)

    # Cell-level retained-class stability after label matching:
    # Cluster IDs are arbitrary across scenarios, so each scenario labelling is
    # rematched to the main typology via maximum-overlap (Hungarian) assignment.
    # Stability = share of scenarios whose rematched label equals the main label.
    from scipy.optimize import linear_sum_assignment

    main_labels = out[main_type_col].astype(str)
    main_classes = sorted(main_labels.dropna().unique())
    stability_hits = np.zeros(len(out), dtype=float)
    stability_den = np.zeros(len(out), dtype=float)

    for col in scenario_label_cols:
        scen = out[col].astype(str)
        scen_classes = sorted(scen.dropna().unique())
        if not main_classes or not scen_classes:
            continue

        contingency = pd.crosstab(main_labels, scen)
        contingency = contingency.reindex(index=main_classes, columns=scen_classes, fill_value=0)
        # Maximize overlap: Hungarian on cost = -overlap
        cost = -contingency.values
        r_ind, c_ind = linear_sum_assignment(cost)
        mapping = {scen_classes[c]: main_classes[r] for r, c in zip(r_ind, c_ind)}
        matched = scen.map(mapping)
        agree = (matched == main_labels).astype(float)
        valid = matched.notna() & main_labels.notna()
        stability_hits[valid.values] += agree[valid].values
        stability_den[valid.values] += 1.0

    with np.errstate(invalid="ignore", divide="ignore"):
        stability_scores = np.where(stability_den > 0, stability_hits / stability_den, np.nan)
    out["scenario_stability"] = stability_scores
    out["scenario_stability_method"] = "hungarian_matched_retained_class"

    bins = [-0.001, 0.40, 0.60, 0.80, 1.001]
    labels = [
        "Highly unstable",
        "Uncertain / transitional",
        "Moderately stable",
        "Highly stable",
    ]
    out["scenario_stability_class"] = pd.cut(
        out["scenario_stability"],
        bins=bins,
        labels=labels,
    ).astype(str)
    out.loc[out["scenario_stability"].isna(), "scenario_stability_class"] = "Not available"

    return out, scenario_summary


# ============================================================
# 7. MAPS
# ============================================================

def plot_numeric_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    output_path: Path,
    cmap: str = "viridis",
):
    if column not in gdf.columns:
        print(f"Map skipped; missing column: {column}")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    gdf.plot(
        ax=ax,
        column=column,
        cmap=cmap,
        legend=True,
        edgecolor="white",
        linewidth=0.12,
        alpha=0.96,
    )
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Map saved: {output_path}")


def plot_categorical_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    output_path: Path,
):
    if column not in gdf.columns:
        print(f"Map skipped; missing column: {column}")
        return

    fig, ax = plt.subplots(figsize=(14, 10))
    gdf.plot(
        ax=ax,
        column=column,
        categorical=True,
        legend=True,
        edgecolor="white",
        linewidth=0.12,
        alpha=0.96,
    )
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Map saved: {output_path}")


# ============================================================
# 8. OUTPUTS
# ============================================================

def write_excel_outputs(
    benchmark_metrics: pd.DataFrame,
    typology_profiles: pd.DataFrame,
    scenario_summary: pd.DataFrame,
    crosstabs: dict[str, pd.DataFrame],
):
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        benchmark_metrics.to_excel(writer, sheet_name="Benchmark metrics", index=False)
        typology_profiles.to_excel(writer, sheet_name="Typology profiles", index=False)
        scenario_summary.to_excel(writer, sheet_name="Scenario sensitivity", index=False)

        for name, tab in crosstabs.items():
            sheet = name[:31]
            tab.to_excel(writer, sheet_name=sheet, index=False)

        metadata = pd.DataFrame({
            "Item": [
                "Purpose",
                "Main typology",
                "Benchmarking",
                "Uncertainty",
                "Scenario stability",
            ],
            "Description": [
                "Benchmark and uncertainty assessment for GeoAI-derived functional URC typology.",
                "Functional URC types from Gaussian Mixture Model / interpreted typology labels.",
                "Agreement with density, built-up, green, accessibility-only, environment-only and equity-only benchmark classifications.",
                "GMM membership uncertainty, boundary heterogeneity, GMM–K-means agreement, and scenario stability.",
                "Cell-level stability across different feature configurations and cluster numbers.",
            ],
        })
        metadata.to_excel(writer, sheet_name="Metadata", index=False)

        wb = writer.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            for col_cells in ws.columns:
                max_len = 0
                column = col_cells[0].column_letter
                for cell in col_cells:
                    if cell.value is not None:
                        max_len = max(max_len, len(str(cell.value)))
                ws.column_dimensions[column].width = min(max(max_len + 2, 12), 45)


def write_validation(
    gdf: gpd.GeoDataFrame,
    main_type_col: str,
    gmm_type_col: str,
    benchmark_cols: list[str],
    scenario_summary: pd.DataFrame,
):
    lines = ["Odense URC benchmarking and uncertainty validation", "=" * 60]
    lines.append(f"Input file: {URC_GPKG}")
    lines.append(f"Cells: {len(gdf)}")
    lines.append(f"Main typology field: {main_type_col}")
    lines.append(f"GMM type field: {gmm_type_col}")
    lines.append(f"libpysal available: {HAS_LIBPYSAL}")
    lines.append("")

    lines.append("Benchmark classifications created:")
    for b in benchmark_cols:
        lines.append(f"  - {b}")

    lines.append("")
    if GMM_UNCERTAINTY_FIELD in gdf.columns:
        u = pd.to_numeric(gdf[GMM_UNCERTAINTY_FIELD], errors="coerce")
        lines.append(
            f"GMM uncertainty: min={u.min():.3f}, mean={u.mean():.3f}, max={u.max():.3f}"
        )

    if "boundary_heterogeneity" in gdf.columns:
        b = pd.to_numeric(gdf["boundary_heterogeneity"], errors="coerce")
        lines.append(
            f"Boundary heterogeneity: min={b.min():.3f}, mean={b.mean():.3f}, max={b.max():.3f}"
        )

    if "scenario_stability" in gdf.columns:
        s = pd.to_numeric(gdf["scenario_stability"], errors="coerce")
        lines.append(
            f"Scenario stability: min={s.min():.3f}, mean={s.mean():.3f}, max={s.max():.3f}"
        )

    lines.append("")
    lines.append("Scenario sensitivity runs:")
    if not scenario_summary.empty:
        ok = scenario_summary[scenario_summary["status"] == "ok"]
        lines.append(f"  successful runs: {len(ok)}")
        lines.append(f"  total runs: {len(scenario_summary)}")
    else:
        lines.append("  none")

    lines.append("")
    lines.append("Interpretation:")
    lines.append("- High GMM uncertainty indicates fuzzy functional transition zones.")
    lines.append("- High boundary heterogeneity indicates cells located at fragmented typology boundaries.")
    lines.append("- Low scenario stability indicates sensitivity to feature selection or cluster number.")
    lines.append("- GMM–K-means disagreement indicates algorithmic instability.")
    lines.append("- Stable cells are those with low membership uncertainty, low boundary heterogeneity, and high scenario stability.")

    VALIDATION_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Validation report saved: {VALIDATION_TXT}")


# ============================================================
# 9. MAIN
# ============================================================

def main():
    gdf = load_urc_typology()

    main_type_col = first_existing_column(gdf, MAIN_TYPE_FIELD_CANDIDATES)
    gmm_type_col = first_existing_column(gdf, GMM_TYPE_FIELD_CANDIDATES)

    if main_type_col is None:
        raise ValueError(f"No main typology field found. Checked: {MAIN_TYPE_FIELD_CANDIDATES}")

    if gmm_type_col is None:
        gmm_type_col = main_type_col

    # 1. Benchmarking
    gdf, benchmark_cols = create_benchmark_classifications(gdf)
    benchmark_metrics = compute_benchmark_agreement(gdf, main_type_col, benchmark_cols)
    typology_profiles = create_typology_profiles(gdf, main_type_col)

    # 2. Cross-tabs
    crosstabs = {}
    for bcol in benchmark_cols:
        try:
            crosstabs[bcol] = contingency_table(gdf, main_type_col, bcol)
        except Exception as e:
            print(f"Could not create crosstab for {bcol}: {e}")

    # 3. Uncertainty
    gdf = add_gmm_uncertainty_classes(gdf)
    gdf = add_algorithmic_agreement(gdf, gmm_type_col)
    gdf = add_boundary_heterogeneity(gdf, gmm_type_col)
    gdf, scenario_summary = run_scenario_sensitivity(gdf, main_type_col)

    # 4. Save GeoPackage
    gdf.to_file(OUTPUT_GPKG, layer="urc_benchmarking_uncertainty_500m_hexagons", driver="GPKG")

    # 5. Save tables
    benchmark_metrics.to_csv(BENCHMARK_METRICS_CSV, index=False)
    typology_profiles.to_csv(TYPOLOGY_PROFILE_CSV, index=False)
    scenario_summary.to_csv(SCENARIO_STABILITY_CSV, index=False)
    write_excel_outputs(benchmark_metrics, typology_profiles, scenario_summary, crosstabs)

    # 6. Maps
    plot_numeric_map(
        gdf,
        GMM_UNCERTAINTY_FIELD,
        "Functional URC: GMM Membership Uncertainty",
        MAP_GMM_UNCERTAINTY,
        cmap="magma",
    )

    plot_numeric_map(
        gdf,
        "boundary_heterogeneity",
        "Functional URC: Boundary Heterogeneity",
        MAP_BOUNDARY_HETEROGENEITY,
        cmap="magma",
    )

    plot_numeric_map(
        gdf,
        "scenario_stability",
        "Odense Functional URC: Scenario Stability",
        MAP_SCENARIO_STABILITY,
        cmap="viridis",
    )

    plot_categorical_map(
        gdf,
        "gmm_kmeans_agreement_class",
        "Functional URC: GMM–K-means Agreement",
        MAP_ALGORITHM_AGREEMENT,
    )

    if "population_density_quintile" in gdf.columns:
        plot_categorical_map(
            gdf,
            "population_density_quintile",
            "Odense Benchmark Classification: Population Density Quintiles",
            MAP_BENCHMARK_DENSITY,
        )

    # 7. Validation
    write_validation(gdf, main_type_col, gmm_type_col, benchmark_cols, scenario_summary)

    print(f"GeoPackage saved: {OUTPUT_GPKG}")
    print(f"Excel tables saved: {OUTPUT_XLSX}")
    print(f"Benchmark metrics saved: {BENCHMARK_METRICS_CSV}")
    print(f"Typology profiles saved: {TYPOLOGY_PROFILE_CSV}")
    print(f"Scenario sensitivity saved: {SCENARIO_STABILITY_CSV}")
    print("Finished.")


if __name__ == "__main__":
    main()
