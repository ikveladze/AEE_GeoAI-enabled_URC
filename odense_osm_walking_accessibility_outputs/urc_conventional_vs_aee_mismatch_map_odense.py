# ============================================================
# Odense Mismatch Map:
# Conventional Urban–Rural Classification vs AEE / GeoAI Typology
# 500 m hexagonal cells | CRS: EPSG:25832
# ============================================================
#
# PURPOSE
# -------
# This script generates a mismatch map between:
#
#   1. A conventional urban–rural / urbanity classification
#      derived from simple morphology or density indicators, and
#
#   2. The GeoAI-derived AEE functional Urban–Rural Continuum typology.
#
# The mismatch map identifies where the functional AEE typology confirms,
# refines, or contradicts conventional classifications.
#
# Main outputs:
#   - GeoPackage with benchmark class, AEE functional class, mismatch category
#   - mismatch map
#   - agreement map
#   - mismatch-intensity map
#   - cross-tabulation table
#   - mismatch summary table
#   - validation report
#
# Required input:
#   odense_geoai_functional_urc_typology_outputs/
#     odense_geoai_functional_urc_typology_500m.gpkg
#   layer:
#     functional_urc_typology_500m_hexagons
#
# Required packages:
#   pip install geopandas pandas numpy matplotlib scikit-learn openpyxl
#
# ============================================================

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, cohen_kappa_score

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "Map Layers").exists() else SCRIPT_DIR
TARGET_CRS = "EPSG:25832"

# Typology GPKG is produced under SCRIPT_DIR (walking-outputs pipeline root),
# not PROJECT_DIR / Map Layers parent.
INPUT_GPKG = (
    SCRIPT_DIR
    / "odense_geoai_functional_urc_typology_outputs"
    / "odense_geoai_functional_urc_typology_500m.gpkg"
)
INPUT_LAYER = "functional_urc_typology_500m_hexagons"

OUTPUT_DIR = SCRIPT_DIR / "odense_urc_mismatch_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_GPKG = OUTPUT_DIR / "odense_urc_conventional_vs_aee_mismatch_500m.gpkg"
OUTPUT_SHP_DIR = OUTPUT_DIR / "odense_urc_conventional_vs_aee_mismatch_500m_shp"
OUTPUT_SHP_DIR.mkdir(exist_ok=True)

OUTPUT_MISMATCH_MAP = OUTPUT_DIR / "odense_urc_conventional_vs_aee_mismatch_map.png"
OUTPUT_MISMATCH_INTENSITY_MAP = OUTPUT_DIR / "odense_urc_mismatch_intensity_map.png"
OUTPUT_AGREEMENT_MAP = OUTPUT_DIR / "odense_urc_agreement_map.png"
OUTPUT_CONVENTIONAL_MAP = OUTPUT_DIR / "odense_conventional_urban_rural_classification_map.png"
OUTPUT_AEE_MAP = OUTPUT_DIR / "odense_aee_functional_urc_class_map.png"

OUTPUT_XLSX = OUTPUT_DIR / "odense_urc_mismatch_tables.xlsx"
CROSSTAB_CSV = OUTPUT_DIR / "odense_urc_conventional_vs_aee_crosstab.csv"
SUMMARY_CSV = OUTPUT_DIR / "odense_urc_mismatch_summary.csv"
VALIDATION_TXT = OUTPUT_DIR / "odense_urc_mismatch_validation.txt"


# ============================================================
# 2. CLASSIFICATION SETTINGS
# ============================================================
#
# Conventional classification can be derived in two ways:
#
#   OPTION A: Use an existing conventional classification field
#             if your input file already has one.
#
#   OPTION B: Derive one from selected conventional indicators:
#             population density, built-up share, and green/URC score.
#
# The default is OPTION B.
#
# Conventional class direction:
#   1 = Urban core
#   2 = Urban
#   3 = Suburban / peri-urban
#   4 = Rural / peripheral
#
# AEE functional class direction:
#   1 = Functional urban / central
#   2 = Functional urban / mixed
#   3 = Functional transition / suburban
#   4 = Functional rural / peripheral

USE_EXISTING_CONVENTIONAL_FIELD = False
EXISTING_CONVENTIONAL_FIELD = "conventional_urc_class"

# Conventional input indicators. The script uses whichever exist.
CONVENTIONAL_URBAN_POSITIVE = [
    "pop_density_km2",
    "builtup_pct",
    "urban_density_norm",
]

CONVENTIONAL_RURAL_POSITIVE = [
    "green_share_pct",
    "functional_urc_score",
]

# AEE / GeoAI functional typology fields.
AEE_LABEL_FIELD_CANDIDATES = [
    "functional_urc_type",
    "functional_urc_class",
]

AEE_CLUSTER_FIELD_CANDIDATES = [
    "urc_type_gmm_smooth",
    "urc_type_gmm",
]

# If a text label exists, the script maps it to a four-level functional class.
# You can edit these rules if your labels differ.
AEE_LABEL_TO_CLASS_RULES = {
    "Central accessible urban core": "Functional urban core",
    "Compact built-up / environmental burden zone": "Functional urban core",
    "Accessible suburban / local-centre zone": "Functional urban / local centre",
    "Mixed functional transition zone": "Functional transition zone",
    "Vulnerable low-access transition zone": "Functionally vulnerable transition zone",
    "Green peri-urban / rural fringe": "Functional green fringe / rural",
    "Peripheral low-access rural zone": "Functional peripheral rural",
}

# Four-level ordered class mapping for mismatch intensity.
ORDERED_CLASS_VALUE = {
    # Conventional
    "Conventional urban core": 1,
    "Conventional urban": 2,
    "Conventional suburban/peri-urban": 3,
    "Conventional rural/peripheral": 4,

    # AEE functional
    "Functional urban core": 1,
    "Functional urban / local centre": 2,
    "Functional transition zone": 3,
    "Functionally vulnerable transition zone": 3,
    "Functional green fringe / rural": 4,
    "Functional peripheral rural": 4,
}

# Mismatch interpretation threshold.
# Difference 0 = agreement;
# Difference 1 = soft mismatch / refinement;
# Difference >=2 = strong mismatch.
SOFT_MISMATCH_THRESHOLD = 1
STRONG_MISMATCH_THRESHOLD = 2


# ============================================================
# 3. HELPERS
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


def quantile_class_4(series: pd.Series) -> pd.Series:
    """
    Creates four ordered classes from a continuous urbanity score.
    Higher score = more urban.
    """
    s = pd.to_numeric(series, errors="coerce")

    labels = [
        "Conventional rural/peripheral",
        "Conventional suburban/peri-urban",
        "Conventional urban",
        "Conventional urban core",
    ]

    try:
        return pd.qcut(s, q=4, labels=labels, duplicates="drop").astype(str)
    except ValueError:
        return pd.Series("Not available", index=series.index)


def area_km2(geoms: gpd.GeoSeries) -> float:
    return float(geoms.area.sum() / 1_000_000)


# ============================================================
# 4. LOAD INPUT
# ============================================================

def load_typology() -> gpd.GeoDataFrame:
    if not INPUT_GPKG.exists():
        raise FileNotFoundError(
            f"Missing input file: {INPUT_GPKG}\n"
            "Run geoai_functional_urc_typology_odense.py first."
        )

    gdf = gpd.read_file(INPUT_GPKG, layer=INPUT_LAYER).to_crs(TARGET_CRS)

    if "hex_id" not in gdf.columns:
        raise ValueError("Input layer must contain hex_id.")

    gdf = gdf[gdf.geometry.notnull()].copy().reset_index(drop=True)

    print(f"Loaded cells: {len(gdf)}")
    print("Available fields:")
    print(list(gdf.columns))

    return gdf


# ============================================================
# 5. CONVENTIONAL CLASSIFICATION
# ============================================================

def derive_conventional_classification(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()

    if USE_EXISTING_CONVENTIONAL_FIELD and EXISTING_CONVENTIONAL_FIELD in out.columns:
        out["conventional_class"] = out[EXISTING_CONVENTIONAL_FIELD].astype(str)
        out["conventional_class_source"] = f"existing field: {EXISTING_CONVENTIONAL_FIELD}"
        return out

    urban_components = []
    rural_components = []

    for col in CONVENTIONAL_URBAN_POSITIVE:
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce")
            if s.notna().any() and not np.isclose(s.dropna().std(), 0):
                urban_components.append(minmax(s))

    for col in CONVENTIONAL_RURAL_POSITIVE:
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce")
            if s.notna().any() and not np.isclose(s.dropna().std(), 0):
                rural_components.append(minmax(s))

    if not urban_components and not rural_components:
        raise ValueError(
            "Could not derive conventional classification. "
            "Missing population density, built-up, green-share, or URC-score fields."
        )

    if urban_components:
        urban_score = pd.concat(urban_components, axis=1).mean(axis=1)
    else:
        urban_score = pd.Series(0.0, index=out.index)

    if rural_components:
        rural_score = pd.concat(rural_components, axis=1).mean(axis=1)
    else:
        rural_score = pd.Series(0.0, index=out.index)

    # Conventional urbanity score:
    # high density/built-up increases urbanity;
    # green/rural/peripheral character decreases urbanity.
    out["conventional_urbanity_score"] = (minmax(urban_score) - minmax(rural_score))
    out["conventional_urbanity_score"] = (minmax(out["conventional_urbanity_score"]) * 100).round(2)

    out["conventional_class"] = quantile_class_4(out["conventional_urbanity_score"])
    out["conventional_class_source"] = (
        "derived from population density / built-up intensity / green or URC-score indicators"
    )

    return out


# ============================================================
# 6. AEE FUNCTIONAL CLASSIFICATION
# ============================================================

def derive_aee_functional_classification(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()

    label_col = first_existing_column(out, AEE_LABEL_FIELD_CANDIDATES)
    cluster_col = first_existing_column(out, AEE_CLUSTER_FIELD_CANDIDATES)

    if label_col is not None:
        raw_labels = out[label_col].astype(str)

        def map_label(label: str) -> str:
            if label in AEE_LABEL_TO_CLASS_RULES:
                return AEE_LABEL_TO_CLASS_RULES[label]

            low = label.lower()

            if "central" in low or "urban core" in low or "compact built" in low:
                return "Functional urban core"
            if "suburban" in low or "local-centre" in low or "local center" in low:
                return "Functional urban / local centre"
            if "transition" in low or "vulnerable" in low:
                return "Functionally vulnerable transition zone" if "vulnerable" in low else "Functional transition zone"
            if "green" in low or "rural" in low or "peripheral" in low:
                return "Functional green fringe / rural"
            return "Functional transition zone"

        out["aee_functional_class"] = raw_labels.apply(map_label)
        out["aee_functional_class_source"] = f"mapped from label field: {label_col}"
        return out

    if cluster_col is not None and "functional_urc_score" in out.columns:
        # If only clusters exist, order clusters by mean URC score and assign four broad classes.
        cluster_mean = out.groupby(cluster_col)["functional_urc_score"].mean().sort_values()
        ordered_clusters = list(cluster_mean.index)

        cluster_to_order = {cluster: i + 1 for i, cluster in enumerate(ordered_clusters)}
        out["_aee_cluster_order"] = out[cluster_col].map(cluster_to_order)

        # Convert cluster order to four broad classes.
        out["_aee_cluster_order_norm"] = minmax(out["_aee_cluster_order"])

        labels = [
            "Functional urban core",
            "Functional urban / local centre",
            "Functional transition zone",
            "Functional green fringe / rural",
        ]
        out["aee_functional_class"] = pd.qcut(
            out["_aee_cluster_order_norm"],
            q=4,
            labels=labels,
            duplicates="drop",
        ).astype(str)

        out["aee_functional_class_source"] = f"derived from ordered cluster field: {cluster_col}"
        return out

    if "functional_urc_score" in out.columns:
        labels = [
            "Functional urban core",
            "Functional urban / local centre",
            "Functional transition zone",
            "Functional green fringe / rural",
        ]
        out["aee_functional_class"] = pd.qcut(
            out["functional_urc_score"],
            q=4,
            labels=labels,
            duplicates="drop",
        ).astype(str)
        out["aee_functional_class_source"] = "derived from functional_urc_score quartiles"
        return out

    raise ValueError(
        "Could not derive AEE functional classification. "
        "Expected functional_urc_type, functional_urc_class, urc_type_gmm, or functional_urc_score."
    )


# ============================================================
# 7. MISMATCH ASSESSMENT
# ============================================================

def calculate_mismatch(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    out = gdf.copy()

    out["conventional_order"] = out["conventional_class"].map(ORDERED_CLASS_VALUE)
    out["aee_order"] = out["aee_functional_class"].map(ORDERED_CLASS_VALUE)

    out["mismatch_intensity"] = (out["conventional_order"] - out["aee_order"]).abs()

    out["agreement_binary"] = np.where(out["mismatch_intensity"] == 0, 1, 0)
    out["agreement_class"] = np.where(
        out["agreement_binary"] == 1,
        "Agreement",
        "Mismatch",
    )

    def mismatch_category(row):
        conv = row["conventional_order"]
        aee = row["aee_order"]
        diff = row["mismatch_intensity"]

        if pd.isna(conv) or pd.isna(aee):
            return "Not available"

        if diff == 0:
            return "Agreement"

        # Conventional is more urban than functional AEE.
        if conv < aee:
            if diff >= STRONG_MISMATCH_THRESHOLD:
                return "Strong mismatch: conventional more urban than AEE"
            return "Soft mismatch: conventional more urban than AEE"

        # AEE is more urban-functional than conventional.
        if aee < conv:
            if diff >= STRONG_MISMATCH_THRESHOLD:
                return "Strong mismatch: AEE more urban-functional than conventional"
            return "Soft mismatch: AEE more urban-functional than conventional"

        return "Mismatch"

    out["mismatch_category"] = out.apply(mismatch_category, axis=1)

    def planning_interpretation(row):
        category = row["mismatch_category"]

        if category == "Agreement":
            return "Stable agreement between conventional and functional classification"
        if "conventional more urban" in category:
            return "Morphologically/density-classified as more urban, but functionally less urban by AEE profile"
        if "AEE more urban-functional" in category:
            return "Functionally more urban/accessibility-oriented than conventional morphology/density suggests"
        return "Uncertain or unavailable"

    out["mismatch_interpretation"] = out.apply(planning_interpretation, axis=1)

    return out


# ============================================================
# 8. TABLES
# ============================================================

def create_crosstab(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    tab = pd.crosstab(
        gdf["conventional_class"],
        gdf["aee_functional_class"],
        margins=True,
        dropna=False,
    )
    return tab.reset_index()


def create_mismatch_summary(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    summary = (
        gdf
        .groupby("mismatch_category")
        .agg(
            n_cells=("hex_id", "count"),
            area_km2=("geometry", area_km2),
            mean_mismatch_intensity=("mismatch_intensity", "mean"),
        )
        .reset_index()
    )

    summary["cell_share_pct"] = (summary["n_cells"] / len(gdf) * 100).round(2)
    summary["area_share_pct"] = (summary["area_km2"] / summary["area_km2"].sum() * 100).round(2)

    return summary


def create_class_profile_table(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    profile_vars = [
        "conventional_urbanity_score",
        "functional_urc_score",
        "walk_access_15m",
        "bike_access_15m",
        "pt_stop_access_20m",
        "functional_accessibility_norm",
        "green_share_pct",
        "env_quality",
        "env_burden",
        "builtup_pct",
        "road_buf_pct",
        "pop_density_km2",
        "demographic_vulnerability",
        "social_vulnerability_norm",
    ]
    profile_vars = [c for c in profile_vars if c in gdf.columns]

    profile = (
        gdf
        .groupby("mismatch_category")
        .agg(
            n_cells=("hex_id", "count"),
            area_km2=("geometry", area_km2),
            **{f"mean_{c}": (c, "mean") for c in profile_vars}
        )
        .reset_index()
    )

    return profile


def agreement_metrics(gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    valid = gdf["conventional_class"].notna() & gdf["aee_functional_class"].notna()

    conv = gdf.loc[valid, "conventional_class"].astype(str)
    aee = gdf.loc[valid, "aee_functional_class"].astype(str)

    # Kappa is meaningful here because both are reduced to four ordered/broad classes,
    # but labels are not identical, so ARI/NMI remain the primary metrics.
    metrics = {
        "n_cells": int(valid.sum()),
        "overall_agreement_share": float((gdf.loc[valid, "agreement_binary"] == 1).mean()),
        "adjusted_rand_index": adjusted_rand_score(conv, aee),
        "normalised_mutual_information": normalized_mutual_info_score(conv, aee),
        "cohen_kappa": cohen_kappa_score(conv, aee),
        "mean_mismatch_intensity": float(gdf.loc[valid, "mismatch_intensity"].mean()),
        "max_mismatch_intensity": float(gdf.loc[valid, "mismatch_intensity"].max()),
    }

    return pd.DataFrame([metrics])


# ============================================================
# 9. MAPS
# ============================================================

def mismatch_palette() -> dict[str, str]:
    return {
        "Agreement": "#2ca25f",
        "Soft mismatch: conventional more urban than AEE": "#fee08b",
        "Strong mismatch: conventional more urban than AEE": "#fdae61",
        "Soft mismatch: AEE more urban-functional than conventional": "#abd9e9",
        "Strong mismatch: AEE more urban-functional than conventional": "#2c7bb6",
        "Mismatch": "#d7191c",
        "Not available": "#d9d9d9",
    }


def conventional_palette() -> dict[str, str]:
    return {
        "Conventional urban core": "#7b3294",
        "Conventional urban": "#c2a5cf",
        "Conventional suburban/peri-urban": "#a6dba0",
        "Conventional rural/peripheral": "#008837",
        "Not available": "#d9d9d9",
    }


def aee_palette() -> dict[str, str]:
    return {
        "Functional urban core": "#7b3294",
        "Functional urban / local centre": "#c2a5cf",
        "Functional transition zone": "#fdb863",
        "Functionally vulnerable transition zone": "#e66101",
        "Functional green fringe / rural": "#5aae61",
        "Functional peripheral rural": "#1b7837",
        "Not available": "#d9d9d9",
    }


def plot_categorical_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    output_path: Path,
    palette: dict[str, str] | None = None,
):
    plot_gdf = gdf.copy()

    if palette is None:
        values = sorted(plot_gdf[column].dropna().unique())
        colors = plt.cm.tab20(np.linspace(0, 1, len(values)))
        palette = {v: colors[i] for i, v in enumerate(values)}

    plot_gdf["map_color"] = plot_gdf[column].map(palette).fillna("#d9d9d9")

    fig, ax = plt.subplots(figsize=(14, 10))
    plot_gdf.plot(
        ax=ax,
        color=plot_gdf["map_color"],
        edgecolor="white",
        linewidth=0.12,
        alpha=0.96,
    )

    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_axis_off()

    order = [k for k in palette if k in plot_gdf[column].unique()]
    patches = [
        mpatches.Patch(facecolor=palette[k], edgecolor="gray", label=k)
        for k in order
    ]

    ax.legend(
        handles=patches,
        title=column,
        loc="lower left",
        frameon=True,
        fontsize=8,
        title_fontsize=9,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Map saved: {output_path}")


def plot_numeric_map(
    gdf: gpd.GeoDataFrame,
    column: str,
    title: str,
    output_path: Path,
    cmap: str = "magma",
):
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


# ============================================================
# 10. OUTPUTS
# ============================================================

def write_excel_outputs(
    crosstab: pd.DataFrame,
    mismatch_summary: pd.DataFrame,
    profile: pd.DataFrame,
    metrics: pd.DataFrame,
):
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        metrics.to_excel(writer, sheet_name="Agreement metrics", index=False)
        crosstab.to_excel(writer, sheet_name="Cross-tabulation", index=False)
        mismatch_summary.to_excel(writer, sheet_name="Mismatch summary", index=False)
        profile.to_excel(writer, sheet_name="Mismatch profiles", index=False)

        metadata = pd.DataFrame({
            "Item": [
                "Purpose",
                "Conventional classification",
                "AEE classification",
                "Mismatch intensity",
                "Mismatch interpretation",
            ],
            "Description": [
                "Mismatch analysis between conventional urban–rural classification and GeoAI-derived AEE functional URC typology.",
                "Derived from population density, built-up intensity, and green/peripheral character unless an existing field is specified.",
                "Derived from functional URC labels/classes produced by the GeoAI typology workflow.",
                "Absolute difference between ordered conventional and AEE functional classes.",
                "Agreement means both classifications assign a comparable urban–rural level; mismatch identifies functional divergence.",
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


def save_outputs(
    gdf: gpd.GeoDataFrame,
    crosstab: pd.DataFrame,
    mismatch_summary: pd.DataFrame,
    profile: pd.DataFrame,
    metrics: pd.DataFrame,
):
    gdf.to_file(OUTPUT_GPKG, layer="conventional_vs_aee_mismatch_500m", driver="GPKG")

    shp_cols = [
        "hex_id",
        "conventional_class",
        "aee_functional_class",
        "mismatch_category",
        "mismatch_intensity",
        "agreement_binary",
        "agreement_class",
        "geometry",
    ]

    shp = gdf[[c for c in shp_cols if c in gdf.columns]].copy()
    shp = shp.rename(columns={
        "conventional_class": "conv_cls",
        "aee_functional_class": "aee_cls",
        "mismatch_category": "mis_cat",
        "mismatch_intensity": "mis_int",
        "agreement_binary": "agr_bin",
        "agreement_class": "agr_cls",
    })

    shp.to_file(
        OUTPUT_SHP_DIR / "odense_urc_conventional_vs_aee_mismatch_500m.shp",
        driver="ESRI Shapefile",
        encoding="UTF-8",
    )

    crosstab.to_csv(CROSSTAB_CSV, index=False)
    mismatch_summary.to_csv(SUMMARY_CSV, index=False)
    write_excel_outputs(crosstab, mismatch_summary, profile, metrics)

    plot_categorical_map(
        gdf,
        "mismatch_category",
        "Odense URC Mismatch: Conventional Classification vs AEE Functional Typology",
        OUTPUT_MISMATCH_MAP,
        mismatch_palette(),
    )

    plot_numeric_map(
        gdf,
        "mismatch_intensity",
        "Odense URC Mismatch Intensity",
        OUTPUT_MISMATCH_INTENSITY_MAP,
        cmap="magma",
    )

    plot_categorical_map(
        gdf,
        "agreement_class",
        "Odense URC Agreement / Mismatch Map",
        OUTPUT_AGREEMENT_MAP,
        {
            "Agreement": "#2ca25f",
            "Mismatch": "#de2d26",
        },
    )

    plot_categorical_map(
        gdf,
        "conventional_class",
        "Odense Conventional Urban–Rural Classification",
        OUTPUT_CONVENTIONAL_MAP,
        conventional_palette(),
    )

    plot_categorical_map(
        gdf,
        "aee_functional_class",
        "Odense AEE Functional URC Classification",
        OUTPUT_AEE_MAP,
        aee_palette(),
    )


def write_validation(gdf: gpd.GeoDataFrame, metrics: pd.DataFrame):
    lines = ["Odense URC mismatch validation", "=" * 45]
    lines.append(f"Input: {INPUT_GPKG}")
    lines.append(f"Cells: {len(gdf)}")
    lines.append("")
    lines.append(f"Conventional class source: {gdf['conventional_class_source'].iloc[0]}")
    lines.append(f"AEE functional class source: {gdf['aee_functional_class_source'].iloc[0]}")
    lines.append("")
    lines.append("Agreement metrics:")
    for col in metrics.columns:
        lines.append(f"  {col}: {metrics[col].iloc[0]}")
    lines.append("")
    lines.append("Mismatch category counts:")
    for cat, n in gdf["mismatch_category"].value_counts().items():
        lines.append(f"  {cat}: {n}")
    lines.append("")
    lines.append("Interpretation notes:")
    lines.append("- Agreement means conventional and AEE classifications assign comparable urban–rural status.")
    lines.append("- 'Conventional more urban than AEE' means the cell appears urban by density/morphology but less urban in functional AEE terms.")
    lines.append("- 'AEE more urban-functional than conventional' means the cell has stronger functional accessibility/urban characteristics than density/morphology alone suggests.")
    lines.append("- Strong mismatch indicates a difference of two or more ordered class levels.")
    lines.append("- This mismatch map is useful for identifying planning-relevant transition zones and areas where conventional classifications obscure functional conditions.")

    VALIDATION_TXT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Validation report saved: {VALIDATION_TXT}")


# ============================================================
# 11. MAIN
# ============================================================

def main():
    gdf = load_typology()

    gdf = derive_conventional_classification(gdf)
    gdf = derive_aee_functional_classification(gdf)
    gdf = calculate_mismatch(gdf)

    crosstab = create_crosstab(gdf)
    mismatch_summary = create_mismatch_summary(gdf)
    profile = create_class_profile_table(gdf)
    metrics = agreement_metrics(gdf)

    save_outputs(gdf, crosstab, mismatch_summary, profile, metrics)
    write_validation(gdf, metrics)

    print(f"GeoPackage saved: {OUTPUT_GPKG}")
    print(f"Excel tables saved: {OUTPUT_XLSX}")
    print(f"Cross-tabulation saved: {CROSSTAB_CSV}")
    print(f"Summary saved: {SUMMARY_CSV}")
    print("Finished.")


if __name__ == "__main__":
    main()
