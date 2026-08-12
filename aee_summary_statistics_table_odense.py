# ============================================================
# Odense AEE Summary Statistics Table
# Accessibility – Environment – Equity indicators
# Reads output GeoPackages from the previously generated scripts
# and creates CSV + Excel summary tables.
# ============================================================
#
# Expected project structure:
#
# Project/
# ├── Map Layers/
# │   ├── Odense_Municipality.gpkg
# │   └── Odense-500mHexaCells_1.gpkg
# ├── odense_osm_walking_accessibility_outputs/
# ├── odense_osm_cycling_accessibility_outputs/
# ├── odense_osm_public_transport_accessibility_outputs/
# ├── odense_osm_green_area_share_outputs/
# ├── odense_multisource_environmental_index_outputs/
# └── odense_demographic_indicator_outputs/
#
# The script is robust to missing files/layers. Missing indicators are skipped
# and listed in the validation report.
# ============================================================

from __future__ import annotations

from pathlib import Path
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "Map Layers").exists() else SCRIPT_DIR
# All AEE indicator GeoPackages are written under this folder (nested output dirs).
AEE_OUTPUTS_DIR = SCRIPT_DIR

OUTPUT_DIR = SCRIPT_DIR / "odense_aee_summary_statistics_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

SUMMARY_XLSX = OUTPUT_DIR / "odense_aee_summary_statistics.xlsx"
SUMMARY_CSV = OUTPUT_DIR / "odense_aee_indicator_summary.csv"
CLASS_CSV = OUTPUT_DIR / "odense_aee_class_distribution.csv"
CORRELATION_CSV = OUTPUT_DIR / "odense_aee_indicator_correlation_matrix.csv"
VALIDATION_TXT = OUTPUT_DIR / "odense_aee_summary_validation.txt"

TARGET_CRS = "EPSG:25832"


# ============================================================
# 1. INPUT CONFIGURATION
# ============================================================

INDICATOR_CONFIG = [
    # ---------------- Accessibility ----------------
    {
        "aee_dimension": "Accessibility",
        "indicator_id": "walk_access_15m",
        "indicator_name": "Walking accessibility to services",
        "gpkg": AEE_OUTPUTS_DIR / "odense_walking_accessibility_services_500m.gpkg",
        "layer": "walking_accessibility_500m_hexagons",
        "value_col": "walk_access_score",
        "class_col": "walk_access_class",
        "rank_col": "walk_access_rank",
        "unit": "0–100 index",
        "direction": "Higher = better accessibility",
        "method": "OSM pedestrian network, selected services, 15-min threshold, exponential decay",
    },
    {
        "aee_dimension": "Accessibility",
        "indicator_id": "bike_access_5m",
        "indicator_name": "Cycling accessibility to services, 5 min",
        "gpkg": AEE_OUTPUTS_DIR / "odense_osm_cycling_accessibility_outputs" / "odense_cycling_accessibility_services_500m.gpkg",
        "layer": "cycling_accessibility_500m_hexagons",
        "value_col": "bike_access_score_5m",
        "class_col": "bike_access_class_5m",
        "rank_col": "bike_access_rank_5m",
        "unit": "0–100 index",
        "direction": "Higher = better accessibility",
        "method": "OSM bicycle network, selected services, 5-min threshold, exponential decay",
    },
    {
        "aee_dimension": "Accessibility",
        "indicator_id": "bike_access_15m",
        "indicator_name": "Cycling accessibility to services, 15 min",
        "gpkg": AEE_OUTPUTS_DIR / "odense_osm_cycling_accessibility_outputs" / "odense_cycling_accessibility_services_500m.gpkg",
        "layer": "cycling_accessibility_500m_hexagons",
        "value_col": "bike_access_score_15m",
        "class_col": "bike_access_class_15m",
        "rank_col": "bike_access_rank_15m",
        "unit": "0–100 index",
        "direction": "Higher = better accessibility",
        "method": "OSM bicycle network, selected services, 15-min threshold, exponential decay",
    },
    {
        "aee_dimension": "Accessibility",
        "indicator_id": "pt_stop_access_10m",
        "indicator_name": "Public transport stop accessibility, 10 min",
        "gpkg": AEE_OUTPUTS_DIR / "odense_osm_public_transport_accessibility_outputs" / "odense_public_transport_stop_accessibility_500m.gpkg",
        "layer": "pt_stop_accessibility_500m_hexagons",
        "value_col": "pt_access_score_10m",
        "class_col": "pt_access_class_10m",
        "rank_col": "pt_access_rank_10m",
        "unit": "0–100 index",
        "direction": "Higher = better stop accessibility",
        "method": "OSM walking network to OSM public transport stops/stations, 10-min threshold",
    },
    {
        "aee_dimension": "Accessibility",
        "indicator_id": "pt_stop_access_20m",
        "indicator_name": "Public transport stop accessibility, 20 min",
        "gpkg": AEE_OUTPUTS_DIR / "odense_osm_public_transport_accessibility_outputs" / "odense_public_transport_stop_accessibility_500m.gpkg",
        "layer": "pt_stop_accessibility_500m_hexagons",
        "value_col": "pt_access_score_20m",
        "class_col": "pt_access_class_20m",
        "rank_col": "pt_access_rank_20m",
        "unit": "0–100 index",
        "direction": "Higher = better stop accessibility",
        "method": "OSM walking network to OSM public transport stops/stations, 20-min threshold",
    },

    # ---------------- Environment ----------------
    {
        "aee_dimension": "Environment",
        "indicator_id": "green_share",
        "indicator_name": "Green-area share",
        "gpkg": AEE_OUTPUTS_DIR / "odense_osm_green_area_share_outputs" / "odense_green_area_share_500m.gpkg",
        "layer": "green_area_share_500m_hexagons",
        "value_col": "green_share_pct",
        "class_col": "green_class",
        "rank_col": "green_rank",
        "unit": "%",
        "direction": "Higher = more green area",
        "method": "OSM green/open/natural polygon share by 500 m cell",
    },
    {
        "aee_dimension": "Environment",
        "indicator_id": "env_quality",
        "indicator_name": "Environmental quality index",
        "gpkg": AEE_OUTPUTS_DIR / "odense_multisource_environmental_index_outputs" / "odense_multisource_environmental_index_500m.gpkg",
        "layer": "environmental_index_500m_hexagons",
        "value_col": "env_quality_score",
        "class_col": "env_quality_class",
        "rank_col": "env_quality_rank",
        "unit": "0–100 index",
        "direction": "Higher = better environmental condition",
        "method": "Green, blue, built-up, road burden, optional official environmental layers",
    },
    {
        "aee_dimension": "Environment",
        "indicator_id": "env_burden",
        "indicator_name": "Environmental burden index",
        "gpkg": AEE_OUTPUTS_DIR / "odense_multisource_environmental_index_outputs" / "odense_multisource_environmental_index_500m.gpkg",
        "layer": "environmental_index_500m_hexagons",
        "value_col": "env_burden_score",
        "class_col": "env_burden_class",
        "rank_col": "env_burden_rank",
        "unit": "0–100 index",
        "direction": "Higher = higher environmental burden",
        "method": "Low green/blue, built-up share, road burden, optional air/heat/noise/ecology layers",
    },
    {
        "aee_dimension": "Environment",
        "indicator_id": "builtup_share",
        "indicator_name": "Built-up / impervious proxy share",
        "gpkg": AEE_OUTPUTS_DIR / "odense_multisource_environmental_index_outputs" / "odense_multisource_environmental_index_500m.gpkg",
        "layer": "environmental_index_500m_hexagons",
        "value_col": "builtup_pct",
        "class_col": None,
        "rank_col": None,
        "unit": "%",
        "direction": "Higher = more built-up/impervious proxy",
        "method": "OSM building and built-up land-use polygon share",
    },
    {
        "aee_dimension": "Environment",
        "indicator_id": "road_burden",
        "indicator_name": "Major-road exposure proxy",
        "gpkg": AEE_OUTPUTS_DIR / "odense_multisource_environmental_index_outputs" / "odense_multisource_environmental_index_500m.gpkg",
        "layer": "environmental_index_500m_hexagons",
        "value_col": "road_buf_pct",
        "class_col": None,
        "rank_col": None,
        "unit": "%",
        "direction": "Higher = stronger major-road buffer exposure",
        "method": "Share of cell intersecting 100 m buffer around OSM major roads",
    },

    # ---------------- Equity / Demography ----------------
    {
        "aee_dimension": "Equity",
        "indicator_id": "population_density",
        "indicator_name": "Population density",
        "gpkg": AEE_OUTPUTS_DIR / "odense_demographic_indicator_outputs" / "odense_demographic_indicators_500m.gpkg",
        "layer": "demographic_indicators_500m_hexagons",
        "value_col": "pop_density_km2",
        "class_col": "pop_density_class",
        "rank_col": "pop_density_rank",
        "unit": "persons/km²",
        "direction": "Higher = denser population",
        "method": "Population aggregated to 500 m cells; density calculated by clipped cell area",
    },
    {
        "aee_dimension": "Equity",
        "indicator_id": "age_vulnerability",
        "indicator_name": "Age vulnerability",
        "gpkg": AEE_OUTPUTS_DIR / "odense_demographic_indicator_outputs" / "odense_demographic_indicators_500m.gpkg",
        "layer": "demographic_indicators_500m_hexagons",
        "value_col": "age_vulnerability_share",
        "class_col": "age_vulnerability_class",
        "rank_col": "age_vulnerability_rank",
        "unit": "share",
        "direction": "Higher = higher age vulnerability",
        "method": "Children share + elderly share",
    },
    {
        "aee_dimension": "Equity",
        "indicator_id": "socioeconomic_vulnerability",
        "indicator_name": "Socio-economic vulnerability",
        "gpkg": AEE_OUTPUTS_DIR / "odense_demographic_indicator_outputs" / "odense_demographic_indicators_500m.gpkg",
        "layer": "demographic_indicators_500m_hexagons",
        "value_col": "socioeconomic_vulnerability_share",
        "class_col": "socioeconomic_vulnerability_class",
        "rank_col": "socioeconomic_vulnerability_rank",
        "unit": "share",
        "direction": "Higher = higher socio-economic vulnerability",
        "method": "Available low education, unemployment, and low-income indicators",
    },
    {
        "aee_dimension": "Equity",
        "indicator_id": "demographic_vulnerability",
        "indicator_name": "Composite demographic vulnerability",
        "gpkg": AEE_OUTPUTS_DIR / "odense_demographic_indicator_outputs" / "odense_demographic_indicators_500m.gpkg",
        "layer": "demographic_indicators_500m_hexagons",
        "value_col": "demo_vulnerability_score",
        "class_col": "demo_vulnerability_class",
        "rank_col": "demo_vulnerability_rank",
        "unit": "0–100 index",
        "direction": "Higher = higher demographic vulnerability",
        "method": "Population density, age vulnerability, socio-economic vulnerability, optional origin indicators",
    },
]


# ============================================================
# 2. HELPERS
# ============================================================

def safe_read_layer(gpkg: Path, layer: str) -> gpd.GeoDataFrame | None:
    if not gpkg.exists():
        return None
    try:
        return gpd.read_file(gpkg, layer=layer).to_crs(TARGET_CRS)
    except Exception as e:
        print(f"Could not read {gpkg.name} / {layer}: {e}")
        return None


def get_cell_area(gdf: gpd.GeoDataFrame) -> pd.Series:
    if "cell_area_m2" in gdf.columns:
        return pd.to_numeric(gdf["cell_area_m2"], errors="coerce")
    return gdf.geometry.area


def describe_indicator(
    gdf: gpd.GeoDataFrame,
    config: dict,
) -> dict:
    value_col = config["value_col"]

    if value_col not in gdf.columns:
        raise KeyError(f"Missing value column: {value_col}")

    values = pd.to_numeric(gdf[value_col], errors="coerce")
    area = get_cell_area(gdf)

    valid = values.notna()
    valid_values = values[valid]
    valid_area = area[valid]

    if valid_values.empty:
        raise ValueError(f"No valid numeric values in {value_col}")

    area_weighted_mean = np.nan
    if valid_area.notna().any() and valid_area.sum() > 0:
        area_weighted_mean = np.average(valid_values, weights=valid_area)

    pop_weighted_mean = np.nan
    if "population_total" in gdf.columns:
        pop = pd.to_numeric(gdf["population_total"], errors="coerce").fillna(0)
        if pop[valid].sum() > 0:
            pop_weighted_mean = np.average(valid_values, weights=pop[valid])

    return {
        "AEE dimension": config["aee_dimension"],
        "Indicator ID": config["indicator_id"],
        "Indicator name": config["indicator_name"],
        "Unit": config["unit"],
        "Direction": config["direction"],
        "Method summary": config["method"],
        "Source GeoPackage": str(config["gpkg"].relative_to(PROJECT_DIR)) if config["gpkg"].is_relative_to(PROJECT_DIR) else str(config["gpkg"]),
        "Layer": config["layer"],
        "Value field": value_col,
        "Class field": config["class_col"] or "",
        "Number of cells": int(len(gdf)),
        "Valid cells": int(valid.sum()),
        "Missing cells": int((~valid).sum()),
        "Mean": round(float(valid_values.mean()), 4),
        "Area-weighted mean": round(float(area_weighted_mean), 4) if not np.isnan(area_weighted_mean) else np.nan,
        "Population-weighted mean": round(float(pop_weighted_mean), 4) if not np.isnan(pop_weighted_mean) else np.nan,
        "Median": round(float(valid_values.median()), 4),
        "Std. dev.": round(float(valid_values.std()), 4),
        "Min": round(float(valid_values.min()), 4),
        "Q25": round(float(valid_values.quantile(0.25)), 4),
        "Q75": round(float(valid_values.quantile(0.75)), 4),
        "Max": round(float(valid_values.max()), 4),
    }


def class_distribution(
    gdf: gpd.GeoDataFrame,
    config: dict,
) -> pd.DataFrame:
    class_col = config.get("class_col")
    if not class_col or class_col not in gdf.columns:
        return pd.DataFrame()

    counts = (
        gdf[class_col]
        .fillna("Missing")
        .value_counts(dropna=False)
        .rename_axis("Class")
        .reset_index(name="Cell count")
    )
    counts["Cell share (%)"] = (counts["Cell count"] / len(gdf) * 100).round(2)
    counts.insert(0, "AEE dimension", config["aee_dimension"])
    counts.insert(1, "Indicator ID", config["indicator_id"])
    counts.insert(2, "Indicator name", config["indicator_name"])
    counts.insert(3, "Class field", class_col)

    return counts


def assign_cell_key(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Stable cross-indicator key: indicator GeoPackages use different hex_id prefixes."""
    out = gdf.copy()
    centroid = out.geometry.centroid
    out["cell_key"] = centroid.x.round(2).astype(str) + "_" + centroid.y.round(2).astype(str)
    return out


def merge_indicator_values_for_correlation(
    base: pd.DataFrame | None,
    gdf: gpd.GeoDataFrame,
    config: dict,
) -> pd.DataFrame | None:
    value_col = config["value_col"]
    indicator_id = config["indicator_id"]

    if value_col not in gdf.columns:
        return base

    keyed = assign_cell_key(gdf)
    tmp = keyed[["cell_key", value_col]].copy().rename(columns={value_col: indicator_id})
    tmp[indicator_id] = pd.to_numeric(tmp[indicator_id], errors="coerce")

    if base is None:
        return tmp

    return base.merge(tmp, on="cell_key", how="inner")


# ============================================================
# 3. MAIN
# ============================================================

def main():
    summary_rows = []
    class_tables = []
    validation_lines = ["Odense AEE indicator summary validation", "=" * 50]
    correlation_base = None

    for cfg in INDICATOR_CONFIG:
        gdf = safe_read_layer(cfg["gpkg"], cfg["layer"])

        if gdf is None:
            validation_lines.append(f"SKIPPED missing file/layer: {cfg['indicator_id']} -> {cfg['gpkg']} / {cfg['layer']}")
            continue

        if cfg["value_col"] not in gdf.columns:
            validation_lines.append(f"SKIPPED missing value field: {cfg['indicator_id']} -> {cfg['value_col']}")
            continue

        try:
            row = describe_indicator(gdf, cfg)
            summary_rows.append(row)
            validation_lines.append(f"OK: {cfg['indicator_id']} ({cfg['value_col']})")

            ctab = class_distribution(gdf, cfg)
            if not ctab.empty:
                class_tables.append(ctab)

            correlation_base = merge_indicator_values_for_correlation(correlation_base, gdf, cfg)

        except Exception as e:
            validation_lines.append(f"ERROR: {cfg['indicator_id']} -> {e}")

    summary_df = pd.DataFrame(summary_rows)

    if summary_df.empty:
        raise RuntimeError(
            "No indicator summaries were generated. Check that output GeoPackages exist and that field names match."
        )

    class_df = pd.concat(class_tables, ignore_index=True) if class_tables else pd.DataFrame()

    if correlation_base is not None:
        value_cols = [c for c in correlation_base.columns if c != "cell_key"]
        correlation_df = correlation_base[value_cols].apply(pd.to_numeric, errors="coerce").corr().round(3)
    else:
        correlation_df = pd.DataFrame()

    # Dimension-level summary
    dimension_summary = (
        summary_df
        .groupby("AEE dimension")
        .agg(
            indicators=("Indicator ID", "count"),
            mean_of_means=("Mean", "mean"),
            mean_valid_cells=("Valid cells", "mean"),
            total_missing_cells=("Missing cells", "sum"),
        )
        .reset_index()
    )

    # Export CSVs
    summary_df.to_csv(SUMMARY_CSV, index=False)
    if not class_df.empty:
        class_df.to_csv(CLASS_CSV, index=False)
    if not correlation_df.empty:
        correlation_df.to_csv(CORRELATION_CSV)

    # Export Excel workbook
    with pd.ExcelWriter(SUMMARY_XLSX, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Indicator summary", index=False)
        dimension_summary.to_excel(writer, sheet_name="AEE dimension summary", index=False)
        if not class_df.empty:
            class_df.to_excel(writer, sheet_name="Class distribution", index=False)
        if not correlation_df.empty:
            correlation_df.to_excel(writer, sheet_name="Correlation matrix")

        # Method/metadata sheet
        meta = pd.DataFrame({
            "Item": [
                "Purpose",
                "Spatial unit",
                "CRS",
                "Accessibility indicators",
                "Environment indicators",
                "Equity indicators",
                "Interpretation caution",
            ],
            "Description": [
                "Summary statistics for Accessibility–Environment–Equity indicators in Odense.",
                "500 m hexagonal cells clipped to Odense Municipality.",
                TARGET_CRS,
                "Walking, cycling, and public transport stop accessibility indices.",
                "Green-area share, environmental quality/burden, built-up share, road exposure.",
                "Population density, age vulnerability, socio-economic vulnerability, demographic vulnerability.",
                "Statistics are based on the generated model outputs and depend on OSM completeness and the availability/quality of official demographic/environmental inputs.",
            ]
        })
        meta.to_excel(writer, sheet_name="Metadata", index=False)

        # Basic formatting with openpyxl
        wb = writer.book
        for ws in wb.worksheets:
            ws.freeze_panes = "A2"
            for col_cells in ws.columns:
                max_len = 0
                column = col_cells[0].column_letter
                for cell in col_cells:
                    try:
                        max_len = max(max_len, len(str(cell.value)) if cell.value is not None else 0)
                    except Exception:
                        pass
                ws.column_dimensions[column].width = min(max(max_len + 2, 12), 45)

    validation_lines.append("")
    validation_lines.append(f"Summary rows generated: {len(summary_df)}")
    validation_lines.append(f"Class-distribution rows generated: {len(class_df)}")
    validation_lines.append(f"Excel output: {SUMMARY_XLSX}")
    validation_lines.append(f"CSV output: {SUMMARY_CSV}")

    VALIDATION_TXT.write_text("\n".join(validation_lines), encoding="utf-8")

    print(f"Created: {SUMMARY_XLSX}")
    print(f"Created: {SUMMARY_CSV}")
    print(f"Created: {CLASS_CSV}")
    print(f"Created: {CORRELATION_CSV}")
    print(f"Created: {VALIDATION_TXT}")


if __name__ == "__main__":
    main()
