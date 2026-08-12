# ============================================================
# Odense Demographic Indicator Map
# 500 m hexagonal cells | Statistics Denmark / demographic grid compatible
# CRS: EPSG:25832
# ============================================================
#
# PURPOSE
# -------
# This script generates demographic indicators for the same 500 m Odense
# hexagonal grid used in the accessibility and environmental analyses.
#
# Main outputs:
#   1. Population density map
#   2. Age vulnerability map, if age columns are available
#   3. Socio-economic vulnerability map, if socio-economic columns are available
#   4. Composite demographic vulnerability index
#
# DATA INPUT LOGIC
# ----------------
# Put demographic files in:
#   Map Layers/Demographic/
#
# Supported formats:
#   - GeoPackage (.gpkg)
#   - Shapefile (.shp)
#   - CSV with coordinates
#
# Recommended source:
#   - Statistics Denmark Kvadratnet / Danish National Grid population data
#   - Other official demographic grid, point, or polygon layers
#
# If the demographic layer is polygonal, the script uses areal interpolation:
#   value transferred to hex = source_value * intersection_area / source_area
#
# If the demographic layer is point-based, the script spatially joins points
# to hexagons and sums demographic variables.
#
# IMPORTANT
# ---------
# Column names in official demographic data may differ. Configure the candidate
# lists below if your files use Danish variable names or table-specific codes.
# ============================================================

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from shapely.geometry import Point

warnings.filterwarnings("ignore")


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "Map Layers").exists() else SCRIPT_DIR
MAP_LAYERS = PROJECT_DIR / "Map Layers"
DEMOGRAPHIC_DIR = MAP_LAYERS / "Demographic"

MUNICIPALITY_CANDIDATES = [
    MAP_LAYERS / "Odense_Municipality.gpkg",
    MAP_LAYERS / "Odense_Municipality_1.gpkg",
    MAP_LAYERS / "Odense_Municipality.shp",
]

HEXGRID_CANDIDATES = [
    MAP_LAYERS / "Odense-500mHexaCells_1.gpkg",
    MAP_LAYERS / "Odense-500mHexaCells.gpkg",
    MAP_LAYERS / "odense_aee_500m_hexgrid.gpkg",
]

DEMOGRAPHIC_CANDIDATES = [
    # v2 preferred: StatBank parish × BBR bolig dasymetric (+ DST class soft-constraint)
    DEMOGRAPHIC_DIR / "dasymetric_population_hex500_v2.gpkg",
    DEMOGRAPHIC_DIR / "dasymetric_population_1km_v2.gpkg",
    # v1: density-class–derived 1 km grid scaled to KOMM_TOT 2021
    DEMOGRAPHIC_DIR / "dst_kvadratnet_population.gpkg",
    DEMOGRAPHIC_DIR / "dst_kvadratnet_population.shp",
    DEMOGRAPHIC_DIR / "odense_basisdata_o" / "odense_1km_population_density_classes.gpkg",
    # legacy / invalid building-proxy grid — keep last so it is not auto-selected
    DEMOGRAPHIC_DIR / "population_grid.gpkg",
    DEMOGRAPHIC_DIR / "population_grid.shp",
    DEMOGRAPHIC_DIR / "population_points.gpkg",
    DEMOGRAPHIC_DIR / "population_points.shp",
    DEMOGRAPHIC_DIR / "demographic_data.csv",
    DEMOGRAPHIC_DIR / "population_data.csv",
    DEMOGRAPHIC_DIR / "demographic_grid.gpkg",
    DEMOGRAPHIC_DIR / "demographic_grid.shp",
]

municipality_layer = "odense_municipality_districts"
municipality_layer_shp = "Odense_Municipality"

# If your hex layer name differs, the script tries common names first and
# then falls back to the first layer in the GeoPackage.
HEX_LAYER_CANDIDATES = [
    "odense500mhexacells__grid",
    "odense_aee_500m_hexgrid",
    "odense500mhexacells_clipped",
]

target_crs = "EPSG:25832"
web_mercator = "EPSG:3857"

output_dir = SCRIPT_DIR / "odense_demographic_indicator_outputs"
output_dir.mkdir(exist_ok=True)

output_gpkg = output_dir / "odense_demographic_indicators_500m.gpkg"
output_shp_dir = output_dir / "odense_demographic_indicators_500m_shp"
output_shp_dir.mkdir(exist_ok=True)

output_population_map = output_dir / "odense_population_density_500m_map.png"
output_age_map = output_dir / "odense_age_vulnerability_500m_map.png"
output_socio_map = output_dir / "odense_socioeconomic_vulnerability_500m_map.png"
output_vulnerability_map = output_dir / "odense_demographic_vulnerability_index_500m_map.png"
output_vulnerability_map_osm = output_dir / "odense_demographic_vulnerability_index_500m_map_osm.png"
validation_report = output_dir / "demographic_indicator_validation.txt"

# CSV coordinate configuration.
# If your CSV has EPSG:25832 coordinates, use x/y columns.
# If your CSV has WGS84 coordinates, use lon/lat columns.
CSV_X_CANDIDATES = ["x", "X", "easting", "Easting", "utm_x", "ETRS89_X"]
CSV_Y_CANDIDATES = ["y", "Y", "northing", "Northing", "utm_y", "ETRS89_Y"]
CSV_LON_CANDIDATES = ["lon", "longitude", "Longitude", "lng", "xcoord"]
CSV_LAT_CANDIDATES = ["lat", "latitude", "Latitude", "ycoord"]


# ============================================================
# 2. COLUMN DETECTION
# ============================================================
# Add your exact Statistics Denmark / project column names here if needed.

COLUMN_CANDIDATES = {
    # Extensive count variables
    "population_total": [
        "pop_total", "population", "POP", "pop", "persons", "PERSONS",
        "antal", "ANTAL", "indbyggere", "IND", "befolkning", "BEFOLK",
        "FOLK1", "total_population", "TOTAL",
    ],
    "children": [
        "age_0_14", "pop_0_14", "children", "CHILDREN", "born", "boern",
        "0_14", "A0_14", "age0_14", "u15", "under15",
    ],
    "elderly": [
        "age_65plus", "pop_65plus", "elderly", "ELDERLY", "65plus",
        "65_plus", "A65PLUS", "age65plus", "over65", "65_",
    ],

    # Socio-economic count or share variables
    # If these are counts, the script converts to shares using population_total.
    # If these are already shares/percentages, the script detects values <= 1 or <= 100.
    "low_education": [
        "low_education", "lowedu", "LOWEDU", "grundskole", "basic_education",
        "ISCED_low", "edu_low", "lav_uddannelse",
    ],
    "unemployment": [
        "unemployment", "unemployed", "UNEMP", "ledig", "ledige", "arbejdsloes",
        "unemp_count", "unemp_share",
    ],
    "low_income": [
        "low_income", "LOWINC", "income_low", "lav_indkomst",
        "poverty", "at_risk_poverty",
    ],
    # Migration-origin fields intentionally NOT mapped:
    # nonwestern / ikkevestlig / turkey represent migrant subsets, not total population.
}


# Composite vulnerability weights (total-population track only).
# Migration-origin weights removed. Only available non-constant indicators are used.
VULNERABILITY_WEIGHT_CANDIDATES = {
    "pop_density_norm": 0.40,
    "age_vulnerability_norm": 0.30,
    "socioeconomic_vulnerability_norm": 0.30,
}

# Never ingest these columns even if present in a source GPKG
EXCLUDED_MIGRATION_FIELDS = {
    "nonwestern",
    "non_western",
    "nonwestern_origin",
    "ikkevestlig",
    "ikke_vestlig",
    "ikkevestlig_pop",
    "turkey_origin_pop",
    "turkey_origin",
}


# ============================================================
# 3. BASIC HELPERS
# ============================================================

def find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def first_existing_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    cols_lower = {c.lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
        if candidate.lower() in cols_lower:
            return cols_lower[candidate.lower()]
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


def classify_quintiles(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Always produce 5 ordered classes via rank-based quintiles (ties broken)."""
    s = pd.to_numeric(series, errors="coerce")
    s = s.fillna(s.mean()) if s.notna().any() else s.fillna(0.0)

    labels = ["Very low", "Low", "Moderate", "High", "Very high"]
    try:
        # rank(method="first") guarantees 5 bins even with many tied zeros
        bins = pd.qcut(s.rank(method="first"), q=5, labels=False, duplicates="drop")
    except ValueError:
        bins = pd.Series(0, index=s.index)

    uniq = sorted(pd.Series(bins).dropna().unique())
    # Map bin ids onto the full 5-label scale from low→high
    if len(uniq) == 5:
        class_map = {u: labels[i] for i, u in enumerate(uniq)}
        rank_map = {u: i + 1 for i, u in enumerate(uniq)}
    else:
        # Fallback: stretch whatever bins exist across the 5 labels evenly
        class_map = {
            u: labels[int(round(i * (len(labels) - 1) / max(len(uniq) - 1, 1)))]
            for i, u in enumerate(uniq)
        }
        rank_map = {u: labels.index(class_map[u]) + 1 for u in uniq}

    rank = pd.Series(bins, index=s.index).map(rank_map).fillna(1).astype(int)
    cls = pd.Series(bins, index=s.index).map(class_map).fillna("Not available")
    return rank, cls


def weighted_score(df: pd.DataFrame, candidate_weights: dict[str, float], score_name: str):
    available = {}
    for col, weight in candidate_weights.items():
        if col in df.columns and df[col].notna().any():
            s = pd.to_numeric(df[col], errors="coerce")
            if not np.isclose(s.std(skipna=True), 0):
                available[col] = weight

    if not available:
        raise ValueError(f"No valid indicators available for {score_name}")

    total = sum(available.values())
    weights = {col: w / total for col, w in available.items()}

    score = sum(df[col].fillna(df[col].mean()) * weight for col, weight in weights.items())
    return score, weights


# ============================================================
# 4. INPUT LOADING
# ============================================================

def resolve_municipality_path() -> tuple[Path, str | None]:
    path = find_first_existing(MUNICIPALITY_CANDIDATES)
    if path is None:
        raise FileNotFoundError("Municipality boundary not found in Map Layers.")
    layer = municipality_layer_shp if path.suffix.lower() == ".shp" else municipality_layer
    return path, layer


def resolve_hexgrid_path() -> Path:
    path = find_first_existing(HEXGRID_CANDIDATES)
    if path is None:
        raise FileNotFoundError("Hex grid not found in Map Layers.")
    return path


def read_hexgrid(path: Path) -> gpd.GeoDataFrame:
    if path.suffix.lower() == ".gpkg":
        import pyogrio
        layers = [row[0] for row in pyogrio.list_layers(path)]
        for lyr in HEX_LAYER_CANDIDATES:
            if lyr in layers:
                return gpd.read_file(path, layer=lyr)
        return gpd.read_file(path, layer=layers[0])
    return gpd.read_file(path)


def load_boundary_and_hexagons():
    muni_path, muni_layer = resolve_municipality_path()
    hex_path = resolve_hexgrid_path()

    municipality = gpd.read_file(muni_path, layer=muni_layer) if muni_layer else gpd.read_file(muni_path)
    hexagons = read_hexgrid(hex_path)

    municipality = municipality.to_crs(target_crs)
    hexagons = hexagons.to_crs(target_crs)

    municipality_boundary = gpd.GeoDataFrame(
        geometry=[municipality.geometry.union_all()],
        crs=target_crs,
    )

    hexagons = gpd.clip(hexagons, municipality_boundary).reset_index(drop=True)
    hexagons["hex_id"] = [f"OD_DEMO_{i + 1:04d}" for i in range(len(hexagons))]
    hexagons["cell_area_m2"] = hexagons.geometry.area
    hexagons["cell_area_km2"] = hexagons["cell_area_m2"] / 1_000_000

    print(f"Loaded {len(hexagons)} clipped 500 m hexagons.")
    print(f"Municipality area: {municipality_boundary.geometry.area.iloc[0] / 1_000_000:.2f} km²")

    return municipality, municipality_boundary, hexagons


def load_demographic_data() -> gpd.GeoDataFrame:
    path = find_first_existing(DEMOGRAPHIC_CANDIDATES)
    if path is None:
        raise FileNotFoundError(
            "No demographic input file found. Put a demographic grid/point file in Map Layers/Demographic/."
        )

    print(f"Loading demographic data: {path}")

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)

        x_col = first_existing_column(df, CSV_X_CANDIDATES)
        y_col = first_existing_column(df, CSV_Y_CANDIDATES)
        lon_col = first_existing_column(df, CSV_LON_CANDIDATES)
        lat_col = first_existing_column(df, CSV_LAT_CANDIDATES)

        if x_col and y_col:
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df[x_col], df[y_col]),
                crs=target_crs,
            )
        elif lon_col and lat_col:
            gdf = gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
                crs="EPSG:4326",
            ).to_crs(target_crs)
        else:
            raise ValueError(
                "CSV must contain either EPSG:25832 x/y columns or lon/lat columns."
            )
    else:
        if path.suffix.lower() == ".gpkg":
            import pyogrio
            layers = [row[0] for row in pyogrio.list_layers(path)]
            gdf = gpd.read_file(path, layer=layers[0])
        else:
            gdf = gpd.read_file(path)

        gdf = gdf.to_crs(target_crs)

    gdf = gdf[gdf.geometry.notnull()].copy()
    print(f"Demographic features loaded: {len(gdf)}")
    print("Available columns:")
    print(list(gdf.columns))
    return gdf


# ============================================================
# 5. VARIABLE PREPARATION
# ============================================================

def standardise_demographic_columns(demo: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, str]]:
    demo = demo.copy()
    # Drop migration-origin fields so they cannot enter detection/aggregation
    drop_cols = [
        c
        for c in demo.columns
        if c.lower() in {x.lower() for x in EXCLUDED_MIGRATION_FIELDS}
        or any(k in c.lower() for k in ("nonwestern", "ikkevest", "turkey_origin"))
    ]
    if drop_cols:
        print(f"Excluding migration-origin columns: {drop_cols}")
        demo = demo.drop(columns=drop_cols, errors="ignore")

    detected = {}

    for standard_name, candidates in COLUMN_CANDIDATES.items():
        col = first_existing_column(demo, candidates)
        if col is not None:
            detected[standard_name] = col
            demo[standard_name] = pd.to_numeric(demo[col], errors="coerce").fillna(0.0)
        else:
            demo[standard_name] = np.nan

    if detected.get("population_total") is None:
        raise ValueError(
            "Population column was not detected. Add the correct column name to COLUMN_CANDIDATES['population_total']."
        )

    print("Detected demographic columns:")
    for k, v in detected.items():
        print(f"  {k}: {v}")

    return demo, detected


def is_polygon_layer(gdf: gpd.GeoDataFrame) -> bool:
    geom_types = set(gdf.geometry.geom_type.unique())
    return bool(geom_types.intersection({"Polygon", "MultiPolygon"}))


# ============================================================
# 6. AGGREGATION TO HEXAGONS
# ============================================================

def aggregate_polygon_demographics_to_hex(
    hexagons: gpd.GeoDataFrame,
    demo: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    print("Aggregating polygon demographic data to hexagons using areal interpolation...")

    extensive_vars = [
        "population_total",
        "children",
        "elderly",
        "low_education",
        "unemployment",
        "low_income",
    ]

    src = demo[extensive_vars + ["geometry"]].copy()
    src["source_area_m2"] = src.geometry.area
    src = src[src["source_area_m2"] > 0].copy()

    intersections = gpd.overlay(
        hexagons[["hex_id", "geometry"]],
        src,
        how="intersection",
        keep_geom_type=True,
    )

    if intersections.empty:
        raise ValueError("No overlap between demographic polygons and Odense hexagons.")

    intersections["intersect_area_m2"] = intersections.geometry.area
    intersections["area_weight"] = intersections["intersect_area_m2"] / intersections["source_area_m2"]

    for var in extensive_vars:
        intersections[f"{var}_weighted"] = intersections[var].fillna(0.0) * intersections["area_weight"]

    agg = (
        intersections
        .groupby("hex_id", as_index=False)
        .agg(**{var: (f"{var}_weighted", "sum") for var in extensive_vars})
    )

    out = hexagons.merge(agg, on="hex_id", how="left")
    for var in extensive_vars:
        out[var] = out[var].fillna(0.0)

    return out


def aggregate_point_demographics_to_hex(
    hexagons: gpd.GeoDataFrame,
    demo: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    print("Aggregating point demographic data to hexagons using spatial join...")

    extensive_vars = [
        "population_total",
        "children",
        "elderly",
        "low_education",
        "unemployment",
        "low_income",
    ]

    joined = gpd.sjoin(
        demo[extensive_vars + ["geometry"]],
        hexagons[["hex_id", "geometry"]],
        how="inner",
        predicate="within",
    )

    agg = (
        joined
        .groupby("hex_id", as_index=False)
        .agg(**{var: (var, "sum") for var in extensive_vars})
    )

    out = hexagons.merge(agg, on="hex_id", how="left")
    for var in extensive_vars:
        out[var] = out[var].fillna(0.0)

    return out


# ============================================================
# 7. INDICATOR CALCULATION
# ============================================================

def count_or_share_to_share(value_series: pd.Series, population_series: pd.Series) -> pd.Series:
    """
    Converts a socio-economic variable to share.
    - If max <= 1, it is assumed to be already a 0–1 share.
    - If max <= 100 and mean > 1, it is assumed to be percentage.
    - Otherwise, it is treated as a count and divided by total population.
    """
    v = pd.to_numeric(value_series, errors="coerce").fillna(0.0)
    pop = pd.to_numeric(population_series, errors="coerce").replace(0, np.nan)

    if v.max() <= 1.0:
        return v.clip(lower=0, upper=1)
    if v.max() <= 100.0 and v.mean() > 1.0:
        return (v / 100.0).clip(lower=0, upper=1)

    return (v / pop).fillna(0.0).clip(lower=0, upper=1)


def compute_demographic_indicators(hexagons: gpd.GeoDataFrame) -> tuple[gpd.GeoDataFrame, dict[str, float]]:
    out = hexagons.copy()

    out["population_total"] = out["population_total"].fillna(0.0)
    out["pop_density_km2"] = out["population_total"] / out["cell_area_km2"]

    # Age vulnerability
    out["children_share"] = count_or_share_to_share(out["children"], out["population_total"])
    out["elderly_share"] = count_or_share_to_share(out["elderly"], out["population_total"])
    out["age_vulnerability_share"] = (out["children_share"] + out["elderly_share"]).clip(0, 1)

    # Socio-economic vulnerability
    out["low_education_share"] = count_or_share_to_share(out["low_education"], out["population_total"])
    out["unemployment_share"] = count_or_share_to_share(out["unemployment"], out["population_total"])
    out["low_income_share"] = count_or_share_to_share(out["low_income"], out["population_total"])
    # Migration-origin shares intentionally not computed (total-population track only)

    socio_components = []
    for col in ["low_education_share", "unemployment_share", "low_income_share"]:
        if out[col].notna().any() and not np.isclose(out[col].std(skipna=True), 0):
            socio_components.append(col)

    if socio_components:
        out["socioeconomic_vulnerability_share"] = out[socio_components].mean(axis=1)
    else:
        out["socioeconomic_vulnerability_share"] = np.nan

    # Normalised indicators
    out["pop_density_norm"] = minmax(out["pop_density_km2"])
    out["age_vulnerability_norm"] = minmax(out["age_vulnerability_share"])

    if out["socioeconomic_vulnerability_share"].notna().any():
        out["socioeconomic_vulnerability_norm"] = minmax(out["socioeconomic_vulnerability_share"])
    else:
        out["socioeconomic_vulnerability_norm"] = np.nan

    # Composite demographic vulnerability index
    out["demo_vulnerability_raw"], weights_used = weighted_score(
        out,
        VULNERABILITY_WEIGHT_CANDIDATES,
        "demographic vulnerability",
    )

    out["demo_vulnerability_score"] = (minmax(out["demo_vulnerability_raw"]) * 100).round(2)
    out["demo_vulnerability_rank"], out["demo_vulnerability_class"] = classify_quintiles(
        out["demo_vulnerability_score"]
    )

    # Map classes for individual indicators
    out["pop_density_rank"], out["pop_density_class"] = classify_quintiles(out["pop_density_km2"])
    out["age_vulnerability_rank"], out["age_vulnerability_class"] = classify_quintiles(
        out["age_vulnerability_share"]
    )

    if out["socioeconomic_vulnerability_share"].notna().any():
        out["socioeconomic_vulnerability_rank"], out["socioeconomic_vulnerability_class"] = classify_quintiles(
            out["socioeconomic_vulnerability_share"].fillna(0.0)
        )
    else:
        out["socioeconomic_vulnerability_rank"] = 0
        out["socioeconomic_vulnerability_class"] = "Not available"

    return out, weights_used


# ============================================================
# 8. MAPPING
# ============================================================

def sequential_colors() -> dict[str, str]:
    return {
        "Very low": "#ffffcc",
        "Low": "#c2e699",
        "Moderate": "#78c679",
        "High": "#31a354",
        "Very high": "#006837",
    }


def vulnerability_colors() -> dict[str, str]:
    return {
        "Very low": "#ffffcc",
        "Low": "#fed976",
        "Moderate": "#fd8d3c",
        "High": "#f03b20",
        "Very high": "#bd0026",
        "Not available": "#d9d9d9",
    }


def plot_class_map(
    hex_output: gpd.GeoDataFrame,
    municipality: gpd.GeoDataFrame,
    municipality_boundary: gpd.GeoDataFrame,
    class_col: str,
    title: str,
    legend_title: str,
    colors: dict[str, str],
    output_path: Path,
):
    from map_style_utils import plot_categorical_hex_map

    plot_categorical_hex_map(
        hexes=hex_output,
        municipality=municipality,
        class_col=class_col,
        colors=colors,
        title=title,
        legend_title=legend_title,
        output_path=output_path,
    )


def plot_osm_basemap_vulnerability_map(
    hex_output: gpd.GeoDataFrame,
    municipality: gpd.GeoDataFrame,
    municipality_boundary: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    # Same styled OSM basemap map (title without OSM secondary line)
    plot_class_map(
        hex_output,
        municipality,
        municipality_boundary,
        "demo_vulnerability_class",
        "Demographic vulnerability index",
        "Demographic vulnerability",
        vulnerability_colors(),
        output_path,
    )


# ============================================================
# 9. OUTPUTS
# ============================================================

def create_validation_report(
    hex_output: gpd.GeoDataFrame,
    detected_columns: dict[str, str],
    weights_used: dict[str, float],
) -> list[str]:
    lines = ["Odense demographic indicator validation", "=" * 45]
    lines.append(f"Number of 500 m cells: {len(hex_output)}")
    lines.append("")
    lines.append("Detected input columns:")
    if detected_columns:
        for k, v in detected_columns.items():
            lines.append(f"  {k}: {v}")
    else:
        lines.append("  none")
    lines.append("")
    lines.append(f"Total population after aggregation: {hex_output['population_total'].sum():.0f}")
    lines.append(
        f"Population density range: {hex_output['pop_density_km2'].min():.2f} – "
        f"{hex_output['pop_density_km2'].max():.2f} persons/km²"
    )
    lines.append(
        f"Age vulnerability share range: {hex_output['age_vulnerability_share'].min():.3f} – "
        f"{hex_output['age_vulnerability_share'].max():.3f}"
    )
    if hex_output["socioeconomic_vulnerability_share"].notna().any():
        lines.append(
            f"Socio-economic vulnerability share range: "
            f"{hex_output['socioeconomic_vulnerability_share'].min():.3f} – "
            f"{hex_output['socioeconomic_vulnerability_share'].max():.3f}"
        )
    lines.append("")
    lines.append("Composite vulnerability weights used:")
    for k, v in weights_used.items():
        lines.append(f"  {k}: {v:.3f}")
    lines.append("")
    lines.append("Class distribution:")
    for cls, n in hex_output["demo_vulnerability_class"].value_counts().items():
        sub = hex_output[hex_output["demo_vulnerability_class"] == cls]
        lines.append(f"  {cls}: {n} cells; mean score {sub['demo_vulnerability_score'].mean():.1f}")
    lines.append("")
    lines.append("Method notes:")
    lines.append("- Grid: 500 m hexagons clipped to Odense Municipality; CRS EPSG:25832")
    lines.append("- Polygon demographic layers are aggregated by areal interpolation")
    lines.append("- Point demographic layers are aggregated by spatial join")
    lines.append("- Population density = aggregated population / clipped cell area")
    lines.append("- Age vulnerability = children share + elderly share, capped at 1")
    lines.append("- Socio-economic vulnerability uses available low education, unemployment, and low income indicators")
    lines.append("- Migration-origin fields (nonwestern / ikkevestlig / turkey) are excluded; total population only")
    lines.append("- Composite weights are re-normalised based on available non-constant indicators")
    return lines


def save_outputs(
    hex_output: gpd.GeoDataFrame,
    municipality: gpd.GeoDataFrame,
    demographic_source: gpd.GeoDataFrame,
    detected_columns: dict[str, str],
    weights_used: dict[str, float],
):
    hex_output.to_file(output_gpkg, layer="demographic_indicators_500m_hexagons", driver="GPKG")
    municipality.to_file(output_gpkg, layer="odense_municipality_districts", driver="GPKG")
    demographic_source.to_file(output_gpkg, layer="demographic_source", driver="GPKG")

    shp_cols = [
        "hex_id",
        "population_total",
        "pop_density_km2",
        "children_share",
        "elderly_share",
        "age_vulnerability_share",
        "socioeconomic_vulnerability_share",
        "demo_vulnerability_score",
        "demo_vulnerability_class",
        "demo_vulnerability_rank",
        "geometry",
    ]

    shp_output = hex_output[shp_cols].copy().rename(columns={
        "population_total": "pop_total",
        "pop_density_km2": "pop_den",
        "children_share": "child_sh",
        "elderly_share": "elder_sh",
        "age_vulnerability_share": "age_vuln",
        "socioeconomic_vulnerability_share": "soc_vuln",
        "demo_vulnerability_score": "dem_scr",
        "demo_vulnerability_class": "dem_cls",
        "demo_vulnerability_rank": "dem_rnk",
    })

    shp_path = output_shp_dir / "odense_demographic_indicators_500m.shp"
    shp_output.to_file(shp_path, driver="ESRI Shapefile", encoding="UTF-8")

    plot_class_map(
        hex_output,
        municipality,
        gpd.GeoDataFrame(geometry=[municipality.geometry.union_all()], crs=target_crs),
        "pop_density_class",
        "Odense Population Density by 500 m Cell",
        "Population density",
        sequential_colors(),
        output_population_map,
    )

    plot_class_map(
        hex_output,
        municipality,
        gpd.GeoDataFrame(geometry=[municipality.geometry.union_all()], crs=target_crs),
        "age_vulnerability_class",
        "Odense Age Vulnerability by 500 m Cell",
        "Age vulnerability",
        vulnerability_colors(),
        output_age_map,
    )

    plot_class_map(
        hex_output,
        municipality,
        gpd.GeoDataFrame(geometry=[municipality.geometry.union_all()], crs=target_crs),
        "socioeconomic_vulnerability_class",
        "Odense Socio-economic Vulnerability by 500 m Cell",
        "Socio-economic vulnerability",
        vulnerability_colors(),
        output_socio_map,
    )

    plot_class_map(
        hex_output,
        municipality,
        gpd.GeoDataFrame(geometry=[municipality.geometry.union_all()], crs=target_crs),
        "demo_vulnerability_class",
        "Demographic vulnerability index",
        "Demographic vulnerability",
        vulnerability_colors(),
        output_vulnerability_map,
    )
    plot_osm_basemap_vulnerability_map(
        hex_output,
        municipality,
        gpd.GeoDataFrame(geometry=[municipality.geometry.union_all()], crs=target_crs),
        output_vulnerability_map_osm,
    )

    validation_report.write_text(
        "\n".join(create_validation_report(hex_output, detected_columns, weights_used)),
        encoding="utf-8",
    )

    summary = (
        hex_output.groupby("demo_vulnerability_class")
        .agg(
            n_cells=("hex_id", "count"),
            mean_score=("demo_vulnerability_score", "mean"),
            total_population=("population_total", "sum"),
            mean_pop_density=("pop_density_km2", "mean"),
            mean_age_vulnerability=("age_vulnerability_share", "mean"),
            mean_socioeconomic_vulnerability=("socioeconomic_vulnerability_share", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "demographic_vulnerability_class_summary.csv", index=False)

    print(f"GeoPackage saved to: {output_gpkg}")
    print(f"Shapefile saved to: {shp_path}")
    print(f"Validation report saved to: {validation_report}")


# ============================================================
# 10. MAIN
# ============================================================

def main():
    municipality, municipality_boundary, hexagons = load_boundary_and_hexagons()
    demo = load_demographic_data()
    demo, detected_columns = standardise_demographic_columns(demo)
    demo = gpd.clip(demo, municipality_boundary)

    if is_polygon_layer(demo):
        hex_output = aggregate_polygon_demographics_to_hex(hexagons, demo)
    else:
        hex_output = aggregate_point_demographics_to_hex(hexagons, demo)

    hex_output, weights_used = compute_demographic_indicators(hex_output)

    save_outputs(
        hex_output=hex_output,
        municipality=municipality,
        demographic_source=demo,
        detected_columns=detected_columns,
        weights_used=weights_used,
    )

    print("Finished.")


if __name__ == "__main__":
    main()
