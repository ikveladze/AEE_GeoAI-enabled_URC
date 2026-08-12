# ============================================================
# Odense Multi-source Environmental Index
# 500 m hexagonal cells | OSM + optional official raster/vector data
# CRS: EPSG:25832
# ============================================================

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import osmnx as ox
import pandas as pd

warnings.filterwarnings("ignore")

try:
    import rasterio
    from rasterstats import zonal_stats
    HAS_RASTER = True
except Exception:
    HAS_RASTER = False


# ============================================================
# 1. PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "Map Layers").exists() else SCRIPT_DIR
MAP_LAYERS = PROJECT_DIR / "Map Layers"
OFFICIAL_DIR = MAP_LAYERS / "Official Environmental"

MUNICIPALITY_CANDIDATES = [
    MAP_LAYERS / "Odense_Municipality.gpkg",
    MAP_LAYERS / "Odense_Municipality_1.gpkg",
    MAP_LAYERS / "Odense_Municipality.shp",
]

hexgrid_gpkg = MAP_LAYERS / "Odense-500mHexaCells_1.gpkg"

municipality_layer = "odense_municipality_districts"
municipality_layer_shp = "Odense_Municipality"
hexgrid_layer = "odense500mhexacells__grid"

target_crs = "EPSG:25832"
web_mercator = "EPSG:3857"

output_dir = SCRIPT_DIR / "odense_multisource_environmental_index_outputs"
output_dir.mkdir(exist_ok=True)

output_gpkg = output_dir / "odense_multisource_environmental_index_500m.gpkg"
output_shp_dir = output_dir / "odense_multisource_environmental_index_500m_shp"
output_shp_dir.mkdir(exist_ok=True)

output_quality_map = output_dir / "odense_environmental_quality_index_500m_map.png"
output_burden_map = output_dir / "odense_environmental_burden_index_500m_map.png"
output_quality_map_osm = output_dir / "odense_environmental_quality_index_500m_map_osm.png"
validation_report = output_dir / "multisource_environmental_index_validation.txt"

hex_alpha = 0.78
include_agriculture_in_green = False
major_road_buffer_m = 100


# ============================================================
# 2. OSM TAGS
# ============================================================

green_tags = {
    "leisure": ["park", "garden", "nature_reserve", "recreation_ground", "common",
                "playground", "pitch", "sports_centre", "golf_course"],
    "landuse": ["forest", "grass", "meadow", "recreation_ground", "village_green",
                "allotments", "orchard", "vineyard", "cemetery"],
    "natural": ["wood", "grassland", "scrub", "heath", "wetland", "fell", "bare_rock"],
    "boundary": ["national_park", "protected_area"],
}

agriculture_tags = {"landuse": ["farmland", "farmyard", "pasture"]}

blue_tags = {
    "natural": ["water", "bay"],
    "waterway": ["riverbank", "dock"],
    "landuse": ["reservoir", "basin"],
    "water": True,
}

builtup_tags = {
    "building": True,
    "landuse": ["residential", "commercial", "industrial", "retail",
                "construction", "garages", "brownfield"],
    "amenity": ["parking"],
}

major_road_tags = {
    "highway": ["motorway", "trunk", "primary", "secondary", "tertiary",
                "motorway_link", "trunk_link", "primary_link", "secondary_link"]
}


# ============================================================
# 3. OPTIONAL OFFICIAL DATA CONFIGURATION
# ============================================================
# Put optional files in: Map Layers/Official Environmental/
# The script runs with OSM only if these files are absent.

OFFICIAL_VECTOR_LAYERS = {
    "official_landcover_natural": {
        "path_candidates": [
            OFFICIAL_DIR / "official_landcover_natural.gpkg",
            OFFICIAL_DIR / "official_landcover_natural.shp",
        ],
        "area_col": "official_natural_m2",
        "share_col": "official_natural_share",
        "pct_col": "official_natural_pct",
    },
    "official_blue": {
        "path_candidates": [
            OFFICIAL_DIR / "official_blue.gpkg",
            OFFICIAL_DIR / "official_blue.shp",
        ],
        "area_col": "official_blue_m2",
        "share_col": "official_blue_share",
        "pct_col": "official_blue_pct",
    },
    "official_builtup": {
        "path_candidates": [
            OFFICIAL_DIR / "official_builtup.gpkg",
            OFFICIAL_DIR / "official_builtup.shp",
        ],
        "area_col": "official_builtup_m2",
        "share_col": "official_builtup_share",
        "pct_col": "official_builtup_pct",
    },
    "ecological_health_area": {
        "path_candidates": [
            OFFICIAL_DIR / "ecological_health_vector.gpkg",
            OFFICIAL_DIR / "ecological_health_vector.shp",
            OFFICIAL_DIR / "protected_nature.gpkg",
            OFFICIAL_DIR / "protected_nature.shp",
            OFFICIAL_DIR / "paragraf3_protected_nature.gpkg",
        ],
        "area_col": "eco_area_m2",
        "share_col": "eco_share",
        "pct_col": "eco_pct",
    },
    "noise_exposure_area": {
        "path_candidates": [
            OFFICIAL_DIR / "noise_vector.gpkg",
            OFFICIAL_DIR / "noise_vector.shp",
        ],
        "area_col": "noise_area_m2",
        "share_col": "noise_share",
        "pct_col": "noise_pct",
    },
}

OFFICIAL_RASTER_LAYERS = {
    "landcover_raster": {
        "path": OFFICIAL_DIR / "landcover_raster.tif",
        "col": "lc_natural_mean",
        "polarity": "positive",
    },
    "ecological_health_raster": {
        "path": OFFICIAL_DIR / "ecological_health_raster.tif",
        "col": "eco_health_mean",
        "polarity": "positive",
    },
    "air_quality_raster": {
        "path": OFFICIAL_DIR / "air_quality_raster.tif",
        "col": "air_pollution_mean",
        "polarity": "negative",
    },
    "heat_raster": {
        "path": OFFICIAL_DIR / "heat_raster.tif",
        "col": "heat_mean",
        "polarity": "negative",
    },
    "noise_raster": {
        "path": OFFICIAL_DIR / "noise_raster.tif",
        "col": "noise_mean",
        "polarity": "negative",
    },
}


# Candidate weights. Only available non-constant indicators are used;
# weights are automatically re-normalised.

QUALITY_WEIGHT_CANDIDATES = {
    "green_share_norm": 0.20,
    "blue_share_norm": 0.08,
    "builtup_inverse_norm": 0.15,
    "road_burden_inverse_norm": 0.10,
    "official_natural_share_norm": 0.15,
    "eco_share_norm": 0.15,
    "noise_share_inverse_norm": 0.08,
    "lc_natural_mean_norm": 0.15,
    "eco_health_mean_norm": 0.15,
    "air_pollution_mean_inverse_norm": 0.10,
    "heat_mean_inverse_norm": 0.10,
    "noise_mean_inverse_norm": 0.10,
}

BURDEN_WEIGHT_CANDIDATES = {
    "low_green_norm": 0.15,
    "low_blue_norm": 0.05,
    "builtup_share_norm": 0.18,
    "road_burden_norm": 0.15,
    "low_official_natural_norm": 0.12,
    "low_eco_share_norm": 0.12,
    "noise_share_norm": 0.10,
    "low_lc_natural_mean_norm": 0.12,
    "low_eco_health_mean_norm": 0.12,
    "air_pollution_mean_norm": 0.15,
    "heat_mean_norm": 0.15,
    "noise_mean_norm": 0.15,
}


# ============================================================
# 4. HELPERS
# ============================================================

def resolve_municipality_path() -> tuple[Path, str | None]:
    for path in MUNICIPALITY_CANDIDATES:
        if path.exists():
            layer = municipality_layer_shp if path.suffix.lower() == ".shp" else municipality_layer
            return path, layer
    raise FileNotFoundError("Municipality boundary not found in Map Layers.")


def load_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    muni_path, muni_layer = resolve_municipality_path()
    if not hexgrid_gpkg.exists():
        raise FileNotFoundError(f"Missing hex grid file: {hexgrid_gpkg}")

    municipality = gpd.read_file(muni_path, layer=muni_layer) if muni_layer else gpd.read_file(muni_path)
    hexagons = gpd.read_file(hexgrid_gpkg, layer=hexgrid_layer)

    municipality = municipality.to_crs(target_crs)
    hexagons = hexagons.to_crs(target_crs)

    municipality_boundary = gpd.GeoDataFrame(geometry=[municipality.geometry.union_all()], crs=target_crs)

    hexagons = gpd.clip(hexagons, municipality_boundary).reset_index(drop=True)
    hexagons["hex_id"] = [f"OD_ENV_{i + 1:04d}" for i in range(len(hexagons))]
    hexagons["cell_area_m2"] = hexagons.geometry.area

    print(f"Number of clipped 500 m hexagons: {len(hexagons)}")
    print(f"Municipality area: {municipality_boundary.geometry.area.iloc[0] / 1_000_000:.2f} km²")
    return municipality, hexagons, municipality_boundary


def merge_tags(base: dict, extra: dict | None = None) -> dict:
    tags = {k: (list(v) if isinstance(v, list) else v) for k, v in base.items()}
    if extra:
        for key, values in extra.items():
            if key not in tags:
                tags[key] = list(values) if isinstance(values, list) else values
            elif isinstance(tags[key], list) and isinstance(values, list):
                tags[key].extend(values)
            else:
                tags[key] = values
    return tags


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
    series = series.fillna(series.mean()) if series.notna().any() else series.fillna(0.0)
    try:
        ranks = pd.qcut(series, q=5, labels=False, duplicates="drop")
    except ValueError:
        ranks = pd.Series(0, index=series.index)
    rank_map = {b: i + 1 for i, b in enumerate(sorted(pd.Series(ranks).dropna().unique()))}
    rank = pd.Series(ranks).map(rank_map).fillna(1).astype(int)
    labels = ["Very low", "Low", "Moderate", "High", "Very high"]
    class_map = {i + 1: labels[i] for i in range(int(rank.max()))}
    return rank, rank.map(class_map)


def find_first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


# ============================================================
# 5. OSM DOWNLOAD AND AREA SHARE
# ============================================================

def download_osm_polygons(municipality_boundary: gpd.GeoDataFrame, tags: dict, layer_name: str) -> gpd.GeoDataFrame:
    boundary_polygon_wgs84 = municipality_boundary.to_crs("EPSG:4326").geometry.iloc[0]
    print(f"Downloading OSM polygon features for: {layer_name}")

    empty = gpd.GeoDataFrame(columns=["source_type", "geometry"], geometry="geometry", crs=target_crs)

    try:
        features = ox.features_from_polygon(boundary_polygon_wgs84, tags=tags)
    except Exception as e:
        print(f"Warning: no OSM features downloaded for {layer_name}: {e}")
        return empty

    if features.empty:
        print(f"No OSM features found for {layer_name}")
        return empty

    features = features.reset_index()
    features = features[features.geometry.notnull()].copy()
    features = features.to_crs(target_crs)
    features = features[features.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()

    if features.empty:
        return empty

    features["geometry"] = features.geometry.make_valid()
    features = features[~features.geometry.is_empty].copy()

    def infer_type(row):
        for key in ["leisure", "landuse", "natural", "water", "waterway", "building", "amenity", "boundary"]:
            if key in row and pd.notna(row[key]):
                return f"{key}={row[key]}"
        return layer_name

    features["source_type"] = features.apply(infer_type, axis=1)
    features = gpd.clip(features[["source_type", "geometry"]], municipality_boundary)
    features = features.reset_index(drop=True)
    features[f"{layer_name}_id"] = [f"{layer_name.upper()}_{i + 1:05d}" for i in range(len(features))]
    features["area_m2"] = features.geometry.area

    print(f"{layer_name}: {len(features)} polygon features after clipping")
    print(f"{layer_name}: raw summed area before union = {features['area_m2'].sum() / 1_000_000:.2f} km²")
    return features


def download_osm_major_roads(municipality_boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    boundary_polygon_wgs84 = municipality_boundary.to_crs("EPSG:4326").geometry.iloc[0]
    print("Downloading OSM major-road features...")

    try:
        roads = ox.features_from_polygon(boundary_polygon_wgs84, tags=major_road_tags)
    except Exception as e:
        print(f"Warning: no OSM major-road features downloaded: {e}")
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)

    if roads.empty:
        return gpd.GeoDataFrame(columns=["geometry"], geometry="geometry", crs=target_crs)

    roads = roads.reset_index()
    roads = roads[roads.geometry.notnull()].copy()
    roads = roads.to_crs(target_crs)
    roads["geometry"] = roads.geometry.make_valid()
    roads = roads[~roads.geometry.is_empty].copy()
    roads = gpd.clip(roads[["geometry"]], municipality_boundary).reset_index(drop=True)
    print(f"Major-road features after clipping: {len(roads)}")
    return roads


def area_share_by_hex(hexagons, polygons, area_col, share_col, pct_col):
    out = hexagons.copy()
    if polygons.empty:
        out[area_col] = 0.0
        out[share_col] = 0.0
        out[pct_col] = 0.0
        return out

    polygons = polygons.to_crs(target_crs)
    polygons = polygons[polygons.geometry.notnull()].copy()
    polygons = polygons[polygons.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    if polygons.empty:
        out[area_col] = 0.0
        out[share_col] = 0.0
        out[pct_col] = 0.0
        return out

    polygons["geometry"] = polygons.geometry.make_valid()
    polygons = polygons[~polygons.geometry.is_empty].copy()

    print(f"Dissolving polygons for {area_col} to avoid double counting...")
    union_gdf = gpd.GeoDataFrame(geometry=[polygons.geometry.union_all()], crs=target_crs)

    intersections = gpd.overlay(
        out[["hex_id", "cell_area_m2", "geometry"]],
        union_gdf[["geometry"]],
        how="intersection",
        keep_geom_type=True,
    )

    if intersections.empty:
        area_by_hex = pd.DataFrame({"hex_id": out["hex_id"], area_col: 0.0})
    else:
        intersections["intersect_area_m2"] = intersections.geometry.area
        area_by_hex = intersections.groupby("hex_id", as_index=False).agg(
            **{area_col: ("intersect_area_m2", "sum")}
        )

    out = out.merge(area_by_hex, on="hex_id", how="left")
    out[area_col] = out[area_col].fillna(0.0)
    out[area_col] = np.minimum(out[area_col], out["cell_area_m2"])
    out[share_col] = out[area_col] / out["cell_area_m2"]
    out[pct_col] = (out[share_col] * 100).round(2)
    return out


def road_burden_by_hex(hexagons, roads, municipality_boundary):
    if roads.empty:
        out = hexagons.copy()
        out["road_buf_m2"] = 0.0
        out["road_buf_share"] = 0.0
        out["road_buf_pct"] = 0.0
        return out

    print(f"Buffering major roads by {major_road_buffer_m} m...")
    road_buffer = gpd.GeoDataFrame(geometry=roads.geometry.buffer(major_road_buffer_m), crs=target_crs)
    road_buffer["geometry"] = road_buffer.geometry.make_valid()
    road_buffer = gpd.clip(road_buffer, municipality_boundary)
    return area_share_by_hex(hexagons, road_buffer, "road_buf_m2", "road_buf_share", "road_buf_pct")


# ============================================================
# 6. OPTIONAL OFFICIAL DATA
# ============================================================

def add_optional_official_vector_indicators(hexagons, municipality_boundary):
    out = hexagons.copy()
    loaded = {}

    for name, cfg in OFFICIAL_VECTOR_LAYERS.items():
        path = find_first_existing(cfg["path_candidates"])
        if path is None:
            print(f"Optional official vector layer not found: {name}")
            out[cfg["area_col"]] = 0.0
            out[cfg["share_col"]] = 0.0
            out[cfg["pct_col"]] = 0.0
            continue

        print(f"Loading official vector layer {name}: {path}")
        gdf = gpd.read_file(path).to_crs(target_crs)
        gdf = gpd.clip(gdf, municipality_boundary)
        loaded[name] = gdf.copy()
        out = area_share_by_hex(out, gdf, cfg["area_col"], cfg["share_col"], cfg["pct_col"])

    return out, loaded


def add_optional_official_raster_indicators(hexagons):
    out = hexagons.copy()
    loaded = {}

    if not HAS_RASTER:
        print("rasterio/rasterstats not installed; optional raster indicators skipped.")
        for cfg in OFFICIAL_RASTER_LAYERS.values():
            out[cfg["col"]] = np.nan
        return out, loaded

    for name, cfg in OFFICIAL_RASTER_LAYERS.items():
        path = cfg["path"]
        if not path.exists():
            print(f"Optional official raster not found: {name} -> {path}")
            out[cfg["col"]] = np.nan
            continue

        print(f"Calculating zonal mean for raster {name}: {path}")
        with rasterio.open(path) as src:
            raster_crs = src.crs
            nodata = src.nodata

        zones = out[["hex_id", "geometry"]].copy()
        if raster_crs is not None:
            zones = zones.to_crs(raster_crs)

        stats = zonal_stats(zones, str(path), stats=["mean"], nodata=nodata, all_touched=True)
        out[cfg["col"]] = [s.get("mean", np.nan) for s in stats]
        loaded[name] = path

    return out, loaded


# ============================================================
# 7. INDEX CALCULATION
# ============================================================

def add_normalised_indicators(out):
    out = out.copy()

    out["green_share_norm"] = minmax(out["green_share"])
    out["blue_share_norm"] = minmax(out["blue_share"])
    out["builtup_share_norm"] = minmax(out["builtup_share"])
    out["road_burden_norm"] = minmax(out["road_buf_share"])

    out["builtup_inverse_norm"] = 1 - out["builtup_share_norm"]
    out["road_burden_inverse_norm"] = 1 - out["road_burden_norm"]
    out["low_green_norm"] = 1 - out["green_share_norm"]
    out["low_blue_norm"] = 1 - out["blue_share_norm"]

    optional_pairs = [
        ("official_natural_share", "official_natural_share_norm", "low_official_natural_norm"),
        ("eco_share", "eco_share_norm", "low_eco_share_norm"),
        ("noise_share", "noise_share_norm", "noise_share_inverse_norm"),
        ("lc_natural_mean", "lc_natural_mean_norm", "low_lc_natural_mean_norm"),
        ("eco_health_mean", "eco_health_mean_norm", "low_eco_health_mean_norm"),
        ("air_pollution_mean", "air_pollution_mean_norm", "air_pollution_mean_inverse_norm"),
        ("heat_mean", "heat_mean_norm", "heat_mean_inverse_norm"),
        ("noise_mean", "noise_mean_norm", "noise_mean_inverse_norm"),
    ]

    for src, norm, inv in optional_pairs:
        if src in out.columns and out[src].notna().any():
            out[norm] = minmax(out[src])
            out[inv] = 1 - out[norm]

    return out


def weighted_score(df, candidates, name):
    available = {}
    for col, weight in candidates.items():
        if col in df.columns and df[col].notna().any():
            s = pd.to_numeric(df[col], errors="coerce")
            if not np.isclose(s.std(skipna=True), 0):
                available[col] = weight

    if not available:
        raise ValueError(f"No valid indicators available for {name}")

    total = sum(available.values())
    weights = {col: w / total for col, w in available.items()}
    score = sum(df[col].fillna(df[col].mean()) * w for col, w in weights.items())
    return score, weights


def compute_environmental_indices(hexagons, green, blue, builtup, roads, municipality_boundary):
    out = hexagons.copy()

    out = area_share_by_hex(out, green, "green_m2", "green_share", "green_pct")
    out = area_share_by_hex(out, blue, "blue_m2", "blue_share", "blue_pct")
    out = area_share_by_hex(out, builtup, "builtup_m2", "builtup_share", "builtup_pct")
    out = road_burden_by_hex(out, roads, municipality_boundary)

    out, official_vectors = add_optional_official_vector_indicators(out, municipality_boundary)
    out, official_rasters = add_optional_official_raster_indicators(out)

    out = add_normalised_indicators(out)

    out["env_quality_raw"], quality_weights_used = weighted_score(out, QUALITY_WEIGHT_CANDIDATES, "quality")
    out["env_quality_score"] = (minmax(out["env_quality_raw"]) * 100).round(2)
    out["env_quality_rank"], out["env_quality_class"] = classify_quintiles(out["env_quality_score"])

    out["env_burden_raw"], burden_weights_used = weighted_score(out, BURDEN_WEIGHT_CANDIDATES, "burden")
    out["env_burden_score"] = (minmax(out["env_burden_raw"]) * 100).round(2)
    out["env_burden_rank"], out["env_burden_class"] = classify_quintiles(out["env_burden_score"])

    return out, official_vectors, official_rasters, quality_weights_used, burden_weights_used


# ============================================================
# 8. MAPS AND OUTPUTS
# ============================================================

def quality_colors():
    return {"Very low": "#ffffcc", "Low": "#c2e699", "Moderate": "#78c679", "High": "#31a354", "Very high": "#006837"}


def burden_colors():
    return {"Very low": "#ffffcc", "Low": "#fed976", "Moderate": "#fd8d3c", "High": "#f03b20", "Very high": "#bd0026"}


def plot_index_map(hex_output, municipality, municipality_boundary, class_col, title, legend_title, colors, output_path):
    plot_gdf = hex_output.copy()
    plot_gdf["map_color"] = plot_gdf[class_col].map(colors)

    fig, ax = plt.subplots(figsize=(14, 10))
    plot_gdf.plot(ax=ax, color=plot_gdf["map_color"], edgecolor="white", linewidth=0.15, alpha=0.96)
    municipality.boundary.plot(ax=ax, linewidth=0.6, edgecolor="black", alpha=0.8)
    municipality_boundary.boundary.plot(ax=ax, linewidth=1.2, edgecolor="black")
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_axis_off()

    legend_order = [v for v in ["Very high", "High", "Moderate", "Low", "Very low"] if v in plot_gdf[class_col].unique()]
    patches = [mpatches.Patch(facecolor=colors[v], edgecolor="gray", label=v) for v in legend_order]
    ax.legend(handles=patches, title=legend_title, loc="lower left", frameon=True, fontsize=10, title_fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Map saved to: {output_path}")


def plot_osm_basemap_quality_map(hex_output, municipality, municipality_boundary):
    try:
        import contextily as cx
    except ImportError:
        print("contextily not installed; skipping OSM basemap map.")
        return

    colors = quality_colors()
    hex_wm = hex_output.to_crs(web_mercator).copy()
    muni_wm = municipality.to_crs(web_mercator)
    boundary_wm = municipality_boundary.to_crs(web_mercator)
    hex_wm["map_color"] = hex_wm["env_quality_class"].map(colors)

    fig, ax = plt.subplots(figsize=(14, 10))
    boundary_wm.plot(ax=ax, facecolor="none", edgecolor="none")
    try:
        cx.add_basemap(ax, crs=hex_wm.crs, source=cx.providers.OpenStreetMap.Mapnik, zoom="auto")
    except Exception:
        cx.add_basemap(ax, crs=hex_wm.crs, source=cx.providers.CartoDB.Positron)

    hex_wm.plot(ax=ax, color=hex_wm["map_color"], edgecolor="white", linewidth=0.2, alpha=hex_alpha)
    muni_wm.boundary.plot(ax=ax, linewidth=0.5, edgecolor="#333333", alpha=0.85)
    boundary_wm.boundary.plot(ax=ax, linewidth=1.2, edgecolor="black")
    ax.set_title("Odense Environmental Quality Index by 500 m Cell\n(multi-source indicators + OSM basemap)",
                 fontsize=16, fontweight="bold")
    ax.set_axis_off()

    legend_order = [v for v in ["Very high", "High", "Moderate", "Low", "Very low"] if v in hex_wm["env_quality_class"].unique()]
    patches = [mpatches.Patch(facecolor=colors[v], edgecolor="gray", label=v, alpha=hex_alpha) for v in legend_order]
    ax.legend(handles=patches, title="Environmental quality", loc="lower left", frameon=True, fontsize=10, title_fontsize=11)

    plt.tight_layout()
    plt.savefig(output_quality_map_osm, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"OSM basemap map saved to: {output_quality_map_osm}")


def validation_lines(hex_output, municipality_boundary, green, blue, builtup, roads, vectors, rasters, q_weights, b_weights):
    lines = ["Multi-source environmental index validation", "=" * 55]
    lines.append(f"include_agriculture_in_green: {include_agriculture_in_green}")
    lines.append(f"Raster support installed: {HAS_RASTER}")
    lines.append(f"Number of 500 m cells: {len(hex_output)}")
    lines.append(f"Municipality area: {municipality_boundary.geometry.area.iloc[0] / 1_000_000:.2f} km²")
    lines.append("")
    lines.append("OSM input feature counts:")
    lines.append(f"  green polygons: {len(green)}")
    lines.append(f"  blue polygons: {len(blue)}")
    lines.append(f"  built-up polygons: {len(builtup)}")
    lines.append(f"  major roads: {len(roads)}")
    lines.append("")
    lines.append("Official vector layers used:")
    lines.extend([f"  {k}: {len(v)} features" for k, v in vectors.items()] or ["  none"])
    lines.append("Official raster layers used:")
    lines.extend([f"  {k}: {v}" for k, v in rasters.items()] or ["  none"])
    lines.append("")
    for col in ["green_pct", "blue_pct", "builtup_pct", "road_buf_pct", "official_natural_pct",
                "eco_pct", "noise_pct", "lc_natural_mean", "eco_health_mean", "air_pollution_mean",
                "heat_mean", "noise_mean"]:
        if col in hex_output.columns and hex_output[col].notna().any():
            s = hex_output[col].dropna()
            lines.append(f"{col}: min={s.min():.2f}; mean={s.mean():.2f}; max={s.max():.2f}")
    lines.append("")
    lines.append("Quality weights used:")
    lines.extend([f"  {k}: {v:.3f}" for k, v in q_weights.items()])
    lines.append("Burden weights used:")
    lines.extend([f"  {k}: {v:.3f}" for k, v in b_weights.items()])
    return lines


def save_outputs(hex_output, municipality, municipality_boundary, green, blue, builtup, roads,
                 vectors, rasters, q_weights, b_weights):
    hex_output.to_file(output_gpkg, layer="environmental_index_500m_hexagons", driver="GPKG")
    municipality.to_file(output_gpkg, layer="odense_municipality_districts", driver="GPKG")
    if len(green): green.to_file(output_gpkg, layer="osm_green_features", driver="GPKG")
    if len(blue): blue.to_file(output_gpkg, layer="osm_blue_features", driver="GPKG")
    if len(builtup): builtup.to_file(output_gpkg, layer="osm_builtup_features", driver="GPKG")
    if len(roads): roads.to_file(output_gpkg, layer="osm_major_roads", driver="GPKG")
    for name, gdf in vectors.items():
        if len(gdf):
            gdf.to_file(output_gpkg, layer=name[:60], driver="GPKG")

    shp_cols = [
        "hex_id", "cell_area_m2", "green_pct", "blue_pct", "builtup_pct", "road_buf_pct",
        "env_quality_score", "env_quality_class", "env_quality_rank",
        "env_burden_score", "env_burden_class", "env_burden_rank", "geometry",
    ]
    for c in ["official_natural_pct", "eco_pct", "noise_pct", "air_pollution_mean", "heat_mean", "noise_mean"]:
        if c in hex_output.columns:
            shp_cols.insert(-1, c)

    shp = hex_output[shp_cols].copy().rename(columns={
        "cell_area_m2": "cell_m2",
        "builtup_pct": "built_pct",
        "road_buf_pct": "road_pct",
        "official_natural_pct": "nat_pct",
        "air_pollution_mean": "air_mean",
        "env_quality_score": "qual_scr",
        "env_quality_class": "qual_cls",
        "env_quality_rank": "qual_rnk",
        "env_burden_score": "burd_scr",
        "env_burden_class": "burd_cls",
        "env_burden_rank": "burd_rnk",
    })
    shp.to_file(output_shp_dir / "odense_multisource_environmental_index_500m.shp",
                driver="ESRI Shapefile", encoding="UTF-8")

    plot_index_map(hex_output, municipality, municipality_boundary, "env_quality_class",
                   "Odense Environmental Quality Index by 500 m Cell",
                   "Environmental quality", quality_colors(), output_quality_map)
    plot_index_map(hex_output, municipality, municipality_boundary, "env_burden_class",
                   "Odense Environmental Burden Index by 500 m Cell",
                   "Environmental burden", burden_colors(), output_burden_map)
    plot_osm_basemap_quality_map(hex_output, municipality, municipality_boundary)

    validation_report.write_text(
        "\n".join(validation_lines(hex_output, municipality_boundary, green, blue, builtup, roads,
                                   vectors, rasters, q_weights, b_weights)),
        encoding="utf-8",
    )
    print(f"GeoPackage saved to: {output_gpkg}")
    print(f"Validation report saved to: {validation_report}")


# ============================================================
# 9. MAIN
# ============================================================

def main():
    municipality, hexagons, municipality_boundary = load_inputs()

    green_query_tags = merge_tags(green_tags, agriculture_tags if include_agriculture_in_green else None)
    green = download_osm_polygons(municipality_boundary, green_query_tags, "green")
    blue = download_osm_polygons(municipality_boundary, blue_tags, "blue")
    builtup = download_osm_polygons(municipality_boundary, builtup_tags, "builtup")
    roads = download_osm_major_roads(municipality_boundary)

    hex_output, vectors, rasters, q_weights, b_weights = compute_environmental_indices(
        hexagons, green, blue, builtup, roads, municipality_boundary
    )

    save_outputs(hex_output, municipality, municipality_boundary, green, blue, builtup, roads,
                 vectors, rasters, q_weights, b_weights)

    print("Finished.")


if __name__ == "__main__":
    main()
