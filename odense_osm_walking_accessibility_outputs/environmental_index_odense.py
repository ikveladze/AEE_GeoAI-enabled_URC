# ============================================================
# Odense Environmental Index Map
# 500 m hexagonal cells, OSM-based environmental indicators
# Output: GeoPackage + Shapefile + maps + validation report
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

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent if (SCRIPT_DIR.parent / "Map Layers").exists() else SCRIPT_DIR
MAP_LAYERS = PROJECT_DIR / "Map Layers"

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

output_dir = SCRIPT_DIR / "odense_osm_environmental_index_outputs"
output_dir.mkdir(exist_ok=True)

output_gpkg = output_dir / "odense_environmental_index_500m.gpkg"
output_shp_dir = output_dir / "odense_environmental_index_500m_shp"
output_shp_dir.mkdir(exist_ok=True)

output_quality_map = output_dir / "odense_environmental_quality_index_500m_map.png"
output_burden_map = output_dir / "odense_environmental_burden_index_500m_map.png"
output_quality_map_osm = output_dir / "odense_environmental_quality_index_500m_map_osm.png"
validation_report = output_dir / "environmental_index_validation.txt"

hex_alpha = 0.78

include_agriculture_in_green = False

green_tags = {
    "leisure": [
        "park", "garden", "nature_reserve", "recreation_ground", "common",
        "playground", "pitch", "sports_centre", "golf_course",
    ],
    "landuse": [
        "forest", "grass", "meadow", "recreation_ground", "village_green",
        "allotments", "orchard", "vineyard", "cemetery",
    ],
    "natural": [
        "wood", "grassland", "scrub", "heath", "wetland", "fell", "bare_rock",
    ],
    "boundary": ["national_park", "protected_area"],
}

agriculture_tags = {
    "landuse": ["farmland", "farmyard", "pasture"]
}

blue_tags = {
    "natural": ["water", "bay"],
    "waterway": ["riverbank", "dock"],
    "landuse": ["reservoir", "basin"],
    "water": True,
}

builtup_tags = {
    "building": True,
    "landuse": [
        "residential", "commercial", "industrial", "retail",
        "construction", "garages", "brownfield",
    ],
    "amenity": ["parking"],
}

major_road_tags = {
    "highway": [
        "motorway", "trunk", "primary", "secondary", "tertiary",
        "motorway_link", "trunk_link", "primary_link", "secondary_link",
    ]
}

major_road_buffer_m = 100

quality_weights = {
    "green_share_norm": 0.45,
    "blue_share_norm": 0.15,
    "builtup_inverse_norm": 0.25,
    "road_burden_inverse_norm": 0.15,
}

burden_weights = {
    "low_green_norm": 0.35,
    "low_blue_norm": 0.10,
    "builtup_share_norm": 0.30,
    "road_burden_norm": 0.25,
}


def _resolve_municipality_path() -> tuple[Path, str | None]:
    for path in MUNICIPALITY_CANDIDATES:
        if path.exists():
            layer = municipality_layer_shp if path.suffix.lower() == ".shp" else municipality_layer
            return path, layer
    raise FileNotFoundError(
        "Municipality boundary not found. Expected one of:\n"
        + "\n".join(f"  - {p}" for p in MUNICIPALITY_CANDIDATES)
    )


def load_inputs() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame]:
    muni_path, muni_layer = _resolve_municipality_path()
    if not hexgrid_gpkg.exists():
        raise FileNotFoundError(f"Missing hex grid file: {hexgrid_gpkg}")

    municipality = gpd.read_file(muni_path, layer=muni_layer) if muni_layer else gpd.read_file(muni_path)
    hexagons = gpd.read_file(hexgrid_gpkg, layer=hexgrid_layer)

    municipality = municipality.to_crs(target_crs)
    hexagons = hexagons.to_crs(target_crs)

    municipality_boundary = gpd.GeoDataFrame(
        geometry=[municipality.geometry.union_all()],
        crs=target_crs,
    )

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


def download_osm_polygons(
    municipality_boundary: gpd.GeoDataFrame,
    tags: dict,
    layer_name: str,
) -> gpd.GeoDataFrame:
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
        print(f"No polygonal features found for {layer_name}")
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
    print(f"{layer_name}: summed raw area before union = {features['area_m2'].sum() / 1_000_000:.2f} km²")
    if len(features):
        print(features["source_type"].value_counts().head(15))

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
    roads = gpd.clip(roads[["geometry"]], municipality_boundary)
    roads = roads.reset_index(drop=True)

    print(f"Major-road features after clipping: {len(roads)}")
    return roads


def area_share_by_hex(
    hexagons: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    area_col_name: str,
    share_col_name: str,
    pct_col_name: str,
) -> gpd.GeoDataFrame:
    out = hexagons.copy()

    if polygons.empty:
        out[area_col_name] = 0.0
        out[share_col_name] = 0.0
        out[pct_col_name] = 0.0
        return out

    print(f"Dissolving polygons for {area_col_name} to avoid overlap double-counting...")
    union_geom = polygons.geometry.union_all()
    union_gdf = gpd.GeoDataFrame(geometry=[union_geom], crs=target_crs)

    print(f"Intersecting {area_col_name} with hexagons...")
    intersections = gpd.overlay(
        out[["hex_id", "cell_area_m2", "geometry"]],
        union_gdf[["geometry"]],
        how="intersection",
        keep_geom_type=True,
    )

    if intersections.empty:
        area_by_hex = pd.DataFrame({"hex_id": out["hex_id"], area_col_name: 0.0})
    else:
        intersections["intersect_area_m2"] = intersections.geometry.area
        area_by_hex = (
            intersections.groupby("hex_id", as_index=False)
            .agg(**{area_col_name: ("intersect_area_m2", "sum")})
        )

    out = out.merge(area_by_hex, on="hex_id", how="left")
    out[area_col_name] = out[area_col_name].fillna(0.0)
    out[area_col_name] = np.minimum(out[area_col_name], out["cell_area_m2"])
    out[share_col_name] = out[area_col_name] / out["cell_area_m2"]
    out[pct_col_name] = (out[share_col_name] * 100).round(2)
    return out


def road_burden_by_hex(
    hexagons: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
    municipality_boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    out = hexagons.copy()

    if roads.empty:
        out["road_buf_m2"] = 0.0
        out["road_buf_share"] = 0.0
        out["road_buf_pct"] = 0.0
        return out

    print(f"Buffering major roads by {major_road_buffer_m} m...")
    road_buffer = gpd.GeoDataFrame(geometry=roads.geometry.buffer(major_road_buffer_m), crs=target_crs)
    road_buffer["geometry"] = road_buffer.geometry.make_valid()
    road_buffer = gpd.clip(road_buffer, municipality_boundary)

    return area_share_by_hex(
        out,
        road_buffer,
        area_col_name="road_buf_m2",
        share_col_name="road_buf_share",
        pct_col_name="road_buf_pct",
    )


def minmax(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    mn, mx = s.min(), s.max()
    if np.isclose(mn, mx):
        return pd.Series(0.0, index=s.index)
    return (s - mn) / (mx - mn)


def classify_quintiles(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    ranks = pd.qcut(series, q=5, labels=False, duplicates="drop")
    rank_map = {b: i + 1 for i, b in enumerate(sorted(ranks.unique()))}
    rank = ranks.map(rank_map).astype(int)
    labels = ["Very low", "Low", "Moderate", "High", "Very high"]
    max_rank = int(rank.max())
    class_map = {i + 1: labels[i] for i in range(max_rank)}
    cls = rank.map(class_map)
    return rank, cls


def compute_environmental_indices(
    hexagons: gpd.GeoDataFrame,
    green_polygons: gpd.GeoDataFrame,
    blue_polygons: gpd.GeoDataFrame,
    builtup_polygons: gpd.GeoDataFrame,
    major_roads: gpd.GeoDataFrame,
    municipality_boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    out = hexagons.copy()

    out = area_share_by_hex(out, green_polygons, "green_m2", "green_share", "green_pct")
    out = area_share_by_hex(out, blue_polygons, "blue_m2", "blue_share", "blue_pct")
    out = area_share_by_hex(out, builtup_polygons, "builtup_m2", "builtup_share", "builtup_pct")
    out = road_burden_by_hex(out, major_roads, municipality_boundary)

    out["green_share_norm"] = minmax(out["green_share"])
    out["blue_share_norm"] = minmax(out["blue_share"])
    out["builtup_share_norm"] = minmax(out["builtup_share"])
    out["road_burden_norm"] = minmax(out["road_buf_share"])

    out["builtup_inverse_norm"] = 1 - out["builtup_share_norm"]
    out["road_burden_inverse_norm"] = 1 - out["road_burden_norm"]
    out["low_green_norm"] = 1 - out["green_share_norm"]
    out["low_blue_norm"] = 1 - out["blue_share_norm"]

    out["env_quality_raw"] = sum(out[col] * weight for col, weight in quality_weights.items())
    out["env_quality_score"] = (minmax(out["env_quality_raw"]) * 100).round(2)
    out["env_quality_rank"], out["env_quality_class"] = classify_quintiles(out["env_quality_score"])

    out["env_burden_raw"] = sum(out[col] * weight for col, weight in burden_weights.items())
    out["env_burden_score"] = (minmax(out["env_burden_raw"]) * 100).round(2)
    out["env_burden_rank"], out["env_burden_class"] = classify_quintiles(out["env_burden_score"])

    print("Environmental quality class distribution:")
    print(out["env_quality_class"].value_counts())
    print("Environmental burden class distribution:")
    print(out["env_burden_class"].value_counts())

    return out


def quality_colors() -> dict[str, str]:
    return {
        "Very low": "#ffffcc",
        "Low": "#c2e699",
        "Moderate": "#78c679",
        "High": "#31a354",
        "Very high": "#006837",
    }


def burden_colors() -> dict[str, str]:
    return {
        "Very low": "#ffffcc",
        "Low": "#fed976",
        "Moderate": "#fd8d3c",
        "High": "#f03b20",
        "Very high": "#bd0026",
    }


def plot_index_map(
    hex_output: gpd.GeoDataFrame,
    municipality: gpd.GeoDataFrame,
    municipality_boundary: gpd.GeoDataFrame,
    class_col: str,
    title: str,
    legend_title: str,
    colors: dict[str, str],
    output_path: Path,
) -> None:
    plot_gdf = hex_output.copy()
    plot_gdf["map_color"] = plot_gdf[class_col].map(colors)

    fig, ax = plt.subplots(figsize=(14, 10))
    plot_gdf.plot(ax=ax, color=plot_gdf["map_color"], edgecolor="white", linewidth=0.15, alpha=0.96)
    municipality.boundary.plot(ax=ax, linewidth=0.6, edgecolor="black", alpha=0.8)
    municipality_boundary.boundary.plot(ax=ax, linewidth=1.2, edgecolor="black")
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.set_axis_off()

    legend_order = [l for l in ["Very high", "High", "Moderate", "Low", "Very low"] if l in plot_gdf[class_col].unique()]
    ax.legend(
        handles=[mpatches.Patch(facecolor=colors[label], edgecolor="gray", label=label) for label in legend_order],
        title=legend_title,
        loc="lower left",
        frameon=True,
        fontsize=10,
        title_fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Map saved to: {output_path}")


def plot_osm_basemap_quality_map(
    hex_output: gpd.GeoDataFrame,
    municipality: gpd.GeoDataFrame,
    municipality_boundary: gpd.GeoDataFrame,
) -> None:
    import contextily as cx

    colors = quality_colors()
    hex_wm = hex_output.to_crs(web_mercator).copy()
    muni_wm = municipality.to_crs(web_mercator)
    boundary_wm = municipality_boundary.to_crs(web_mercator)
    hex_wm["map_color"] = hex_wm["env_quality_class"].map(colors)

    fig, ax = plt.subplots(figsize=(14, 10))
    boundary_wm.plot(ax=ax, facecolor="none", edgecolor="none")

    try:
        cx.add_basemap(ax, crs=hex_wm.crs, source=cx.providers.OpenStreetMap.Mapnik, zoom="auto")
    except Exception as e:
        print(f"OSM basemap warning ({e}); trying Carto Positron fallback.")
        cx.add_basemap(ax, crs=hex_wm.crs, source=cx.providers.CartoDB.Positron)

    hex_wm.plot(ax=ax, color=hex_wm["map_color"], edgecolor="white", linewidth=0.2, alpha=hex_alpha)
    muni_wm.boundary.plot(ax=ax, linewidth=0.5, edgecolor="#333333", alpha=0.85)
    boundary_wm.boundary.plot(ax=ax, linewidth=1.2, edgecolor="black")
    ax.set_title(
        "Odense Environmental Quality Index by 500 m Cell",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_axis_off()

    legend_order = [l for l in ["Very high", "High", "Moderate", "Low", "Very low"] if l in hex_wm["env_quality_class"].unique()]
    ax.legend(
        handles=[mpatches.Patch(facecolor=colors[label], edgecolor="gray", label=label, alpha=hex_alpha) for label in legend_order],
        title="Environmental quality",
        loc="lower left",
        frameon=True,
        fontsize=10,
        title_fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_quality_map_osm, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"OSM basemap map saved to: {output_quality_map_osm}")


def validate_environmental_index(
    hex_output: gpd.GeoDataFrame,
    municipality_boundary: gpd.GeoDataFrame,
    green: gpd.GeoDataFrame,
    blue: gpd.GeoDataFrame,
    builtup: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
) -> list[str]:
    lines = ["Environmental index validation", "=" * 45]
    lines.append(f"include_agriculture_in_green: {include_agriculture_in_green}")
    lines.append(f"Number of 500 m cells: {len(hex_output)}")
    lines.append(f"Municipality area: {municipality_boundary.geometry.area.iloc[0] / 1_000_000:.2f} km²")
    lines.append("")
    lines.append("Input feature counts:")
    lines.append(f"  OSM green polygon features: {len(green)}")
    lines.append(f"  OSM blue polygon features: {len(blue)}")
    lines.append(f"  OSM built-up polygon features: {len(builtup)}")
    lines.append(f"  OSM major-road features: {len(roads)}")
    lines.append("")
    for col in ["green_pct", "blue_pct", "builtup_pct", "road_buf_pct"]:
        lines.append(
            f"{col}: min={hex_output[col].min():.2f}, mean={hex_output[col].mean():.2f}, max={hex_output[col].max():.2f}"
        )
    lines.append("")
    lines.append(f"Quality weights sum: {sum(quality_weights.values()):.2f}")
    lines.append(f"Burden weights sum: {sum(burden_weights.values()):.2f}")
    lines.append(f"Environmental quality score range: {hex_output['env_quality_score'].min():.2f} – {hex_output['env_quality_score'].max():.2f}")
    lines.append(f"Environmental burden score range: {hex_output['env_burden_score'].min():.2f} – {hex_output['env_burden_score'].max():.2f}")
    return lines


def save_outputs(
    hexagons: gpd.GeoDataFrame,
    municipality: gpd.GeoDataFrame,
    municipality_boundary: gpd.GeoDataFrame,
    green: gpd.GeoDataFrame,
    blue: gpd.GeoDataFrame,
    builtup: gpd.GeoDataFrame,
    roads: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    output_cols = [
        "hex_id", "cell_area_m2",
        "green_m2", "green_share", "green_pct",
        "blue_m2", "blue_share", "blue_pct",
        "builtup_m2", "builtup_share", "builtup_pct",
        "road_buf_m2", "road_buf_share", "road_buf_pct",
        "env_quality_raw", "env_quality_score", "env_quality_class", "env_quality_rank",
        "env_burden_raw", "env_burden_score", "env_burden_class", "env_burden_rank",
        "geometry",
    ]
    hex_output = hexagons[output_cols].copy()

    hex_output.to_file(output_gpkg, layer="environmental_index_500m_hexagons", driver="GPKG")
    municipality.to_file(output_gpkg, layer="odense_municipality_districts", driver="GPKG")
    if len(green):
        green.to_file(output_gpkg, layer="osm_green_features", driver="GPKG")
    if len(blue):
        blue.to_file(output_gpkg, layer="osm_blue_features", driver="GPKG")
    if len(builtup):
        builtup.to_file(output_gpkg, layer="osm_builtup_features", driver="GPKG")
    if len(roads):
        roads.to_file(output_gpkg, layer="osm_major_roads", driver="GPKG")
    print(f"GeoPackage saved to: {output_gpkg}")

    shp_output = hex_output.rename(
        columns={
            "cell_area_m2": "cell_m2",
            "green_share": "green_sh",
            "builtup_share": "built_sh",
            "road_buf_share": "road_sh",
            "road_buf_pct": "road_pct",
            "env_quality_score": "qual_scr",
            "env_quality_class": "qual_cls",
            "env_quality_rank": "qual_rnk",
            "env_burden_score": "burd_scr",
            "env_burden_class": "burd_cls",
            "env_burden_rank": "burd_rnk",
        }
    )[
        [
            "hex_id", "cell_m2", "green_pct", "blue_pct", "builtup_pct", "road_pct",
            "qual_scr", "qual_cls", "qual_rnk",
            "burd_scr", "burd_cls", "burd_rnk", "geometry",
        ]
    ]
    shp_path = output_shp_dir / "odense_environmental_index_500m.shp"
    shp_output.to_file(shp_path, driver="ESRI Shapefile", encoding="UTF-8")
    print(f"Shapefile saved to: {shp_path}")

    plot_index_map(
        hex_output, municipality, municipality_boundary,
        class_col="env_quality_class",
        title="Odense Environmental Quality Index by 500 m Cell",
        legend_title="Environmental quality",
        colors=quality_colors(),
        output_path=output_quality_map,
    )
    plot_index_map(
        hex_output, municipality, municipality_boundary,
        class_col="env_burden_class",
        title="Odense Environmental Burden Index by 500 m Cell",
        legend_title="Environmental burden",
        colors=burden_colors(),
        output_path=output_burden_map,
    )
    plot_osm_basemap_quality_map(hex_output, municipality, municipality_boundary)

    validation_report.write_text(
        "\n".join(validate_environmental_index(hex_output, municipality_boundary, green, blue, builtup, roads)),
        encoding="utf-8",
    )
    print(f"Validation report saved to: {validation_report}")

    return hex_output


def map_only_from_gpkg() -> None:
    if not output_gpkg.exists():
        raise FileNotFoundError(f"No saved results at {output_gpkg}. Run the full pipeline first.")
    municipality = gpd.read_file(output_gpkg, layer="odense_municipality_districts").to_crs(target_crs)
    municipality_boundary = gpd.GeoDataFrame(geometry=[municipality.geometry.union_all()], crs=target_crs)
    hex_output = gpd.read_file(output_gpkg, layer="environmental_index_500m_hexagons").to_crs(target_crs)
    plot_index_map(
        hex_output, municipality, municipality_boundary,
        class_col="env_quality_class",
        title="Odense Environmental Quality Index by 500 m Cell",
        legend_title="Environmental quality",
        colors=quality_colors(),
        output_path=output_quality_map,
    )
    plot_index_map(
        hex_output, municipality, municipality_boundary,
        class_col="env_burden_class",
        title="Odense Environmental Burden Index by 500 m Cell",
        legend_title="Environmental burden",
        colors=burden_colors(),
        output_path=output_burden_map,
    )
    plot_osm_basemap_quality_map(hex_output, municipality, municipality_boundary)


def main() -> None:
    import sys

    if "--map-only" in sys.argv:
        map_only_from_gpkg()
        print("Finished (map-only).")
        return

    municipality, hexagons, municipality_boundary = load_inputs()
    green_query_tags = merge_tags(green_tags, agriculture_tags if include_agriculture_in_green else None)
    green = download_osm_polygons(municipality_boundary, green_query_tags, "green")
    blue = download_osm_polygons(municipality_boundary, blue_tags, "blue")
    builtup = download_osm_polygons(municipality_boundary, builtup_tags, "builtup")
    roads = download_osm_major_roads(municipality_boundary)
    hexagons = compute_environmental_indices(hexagons, green, blue, builtup, roads, municipality_boundary)
    save_outputs(hexagons, municipality, municipality_boundary, green, blue, builtup, roads)
    print("Finished.")


if __name__ == "__main__":
    main()
