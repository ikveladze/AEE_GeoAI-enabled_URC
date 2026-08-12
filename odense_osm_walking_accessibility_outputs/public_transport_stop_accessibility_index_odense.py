# ============================================================
# OSM-based Public Transport Stop Accessibility Index
# Odense Municipality, 500 m hexagonal cells
# Output: GeoPackage + Shapefile + maps + validation report
# CRS: EPSG:25832
# ============================================================
#
# IMPORTANT:
# This model measures walking accessibility from each 500 m hexagon to
# OSM-mapped public transport stops/stations. It does NOT model in-vehicle
# public transport travel time, frequency, waiting time, or transfers.
# For that, GTFS schedule data would be required.
# ============================================================

from __future__ import annotations

import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
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

output_dir = SCRIPT_DIR / "odense_osm_public_transport_accessibility_outputs"
output_dir.mkdir(exist_ok=True)

output_gpkg = output_dir / "odense_public_transport_stop_accessibility_500m.gpkg"
output_shp_dir = output_dir / "odense_public_transport_stop_accessibility_500m_shp"
output_shp_dir.mkdir(exist_ok=True)

output_map_10 = output_dir / "odense_public_transport_stop_accessibility_500m_map_10min.png"
output_map_20 = output_dir / "odense_public_transport_stop_accessibility_500m_map_20min.png"
output_map_osm_20 = output_dir / "odense_public_transport_stop_accessibility_500m_map_20min_osm.png"
validation_report = output_dir / "public_transport_stop_accessibility_validation.txt"

hex_alpha = 0.62

# Public transport stop accessibility is modelled as walking access to stops.
walking_speed_m_per_s = 1.34  # ≈ 4.8 km/h

# Required thresholds: 10–20 min walking access to public transport stops.
threshold_minutes = [10, 20]
max_access_distances_m = {
    minutes: walking_speed_m_per_s * 60 * minutes
    for minutes in threshold_minutes
}

# Distance decay for walking access to public transport stops.
decay_beta_m = 1000

# Snapping diagnostics.
max_stop_snap_distance_m = 200
max_origin_snap_warning_m = 250

# OSM public transport categories.
pt_categories = {
    "bus_stop": {
        "tags": {"highway": "bus_stop"},
        "weight": 0.55,
    },
    "pt_platform_stop": {
        "tags": {"public_transport": ["platform", "stop_position"]},
        "weight": 0.25,
    },
    "rail_tram_station": {
        "tags": {"railway": ["station", "halt", "tram_stop"]},
        "weight": 0.15,
    },
    "transport_hub": {
        "tags": {
            "amenity": "bus_station",
            "public_transport": "station",
        },
        "weight": 0.05,
    },
}

weight_sum = sum(v["weight"] for v in pt_categories.values())
if not np.isclose(weight_sum, 1.0):
    raise ValueError(f"PT category weights must sum to 1. Current sum = {weight_sum}")


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
    hexagons["hex_id"] = [f"OD_PT_{i + 1:04d}" for i in range(len(hexagons))]

    print(f"Number of clipped 500 m hexagons: {len(hexagons)}")
    return municipality, hexagons, municipality_boundary


def download_access_network(municipality_boundary: gpd.GeoDataFrame) -> nx.MultiGraph:
    boundary_wgs84 = municipality_boundary.to_crs("EPSG:4326")
    boundary_polygon_wgs84 = boundary_wgs84.geometry.iloc[0]

    print("Downloading OSM walking network for public-transport stop access...")
    G = ox.graph_from_polygon(
        boundary_polygon_wgs84,
        network_type="walk",
        simplify=True,
        retain_all=True,
        truncate_by_edge=True,
    )

    G = ox.project_graph(G, to_crs=target_crs)
    G = ox.convert.to_undirected(G)

    for _, _, data in G.edges(data=True):
        if "length" not in data:
            data["length"] = 0

    print(f"Network nodes: {len(G.nodes)}")
    print(f"Network edges: {len(G.edges)} (undirected)")
    return G


def _geom_to_point(geom):
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom.representative_point()
    if geom.geom_type in ("LineString", "MultiLineString"):
        return geom.interpolate(0.5, normalized=True)
    return geom


def download_osm_pt_features(category_name: str, tags: dict, polygon_wgs84) -> gpd.GeoDataFrame:
    print(f"Downloading OSM public transport features for: {category_name}")

    empty = gpd.GeoDataFrame(
        columns=["category", "geometry"],
        geometry="geometry",
        crs=target_crs,
    )

    try:
        features = ox.features_from_polygon(polygon_wgs84, tags=tags)
    except Exception as e:
        print(f"Warning: no features downloaded for {category_name}: {e}")
        return empty

    if features.empty:
        print(f"No OSM features found for {category_name}")
        return empty

    features = features.reset_index()
    features = features[features.geometry.notnull()].copy()
    features = features.to_crs(target_crs)

    features["geometry"] = features.geometry.apply(_geom_to_point)
    features = features[features.geometry.geom_type == "Point"].copy()

    features["category"] = category_name
    features["x"] = features.geometry.x.round(2)
    features["y"] = features.geometry.y.round(2)
    features = features.drop_duplicates(subset=["category", "x", "y"])

    return features[["category", "geometry"]]


def download_all_pt_stops(boundary_polygon_wgs84) -> gpd.GeoDataFrame:
    frames = [
        download_osm_pt_features(name, cfg["tags"], boundary_polygon_wgs84)
        for name, cfg in pt_categories.items()
    ]

    stops = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=target_crs,
    )

    if len(stops):
        stops["x"] = stops.geometry.x.round(2)
        stops["y"] = stops.geometry.y.round(2)
        stops_unique = stops.drop_duplicates(subset=["x", "y"]).copy()
    else:
        stops_unique = stops.copy()

    print(f"Total OSM PT records before cross-category de-duplication: {len(stops)}")
    print(f"Total unique OSM PT stop/platform/station points: {len(stops_unique)}")
    if len(stops):
        print(stops["category"].value_counts())

    return stops_unique[["category", "geometry"]]


def calculate_category_accessibility(
    G: nx.MultiGraph,
    origin_nodes,
    stop_nodes,
    cutoff_m: float,
    decay_beta_m: float,
) -> pd.DataFrame:
    stop_nodes = list(set(stop_nodes))

    if len(stop_nodes) == 0:
        n = len(origin_nodes)
        return pd.DataFrame({
            "nearest_distance_m": [np.nan] * n,
            "accessibility_score": [0.0] * n,
        })

    distances = nx.multi_source_dijkstra_path_length(
        G,
        sources=stop_nodes,
        cutoff=cutoff_m,
        weight="length",
    )

    nearest_distances = []
    scores = []

    for node in origin_nodes:
        d = distances.get(node, np.nan)
        if pd.isna(d):
            nearest_distances.append(np.nan)
            scores.append(0.0)
        else:
            nearest_distances.append(d)
            scores.append(np.exp(-d / decay_beta_m))

    return pd.DataFrame({
        "nearest_distance_m": nearest_distances,
        "accessibility_score": scores,
    })


def compute_pt_stop_accessibility_index(
    G: nx.MultiGraph,
    hexagons: gpd.GeoDataFrame,
    stops: gpd.GeoDataFrame,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    hexagons = hexagons.copy()
    hexagons["origin_geom"] = hexagons.geometry.representative_point()

    hexagons["origin_node"], hexagons["origin_snap_m"] = ox.distance.nearest_nodes(
        G,
        X=hexagons["origin_geom"].x.values,
        Y=hexagons["origin_geom"].y.values,
        return_dist=True,
    )

    n_far_origins = int((hexagons["origin_snap_m"] > max_origin_snap_warning_m).sum())
    if n_far_origins:
        print(
            f"Warning: {n_far_origins} hexagon origins are farther than "
            f"{max_origin_snap_warning_m} m from the walking access network."
        )

    if len(stops):
        stops = stops.copy()
        stops["stop_node"], stops["stop_snap_m"] = ox.distance.nearest_nodes(
            G,
            X=stops.geometry.x.values,
            Y=stops.geometry.y.values,
            return_dist=True,
        )

        before = len(stops)
        stops = stops[stops["stop_snap_m"] <= max_stop_snap_distance_m].copy()
        removed = before - len(stops)
        if removed:
            print(
                f"Removed {removed} OSM public-transport stop features because they were farther than "
                f"{max_stop_snap_distance_m} m from the walking network."
            )
    else:
        stops = gpd.GeoDataFrame(
            columns=["category", "geometry", "stop_node", "stop_snap_m"],
            geometry="geometry",
            crs=target_crs,
        )

    print("Public transport stop access thresholds:")
    for minutes, dist_m in max_access_distances_m.items():
        print(f"  {minutes} min = {dist_m:.0f} m at {walking_speed_m_per_s:.2f} m/s")

    for minutes, cutoff_m in max_access_distances_m.items():
        print(f"\nCalculating {minutes}-minute PT stop accessibility...")

        all_result = calculate_category_accessibility(
            G=G,
            origin_nodes=hexagons["origin_node"].values,
            stop_nodes=stops["stop_node"].values,
            cutoff_m=cutoff_m,
            decay_beta_m=decay_beta_m,
        )
        hexagons[f"dist_all_pt_{minutes}m"] = all_result["nearest_distance_m"].round(1)
        hexagons[f"score_all_pt_{minutes}m"] = all_result["accessibility_score"]

        for category_name in pt_categories:
            print(f"  Category: {category_name}")
            category_stops = stops[stops["category"] == category_name]

            result = calculate_category_accessibility(
                G=G,
                origin_nodes=hexagons["origin_node"].values,
                stop_nodes=category_stops["stop_node"].values,
                cutoff_m=cutoff_m,
                decay_beta_m=decay_beta_m,
            )

            hexagons[f"dist_{category_name}_{minutes}m"] = result["nearest_distance_m"].round(1)
            hexagons[f"score_{category_name}_{minutes}m"] = result["accessibility_score"]

        raw_col = f"pt_access_raw_{minutes}m"
        score_col = f"pt_access_score_{minutes}m"
        class_col = f"pt_access_class_{minutes}m"
        rank_col = f"pt_access_rank_{minutes}m"

        category_composite = sum(
            hexagons[f"score_{name}_{minutes}m"] * cfg["weight"]
            for name, cfg in pt_categories.items()
        )

        # Final PT-stop accessibility combines nearest all-stop access and
        # weighted category access. This avoids over-penalising areas where only
        # one OSM stop tagging scheme is dominant.
        hexagons[raw_col] = 0.5 * hexagons[f"score_all_pt_{minutes}m"] + 0.5 * category_composite

        raw_min = hexagons[raw_col].min()
        raw_max = hexagons[raw_col].max()

        if np.isclose(raw_min, raw_max):
            hexagons[score_col] = 0.0
        else:
            hexagons[score_col] = ((hexagons[raw_col] - raw_min) / (raw_max - raw_min) * 100).round(2)

        beyond_cutoff = hexagons[raw_col] <= 0
        hexagons[class_col] = f"Beyond {minutes} min cutoff"
        hexagons[rank_col] = 0

        active = ~beyond_cutoff
        if active.any():
            ranks = pd.qcut(
                hexagons.loc[active, score_col],
                q=5,
                labels=False,
                duplicates="drop",
            )

            class_labels = ["Very low", "Low", "Moderate", "High", "Very high"]
            rank_map = {b: i + 1 for i, b in enumerate(sorted(ranks.unique()))}
            hexagons.loc[active, rank_col] = ranks.map(rank_map).astype(int)

            max_rank = int(hexagons.loc[active, rank_col].max())
            class_map = {i + 1: class_labels[i] for i in range(max_rank)}
            hexagons.loc[active, class_col] = hexagons.loc[active, rank_col].map(class_map)

        print(hexagons[class_col].value_counts())

    return hexagons, stops


def class_colors(minutes: int) -> dict[str, str]:
    return {
        f"Beyond {minutes} min cutoff": "#e0e0e0",
        "Very low": "#fff7bc",
        "Low": "#fec44f",
        "Moderate": "#fe9929",
        "High": "#ec7014",
        "Very high": "#cc0000",
    }


def save_outputs(
    hexagons: gpd.GeoDataFrame,
    municipality: gpd.GeoDataFrame,
    stops: gpd.GeoDataFrame,
    municipality_boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    output_cols = [
        "hex_id",
        "origin_snap_m",
        "pt_access_score_10m",
        "pt_access_class_10m",
        "pt_access_rank_10m",
        "pt_access_raw_10m",
        "pt_access_score_20m",
        "pt_access_class_20m",
        "pt_access_rank_20m",
        "pt_access_raw_20m",
        *[c for c in hexagons.columns if c.startswith("dist_")],
        *[c for c in hexagons.columns if c.startswith("score_")],
        "geometry",
    ]

    hex_output = hexagons[output_cols].copy()

    if len(stops):
        stops.to_file(output_gpkg, layer="osm_public_transport_stops", driver="GPKG")

    hex_output.to_file(
        output_gpkg,
        layer="pt_stop_accessibility_500m_hexagons",
        driver="GPKG",
    )

    municipality.to_file(
        output_gpkg,
        layer="odense_municipality_districts",
        driver="GPKG",
    )

    print(f"GeoPackage saved to: {output_gpkg}")

    shp_output = hex_output.rename(
        columns={
            "origin_snap_m": "orig_snap",
            "pt_access_score_10m": "pt10_scr",
            "pt_access_class_10m": "pt10_cls",
            "pt_access_rank_10m": "pt10_rnk",
            "pt_access_raw_10m": "pt10_raw",
            "pt_access_score_20m": "pt20_scr",
            "pt_access_class_20m": "pt20_cls",
            "pt_access_rank_20m": "pt20_rnk",
            "pt_access_raw_20m": "pt20_raw",
        }
    )[
        [
            "hex_id",
            "orig_snap",
            "pt10_scr",
            "pt10_cls",
            "pt10_rnk",
            "pt10_raw",
            "pt20_scr",
            "pt20_cls",
            "pt20_rnk",
            "pt20_raw",
            "geometry",
        ]
    ]

    shp_path = output_shp_dir / "odense_public_transport_stop_accessibility_500m.shp"
    shp_output.to_file(shp_path, driver="ESRI Shapefile", encoding="UTF-8")
    print(f"Shapefile saved to: {shp_path}")

    plot_static_map(hex_output, municipality, municipality_boundary, minutes=10, output_path=output_map_10)
    plot_static_map(hex_output, municipality, municipality_boundary, minutes=20, output_path=output_map_20)
    plot_osm_basemap_map(hex_output, municipality, municipality_boundary, minutes=20)

    report_lines = validate_pt_accessibility_index(hex_output)
    validation_report.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Validation report saved to: {validation_report}")

    summary_frames = []
    for minutes in threshold_minutes:
        class_col = f"pt_access_class_{minutes}m"
        score_col = f"pt_access_score_{minutes}m"
        summary = (
            hex_output.groupby(class_col)
            .agg(
                n_cells=("hex_id", "count"),
                mean_score=(score_col, "mean"),
                min_score=(score_col, "min"),
                max_score=(score_col, "max"),
            )
            .reset_index()
            .rename(columns={class_col: "class"})
        )
        summary["threshold_min"] = minutes
        summary_frames.append(summary)

    summary_all = pd.concat(summary_frames, ignore_index=True)
    summary_all.to_csv(output_dir / "public_transport_stop_accessibility_class_summary.csv", index=False)
    print(summary_all)

    return hex_output


def plot_static_map(
    hex_output: gpd.GeoDataFrame,
    municipality: gpd.GeoDataFrame,
    municipality_boundary: gpd.GeoDataFrame,
    minutes: int,
    output_path: Path,
) -> None:
    colors = class_colors(minutes)
    class_col = f"pt_access_class_{minutes}m"

    plot_gdf = hex_output.copy()
    plot_gdf["map_color"] = plot_gdf[class_col].map(colors)

    fig, ax = plt.subplots(figsize=(14, 10))

    plot_gdf.plot(
        ax=ax,
        color=plot_gdf["map_color"],
        edgecolor="white",
        linewidth=0.15,
        alpha=0.95,
    )
    municipality.boundary.plot(ax=ax, linewidth=0.6, edgecolor="black", alpha=0.8)
    municipality_boundary.boundary.plot(ax=ax, linewidth=1.2, edgecolor="black")

    ax.set_title(
        f"Odense Public Transport Stop Accessibility ({minutes}-min OSM walking network)",
        fontsize=18,
        fontweight="bold",
    )
    ax.set_axis_off()

    legend_order = ["Very high", "High", "Moderate", "Low", "Very low", f"Beyond {minutes} min cutoff"]
    legend_order = [label for label in legend_order if label in plot_gdf[class_col].unique()]
    legend_patches = [
        mpatches.Patch(facecolor=colors[label], edgecolor="gray", label=label)
        for label in legend_order
    ]

    ax.legend(
        handles=legend_patches,
        title=f"PT stop accessibility ({minutes} min)",
        loc="lower left",
        frameon=True,
        fontsize=10,
        title_fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Map saved to: {output_path}")


def plot_osm_basemap_map(
    hex_output: gpd.GeoDataFrame,
    municipality: gpd.GeoDataFrame,
    municipality_boundary: gpd.GeoDataFrame,
    minutes: int = 20,
) -> None:
    try:
        import contextily as cx
    except ImportError:
        print("contextily is not installed; skipping OSM basemap map.")
        return

    colors = class_colors(minutes)
    class_col = f"pt_access_class_{minutes}m"

    hex_wm = hex_output.to_crs(web_mercator).copy()
    muni_wm = municipality.to_crs(web_mercator)
    boundary_wm = municipality_boundary.to_crs(web_mercator)

    hex_wm["map_color"] = hex_wm[class_col].map(colors)

    fig, ax = plt.subplots(figsize=(14, 10))
    boundary_wm.plot(ax=ax, facecolor="none", edgecolor="none")

    try:
        cx.add_basemap(
            ax,
            crs=hex_wm.crs,
            source=cx.providers.OpenStreetMap.Mapnik,
            zoom="auto",
        )
    except Exception as e:
        print(f"OSM basemap warning ({e}); trying Carto Positron fallback.")
        cx.add_basemap(ax, crs=hex_wm.crs, source=cx.providers.CartoDB.Positron)

    hex_wm.plot(
        ax=ax,
        color=hex_wm["map_color"],
        edgecolor="white",
        linewidth=0.2,
        alpha=hex_alpha,
    )
    muni_wm.boundary.plot(ax=ax, linewidth=0.5, edgecolor="#333333", alpha=0.85)
    boundary_wm.boundary.plot(ax=ax, linewidth=1.2, edgecolor="black")

    ax.set_title(
        # Retain the original two-line title height so tight_layout preserves
        # the established map composition, while omitting the former subtitle.
        "Odense Public Transport Stop Accessibility\n\u00A0",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_axis_off()

    legend_order = ["Very high", "High", "Moderate", "Low", "Very low", f"Beyond {minutes} min cutoff"]
    legend_order = [label for label in legend_order if label in hex_wm[class_col].unique()]
    legend_patches = [
        mpatches.Patch(facecolor=colors[label], edgecolor="gray", label=label, alpha=hex_alpha)
        for label in legend_order
    ]

    ax.legend(
        handles=legend_patches,
        title=f"PT stop accessibility",
        loc="lower left",
        frameon=True,
        fontsize=10,
        title_fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(output_map_osm_20, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"OSM basemap map saved to: {output_map_osm_20}")


def validate_pt_accessibility_index(hexagons: gpd.GeoDataFrame) -> list[str]:
    lines = ["Public transport stop accessibility validation", "=" * 55]
    weights = {k: v["weight"] for k, v in pt_categories.items()}
    lines.append(f"PT category weights sum: {sum(weights.values()):.4f} (expect 1.0)")
    lines.append(f"Access speed: {walking_speed_m_per_s:.2f} m/s ({walking_speed_m_per_s*3.6:.1f} km/h)")
    lines.append(f"Distance decay beta: {decay_beta_m:.0f} m")
    lines.append("Interpretation: walking access to OSM public transport stops/stations, not timetable-based PT travel time.")

    for minutes in threshold_minutes:
        lines.append("")
        lines.append(f"{minutes}-minute threshold")
        lines.append("-" * 30)
        lines.append(f"Maximum walking network distance: {max_access_distances_m[minutes]:.0f} m")

        score_cols = [f"score_{k}_{minutes}m" for k in weights]
        raw_col = f"pt_access_raw_{minutes}m"
        class_col = f"pt_access_class_{minutes}m"
        score_col = f"pt_access_score_{minutes}m"

        category_composite = sum(
            hexagons[c] * weights[k]
            for k, c in zip(weights, score_cols, strict=True)
        )
        recomputed = 0.5 * hexagons[f"score_all_pt_{minutes}m"] + 0.5 * category_composite
        diff = (recomputed - hexagons[raw_col]).abs().max()
        lines.append(f"Composite raw score max deviation: {diff:.6f} (expect 0)")

        for col in [f"score_all_pt_{minutes}m", *score_cols]:
            bad = ((hexagons[col] < 0) | (hexagons[col] > 1)).sum()
            lines.append(f"  {col}: values outside [0,1] = {bad}")

        beyond = (hexagons[raw_col] <= 0).sum()
        lines.append(f"Hexagons beyond {minutes} min cutoff from all selected PT stop features: {beyond}")
        lines.append(f"Score range (0–100): {hexagons[score_col].min():.2f} – {hexagons[score_col].max():.2f}")
        lines.append("Class distribution:")
        for cls, n in hexagons[class_col].value_counts().items():
            sub = hexagons[hexagons[class_col] == cls]
            lines.append(f"  {cls}: {n} cells, mean score {sub[score_col].mean():.1f}")

    lines.append("")
    lines.append("Method notes:")
    lines.append("- Access network: OSM walk network, undirected, metric CRS EPSG:25832")
    lines.append("- PT data: OSM highway=bus_stop, public_transport=platform/stop_position/station, railway=station/halt/tram_stop, amenity=bus_station")
    lines.append("- Access: multi-source Dijkstra from stop/station nodes")
    lines.append("- Thresholds: 10 and 20 min walking access to stops, using 1.34 m/s")
    lines.append("- Decay: exp(-d/1000)")
    lines.append("- Index: 50% nearest all-stop proximity + 50% weighted PT-category composite, min-max scaled to 0–100")
    lines.append("- Classes: cutoff class = raw 0; others = quintiles among accessible cells")
    lines.append("- Limitation: this is stop accessibility, not schedule/frequency/transfer-based public transport accessibility")
    return lines


def map_only_from_gpkg() -> None:
    if not output_gpkg.exists():
        raise FileNotFoundError(f"No saved results at {output_gpkg}. Run the full pipeline first.")

    municipality = gpd.read_file(output_gpkg, layer="odense_municipality_districts").to_crs(target_crs)
    municipality_boundary = gpd.GeoDataFrame(
        geometry=[municipality.geometry.union_all()],
        crs=target_crs,
    )
    hex_output = gpd.read_file(output_gpkg, layer="pt_stop_accessibility_500m_hexagons").to_crs(target_crs)

    plot_static_map(hex_output, municipality, municipality_boundary, minutes=10, output_path=output_map_10)
    plot_static_map(hex_output, municipality, municipality_boundary, minutes=20, output_path=output_map_20)
    plot_osm_basemap_map(hex_output, municipality, municipality_boundary, minutes=20)

    report_lines = validate_pt_accessibility_index(hex_output)
    validation_report.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Validation report: {validation_report}")


def main() -> None:
    import sys

    if "--map-only" in sys.argv:
        map_only_from_gpkg()
        print("Finished (map-only).")
        return

    municipality, hexagons, municipality_boundary = load_inputs()
    boundary_wgs84 = municipality_boundary.to_crs("EPSG:4326").geometry.iloc[0]

    G = download_access_network(municipality_boundary)
    stops = download_all_pt_stops(boundary_wgs84)

    hexagons, stops = compute_pt_stop_accessibility_index(G, hexagons, stops)
    save_outputs(hexagons, municipality, stops, municipality_boundary)

    print("Finished.")


if __name__ == "__main__":
    main()
