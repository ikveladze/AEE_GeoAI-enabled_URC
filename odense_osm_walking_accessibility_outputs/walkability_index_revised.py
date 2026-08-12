# ============================================================
# OSM-based Walking Accessibility to Services (Walkability Index)
# Odense Municipality, 500 m hexagonal cells
# Output: GeoPackage + Shapefile + map
# CRS: EPSG:25832
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

# ------------------------------------------------------------
# 1. INPUT FILES (relative to this script)
# ------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
# Script may live in project root or in odense_osm_walking_accessibility_outputs/
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


def _resolve_municipality_path() -> tuple[Path, str | None]:
    for path in MUNICIPALITY_CANDIDATES:
        if path.exists():
            layer = municipality_layer_shp if path.suffix.lower() == ".shp" else municipality_layer
            return path, layer
    raise FileNotFoundError(
        "Municipality boundary not found. Expected one of:\n"
        + "\n".join(f"  - {p}" for p in MUNICIPALITY_CANDIDATES)
    )

output_dir = SCRIPT_DIR
output_dir.mkdir(exist_ok=True)

output_gpkg = output_dir / "odense_walking_accessibility_services_500m.gpkg"
output_shp_dir = output_dir / "odense_walking_accessibility_services_500m_shp"
output_shp_dir.mkdir(exist_ok=True)

output_map = output_dir / "odense_walking_accessibility_services_500m_map.png"
output_map_osm = output_dir / "odense_walking_accessibility_services_500m_map_osm.png"
validation_report = output_dir / "walkability_index_validation.txt"

target_crs = "EPSG:25832"
web_mercator = "EPSG:3857"
hex_alpha = 0.58  # transparency so OSM basemap shows through

# ------------------------------------------------------------
# 2. LOAD MUNICIPALITY AND HEXAGONS
# ------------------------------------------------------------


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
    hexagons["hex_id"] = [f"OD_WALK_{i + 1:04d}" for i in range(len(hexagons))]

    print(f"Number of clipped 500 m hexagons: {len(hexagons)}")
    return municipality, hexagons, municipality_boundary


# ------------------------------------------------------------
# 3. OSM WALKING NETWORK
# ------------------------------------------------------------


def download_walk_network(municipality_boundary: gpd.GeoDataFrame) -> nx.MultiDiGraph:
    boundary_wgs84 = municipality_boundary.to_crs("EPSG:4326")
    boundary_polygon_wgs84 = boundary_wgs84.geometry.iloc[0]

    print("Downloading OSM walking network...")
    G = ox.graph_from_polygon(
        boundary_polygon_wgs84,
        network_type="walk",
        simplify=True,
        # Keep all disconnected pedestrian components. Using False can remove small settlements
        # and artificially create inaccessible cells in peripheral areas.
        retain_all=True,
        truncate_by_edge=True,
    )
    G = ox.project_graph(G, to_crs=target_crs)
    # Walking is bidirectional; undirected graph avoids one-way street bias in distances.
    G = ox.convert.to_undirected(G)

    for _, _, data in G.edges(data=True):
        if "length" not in data:
            data["length"] = 0

    print(f"Network nodes: {len(G.nodes)}")
    print(f"Network edges: {len(G.edges)} (undirected)")
    return G


# ------------------------------------------------------------
# 4. SERVICE CATEGORIES
# ------------------------------------------------------------

service_categories = {
    "food_retail": {
        "tags": {
            "shop": [
                "supermarket",
                "convenience",
                "bakery",
                "greengrocer",
                "butcher",
                "deli",
            ]
        },
        "weight": 0.25,
    },
    "health": {
        "tags": {
            "amenity": [
                "pharmacy",
                "doctors",
                "clinic",
                "dentist",
                "hospital",
            ]
        },
        "weight": 0.20,
    },
    "education_childcare": {
        "tags": {
            "amenity": [
                "school",
                "kindergarten",
                "childcare",
                "college",
                "university",
            ]
        },
        "weight": 0.15,
    },
    "food_drink": {
        "tags": {
            "amenity": [
                "restaurant",
                "cafe",
                "fast_food",
                "pub",
            ]
        },
        "weight": 0.15,
    },
    "civic_daily_services": {
        "tags": {
            "amenity": [
                "library",
                "post_office",
                "bank",
                "atm",
                "townhall",
                "community_centre",
            ]
        },
        "weight": 0.15,
    },
    "recreation": {
        "tags": {
            "leisure": [
                "park",
                "playground",
                "sports_centre",
                "fitness_centre",
            ],
            "amenity": [
                "theatre",
                "cinema",
                "arts_centre",
            ],
        },
        "weight": 0.10,
    },
}

weight_sum = sum(v["weight"] for v in service_categories.values())
if not np.isclose(weight_sum, 1.0):
    raise ValueError(f"Service category weights must sum to 1. Current sum = {weight_sum}")

# Walking model parameters
# 1.34 m/s ≈ 4.8 km/h, a commonly used average adult walking speed.
walking_speed_m_per_s = 1.34
max_walk_minutes = 15
max_walk_distance_m = walking_speed_m_per_s * 60 * max_walk_minutes
decay_beta = 700

# Snapping diagnostics. Service points farther than this from the pedestrian
# graph are excluded because they are likely detached from the walkable network.
max_service_snap_distance_m = 200
max_origin_snap_warning_m = 250


# ------------------------------------------------------------
# 5. OSM SERVICE FEATURES
# ------------------------------------------------------------


def _geom_to_point(geom):
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom.representative_point()
    if geom.geom_type in ("LineString", "MultiLineString"):
        return geom.interpolate(0.5, normalized=True)
    return geom


def download_osm_features(
    category_name: str,
    tags: dict,
    polygon_wgs84,
) -> gpd.GeoDataFrame:
    print(f"Downloading OSM features for: {category_name}")

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


def download_all_services(boundary_polygon_wgs84) -> gpd.GeoDataFrame:
    frames = [
        download_osm_features(name, cfg["tags"], boundary_polygon_wgs84)
        for name, cfg in service_categories.items()
    ]
    services = gpd.GeoDataFrame(
        pd.concat(frames, ignore_index=True),
        geometry="geometry",
        crs=target_crs,
    )
    print(f"Total OSM service points: {len(services)}")
    if len(services):
        print(services["category"].value_counts())
    return services


# ------------------------------------------------------------
# 6. NETWORK ACCESSIBILITY
# ------------------------------------------------------------


def calculate_category_accessibility(
    G: nx.MultiDiGraph,
    origin_nodes,
    service_nodes,
    cutoff_m: float,
    decay_beta: float,
) -> pd.DataFrame:
    service_nodes = list(set(service_nodes))

    if len(service_nodes) == 0:
        n = len(origin_nodes)
        return pd.DataFrame({
            "nearest_distance_m": [np.nan] * n,
            "accessibility_score": [0.0] * n,
        })

    distances = nx.multi_source_dijkstra_path_length(
        G,
        sources=service_nodes,
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
            scores.append(np.exp(-d / decay_beta))

    return pd.DataFrame({
        "nearest_distance_m": nearest_distances,
        "accessibility_score": scores,
    })


def compute_walkability_index(
    G: nx.MultiDiGraph,
    hexagons: gpd.GeoDataFrame,
    services: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
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
            f"{max_origin_snap_warning_m} m from the walking network. "
            "Inspect peripheral cells and islands/disconnected components."
        )

    if len(services):
        services = services.copy()
        services["service_node"], services["service_snap_m"] = ox.distance.nearest_nodes(
            G,
            X=services.geometry.x.values,
            Y=services.geometry.y.values,
            return_dist=True,
        )
        before = len(services)
        services = services[services["service_snap_m"] <= max_service_snap_distance_m].copy()
        removed = before - len(services)
        if removed:
            print(
                f"Removed {removed} OSM service features because they were farther than "
                f"{max_service_snap_distance_m} m from the pedestrian network."
            )
    else:
        services = gpd.GeoDataFrame(
            columns=["category", "geometry", "service_node"],
            geometry="geometry",
            crs=target_crs,
        )

    print(f"Maximum walking distance: {max_walk_distance_m:.0f} m")

    for category_name in service_categories:
        print(f"Calculating accessibility for category: {category_name}")
        category_services = services[services["category"] == category_name]
        result = calculate_category_accessibility(
            G=G,
            origin_nodes=hexagons["origin_node"].values,
            service_nodes=category_services["service_node"].values,
            cutoff_m=max_walk_distance_m,
            decay_beta=decay_beta,
        )
        hexagons[f"dist_{category_name}"] = result["nearest_distance_m"].round(1)
        hexagons[f"score_{category_name}"] = result["accessibility_score"]

    hexagons["walk_access_raw"] = sum(
        hexagons[f"score_{name}"] * cfg["weight"]
        for name, cfg in service_categories.items()
    )

    raw_min = hexagons["walk_access_raw"].min()
    raw_max = hexagons["walk_access_raw"].max()
    hexagons["walk_access_score"] = (
        (hexagons["walk_access_raw"] - raw_min) / (raw_max - raw_min) * 100
    ).round(2)

    no_access = hexagons["walk_access_raw"] <= 0
    hexagons["walk_access_class"] = "Beyond 15 min cutoff"
    hexagons["walk_access_rank"] = 0

    active = ~no_access
    if active.any():
        ranks = pd.qcut(
            hexagons.loc[active, "walk_access_score"],
            q=5,
            labels=False,
            duplicates="drop",
        )
        class_labels = ["Very low", "Low", "Moderate", "High", "Very high"]
        rank_map = {b: i + 1 for i, b in enumerate(sorted(ranks.unique()))}
        hexagons.loc[active, "walk_access_rank"] = ranks.map(rank_map).astype(int)
        class_map = {i + 1: class_labels[i] for i in range(int(hexagons.loc[active, "walk_access_rank"].max()))}
        hexagons.loc[active, "walk_access_class"] = hexagons.loc[active, "walk_access_rank"].map(class_map)

    print(hexagons["walk_access_class"].value_counts())
    return hexagons


def validate_walkability_index(hexagons: gpd.GeoDataFrame) -> list[str]:
    """Sanity checks on composite index; returns report lines."""
    lines = ["Walkability index validation", "=" * 40]
    weights = {k: v["weight"] for k, v in service_categories.items()}
    lines.append(f"Category weights sum: {sum(weights.values()):.4f} (expect 1.0)")

    score_cols = [f"score_{k}" for k in weights]
    recomputed = sum(hexagons[c] * weights[k] for k, c in zip(weights, score_cols, strict=True))
    diff = (recomputed - hexagons["walk_access_raw"]).abs().max()
    lines.append(f"Composite raw score max deviation: {diff:.6f} (expect 0)")

    for col in score_cols:
        bad = ((hexagons[col] < 0) | (hexagons[col] > 1)).sum()
        lines.append(f"  {col}: values outside [0,1] = {bad}")

    no_access = (hexagons["walk_access_raw"] <= 0).sum()
    lines.append(f"Hexagons beyond {max_walk_minutes} min cutoff from all selected service categories: {no_access}")

    dist_cols = [f"dist_{k}" for k in weights]
    beyond = sum((hexagons[c] > max_walk_distance_m).sum() for c in dist_cols if c in hexagons.columns)
    lines.append(f"Distance values above {max_walk_distance_m:.0f} m cutoff: {beyond} (expect 0)")

    lines.append(f"Score range (0–100): {hexagons['walk_access_score'].min():.2f} – {hexagons['walk_access_score'].max():.2f}")
    lines.append("Class distribution:")
    for cls, n in hexagons["walk_access_class"].value_counts().items():
        sub = hexagons[hexagons["walk_access_class"] == cls]
        lines.append(
            f"  {cls}: {n} cells, mean score {sub['walk_access_score'].mean():.1f}"
        )

    lines.append("")
    lines.append("Method notes:")
    lines.append("- Network: OSM walk, undirected, metric CRS EPSG:25832")
    lines.append("- Access: multi-source Dijkstra from services, exp(-d/700) decay, 15 min cutoff")
    lines.append("- Index: weighted category scores, min-max scaled to 0–100")
    lines.append("- Classes: 'Beyond 15 min cutoff' = raw 0; others = quintiles among accessible cells")
    return lines


# ------------------------------------------------------------
# 7. SAVE OUTPUTS
# ------------------------------------------------------------


def save_outputs(
    hexagons: gpd.GeoDataFrame,
    municipality: gpd.GeoDataFrame,
    services: gpd.GeoDataFrame,
    municipality_boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    distance_cols = [c for c in hexagons.columns if c.startswith("dist_")]
    score_cols = [c for c in hexagons.columns if c.startswith("score_")]

    output_cols = [
        "hex_id",
        "walk_access_score",
        "walk_access_class",
        "walk_access_rank",
        "walk_access_raw",
        "origin_snap_m",
        *distance_cols,
        *score_cols,
        "geometry",
    ]
    hex_output = hexagons[output_cols].copy()

    if len(services):
        services.to_file(output_gpkg, layer="osm_service_points", driver="GPKG")

    hex_output.to_file(
        output_gpkg,
        layer="walking_accessibility_500m_hexagons",
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
            "walk_access_score": "walk_scr",
            "walk_access_class": "walk_cls",
            "walk_access_rank": "walk_rnk",
            "walk_access_raw": "walk_raw",
        }
    )[["hex_id", "walk_scr", "walk_cls", "walk_rnk", "walk_raw", "geometry"]]

    shp_path = output_shp_dir / "odense_walking_accessibility_500m.shp"
    shp_output.to_file(shp_path, driver="ESRI Shapefile", encoding="UTF-8")
    print(f"Shapefile saved to: {shp_path}")

    all_colors = {
        "Beyond 15 min cutoff": "#e0e0e0",
        "Very low": "#fff7bc",
        "Low": "#fec44f",
        "Moderate": "#fe9929",
        "High": "#ec7014",
        "Very high": "#cc0000",
    }
    present_classes = hex_output["walk_access_class"].dropna().unique().tolist()
    colors = {k: all_colors[k] for k in present_classes if k in all_colors}
    hex_output = hex_output.copy()
    hex_output["map_color"] = hex_output["walk_access_class"].map(colors)

    fig, ax = plt.subplots(figsize=(14, 10))
    municipality_boundary.boundary.plot(ax=ax, linewidth=1.2, edgecolor="black")
    municipality.boundary.plot(ax=ax, linewidth=0.5, edgecolor="black", alpha=0.7)
    hex_output.plot(
        ax=ax,
        color=hex_output["map_color"],
        edgecolor="white",
        linewidth=0.15,
        alpha=0.95,
    )
    municipality.boundary.plot(ax=ax, linewidth=0.6, edgecolor="black", alpha=0.8)
    municipality_boundary.boundary.plot(ax=ax, linewidth=1.1, edgecolor="black")
    ax.set_title(
        "Odense Walking Accessibility to Services",
        fontsize=18,
        fontweight="bold",
    )
    ax.set_axis_off()

    legend_order = ["Very high", "High", "Moderate", "Low", "Very low", "Beyond 15 min cutoff"]
    legend_order = [label for label in legend_order if label in colors]
    legend_patches = [mpatches.Patch(color=colors[label], label=label) for label in legend_order]
    ax.legend(
        handles=legend_patches,
        title="Walking accessibility to services",
        loc="lower left",
        frameon=True,
        fontsize=11,
        title_fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_map, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Map saved to: {output_map}")

    plot_osm_basemap_map(hex_output, municipality, municipality_boundary, colors)

    report_lines = validate_walkability_index(hex_output)
    validation_report.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Validation report: {validation_report}")
    print("\n".join(report_lines[:12]))

    summary = (
        hex_output.groupby("walk_access_class")
        .agg(
            n_cells=("hex_id", "count"),
            mean_score=("walk_access_score", "mean"),
            min_score=("walk_access_score", "min"),
            max_score=("walk_access_score", "max"),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "walking_accessibility_class_summary.csv", index=False)
    print(summary)
    return hex_output


def plot_osm_basemap_map(
    hex_output: gpd.GeoDataFrame,
    municipality: gpd.GeoDataFrame,
    municipality_boundary: gpd.GeoDataFrame,
    colors: dict[str, str],
) -> None:
    """Choropleth over OpenStreetMap basemap with semi-transparent hexagons."""
    import contextily as cx

    hex_wm = hex_output.to_crs(web_mercator)
    muni_wm = municipality.to_crs(web_mercator)
    boundary_wm = municipality_boundary.to_crs(web_mercator)

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

    hex_wm = hex_wm.copy()
    hex_wm["map_color"] = hex_wm["walk_access_class"].map(colors)
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
        "Odense Walking Accessibility to Services\n ",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_axis_off()

    legend_order = ["Very high", "High", "Moderate", "Low", "Very low", "Beyond 15 min cutoff"]
    legend_order = [label for label in legend_order if label in colors]
    legend_patches = [
        mpatches.Patch(facecolor=colors[label], edgecolor="gray", label=label, alpha=hex_alpha)
        for label in legend_order
    ]
    ax.legend(
        handles=legend_patches,
        title="Walking accessibility",
        loc="lower left",
        frameon=True,
        fontsize=10,
        title_fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(output_map_osm, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"OSM basemap map saved to: {output_map_osm}")


def map_only_from_gpkg() -> None:
    """Regenerate maps and validation from existing GeoPackage (no OSM re-download)."""
    municipality = gpd.read_file(output_gpkg, layer="odense_municipality_districts").to_crs(target_crs)
    municipality_boundary = gpd.GeoDataFrame(
        geometry=[municipality.geometry.union_all()],
        crs=target_crs,
    )
    hex_output = gpd.read_file(output_gpkg, layer="walking_accessibility_500m_hexagons").to_crs(target_crs)
    services = gpd.GeoDataFrame()
    try:
        services = gpd.read_file(output_gpkg, layer="osm_service_points")
    except Exception:
        pass

    all_colors = {
        "Beyond 15 min cutoff": "#e0e0e0",
        "Very low": "#fff7bc",
        "Low": "#fec44f",
        "Moderate": "#fe9929",
        "High": "#ec7014",
        "Very high": "#cc0000",
    }
    present = hex_output["walk_access_class"].dropna().unique().tolist()
    colors = {k: all_colors[k] for k in present if k in all_colors}

    hex_output = hex_output.copy()
    hex_output["map_color"] = hex_output["walk_access_class"].map(colors)

    fig, ax = plt.subplots(figsize=(14, 10))
    municipality_boundary.boundary.plot(ax=ax, linewidth=1.2, edgecolor="black")
    municipality.boundary.plot(ax=ax, linewidth=0.5, edgecolor="black", alpha=0.7)
    hex_output.plot(
        ax=ax,
        color=hex_output["map_color"],
        edgecolor="white",
        linewidth=0.15,
        alpha=0.95,
    )
    municipality.boundary.plot(ax=ax, linewidth=0.6, edgecolor="black", alpha=0.8)
    municipality_boundary.boundary.plot(ax=ax, linewidth=1.1, edgecolor="black")
    ax.set_title(
        "Odense Walking Accessibility to Services",
        fontsize=18,
        fontweight="bold",
    )
    ax.set_axis_off()
    legend_order = ["Very high", "High", "Moderate", "Low", "Very low", "Beyond 15 min cutoff"]
    legend_order = [label for label in legend_order if label in colors]
    legend_patches = [mpatches.Patch(color=colors[label], label=label) for label in legend_order]
    ax.legend(
        handles=legend_patches,
        title="Walking accessibility to services",
        loc="lower left",
        frameon=True,
        fontsize=11,
        title_fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_map, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Map saved to: {output_map}")

    plot_osm_basemap_map(hex_output, municipality, municipality_boundary, colors)
    report_lines = validate_walkability_index(hex_output)
    validation_report.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Validation report: {validation_report}")


# ------------------------------------------------------------
# 8. MAIN
# ------------------------------------------------------------


def main() -> None:
    import sys

    if "--map-only" in sys.argv:
        if not output_gpkg.exists():
            raise FileNotFoundError(f"No saved results at {output_gpkg}. Run full pipeline first.")
        map_only_from_gpkg()
        print("Finished (map-only).")
        return

    municipality, hexagons, municipality_boundary = load_inputs()
    boundary_wgs84 = municipality_boundary.to_crs("EPSG:4326").geometry.iloc[0]

    G = download_walk_network(municipality_boundary)
    services = download_all_services(boundary_wgs84)
    hexagons = compute_walkability_index(G, hexagons, services)
    save_outputs(hexagons, municipality, services, municipality_boundary)
    print("Finished.")


if __name__ == "__main__":
    main()
