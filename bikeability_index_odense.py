# ============================================================
# OSM-based Cycling Accessibility to Services (Bikeability Index)
# Odense Municipality, 500 m hexagonal cells
# Output: GeoPackage + Shapefile + maps + validation report
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
# 1. INPUT FILES
# ------------------------------------------------------------

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

output_dir = SCRIPT_DIR / "odense_osm_cycling_accessibility_outputs"
output_dir.mkdir(exist_ok=True)

output_gpkg = output_dir / "odense_cycling_accessibility_services_500m.gpkg"
output_shp_dir = output_dir / "odense_cycling_accessibility_services_500m_shp"
output_shp_dir.mkdir(exist_ok=True)

output_map_5 = output_dir / "odense_cycling_accessibility_services_500m_map_5min.png"
output_map_15 = output_dir / "odense_cycling_accessibility_services_500m_map_15min.png"
output_map_osm_15 = output_dir / "odense_cycling_accessibility_services_500m_map_15min_osm.png"
validation_report = output_dir / "bikeability_index_validation.txt"

hex_alpha = 0.62

# ------------------------------------------------------------
# 2. MODEL PARAMETERS
# ------------------------------------------------------------

# Cycling speed assumption:
# 15 km/h = 4.17 m/s. This is a defensible urban utility-cycling speed.
cycling_speed_m_per_s = 15_000 / 3_600

# Required thresholds: 5–15 min.
# The 5-min index captures very local service access.
# The 15-min index captures neighbourhood/municipal-scale cycling access.
threshold_minutes = [5, 15]
max_cycling_distances_m = {
    minutes: cycling_speed_m_per_s * 60 * minutes
    for minutes in threshold_minutes
}

# Distance decay for cycling.
# Larger than walking because cycling covers longer effective distances.
decay_beta_m = 1800

# Snapping diagnostics.
# Services far from the bike network are excluded to avoid unrealistic snapping.
max_service_snap_distance_m = 250
max_origin_snap_warning_m = 300

# ------------------------------------------------------------
# 3. SERVICE CATEGORIES
# ------------------------------------------------------------
# Same service taxonomy and weights as the walking accessibility model.

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


# ------------------------------------------------------------
# 4. INPUT LOADING
# ------------------------------------------------------------

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
    hexagons["hex_id"] = [f"OD_BIKE_{i + 1:04d}" for i in range(len(hexagons))]

    print(f"Number of clipped 500 m hexagons: {len(hexagons)}")
    return municipality, hexagons, municipality_boundary


# ------------------------------------------------------------
# 5. OSM CYCLING NETWORK
# ------------------------------------------------------------

def download_bike_network(municipality_boundary: gpd.GeoDataFrame) -> nx.MultiGraph:
    boundary_wgs84 = municipality_boundary.to_crs("EPSG:4326")
    boundary_polygon_wgs84 = boundary_wgs84.geometry.iloc[0]

    print("Downloading OSM cycling network...")
    G = ox.graph_from_polygon(
        boundary_polygon_wgs84,
        network_type="bike",
        simplify=True,
        # Keep all disconnected components, otherwise peripheral settlements
        # may be removed and artificially assigned low/no access.
        retain_all=True,
        truncate_by_edge=True,
    )

    G = ox.project_graph(G, to_crs=target_crs)

    # Same analytical logic as walking: accessibility is treated as proximity on
    # the bicycle-permitted network. Converting to undirected avoids directional
    # bias in a multi-source Dijkstra calculation.
    #
    # If you want strict legal routing with one-way cycling restrictions, keep
    # the graph directed and calculate origin-to-destination paths from each
    # origin separately. That is slower but more directionally realistic.
    G = ox.convert.to_undirected(G)

    for _, _, data in G.edges(data=True):
        if "length" not in data:
            data["length"] = 0

    print(f"Network nodes: {len(G.nodes)}")
    print(f"Network edges: {len(G.edges)} (undirected)")
    return G


# ------------------------------------------------------------
# 6. OSM SERVICE FEATURES
# ------------------------------------------------------------

def _geom_to_point(geom):
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        return geom.representative_point()
    if geom.geom_type in ("LineString", "MultiLineString"):
        return geom.interpolate(0.5, normalized=True)
    return geom


def download_osm_features(category_name: str, tags: dict, polygon_wgs84) -> gpd.GeoDataFrame:
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

    print(f"Total OSM service points before snapping QA: {len(services)}")
    if len(services):
        print(services["category"].value_counts())
    return services


# ------------------------------------------------------------
# 7. NETWORK ACCESSIBILITY
# ------------------------------------------------------------

def calculate_category_accessibility(
    G: nx.MultiGraph,
    origin_nodes,
    service_nodes,
    cutoff_m: float,
    decay_beta_m: float,
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
            scores.append(np.exp(-d / decay_beta_m))

    return pd.DataFrame({
        "nearest_distance_m": nearest_distances,
        "accessibility_score": scores,
    })


def compute_bikeability_index(
    G: nx.MultiGraph,
    hexagons: gpd.GeoDataFrame,
    services: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    hexagons = hexagons.copy()

    # Internal origin point for each clipped hexagon.
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
            f"{max_origin_snap_warning_m} m from the cycling network."
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
                f"{max_service_snap_distance_m} m from the cycling network."
            )
    else:
        services = gpd.GeoDataFrame(
            columns=["category", "geometry", "service_node", "service_snap_m"],
            geometry="geometry",
            crs=target_crs,
        )

    print("Cycling thresholds:")
    for minutes, dist_m in max_cycling_distances_m.items():
        print(f"  {minutes} min = {dist_m:.0f} m at {cycling_speed_m_per_s:.2f} m/s")

    # Calculate category distances/scores for each threshold.
    for minutes, cutoff_m in max_cycling_distances_m.items():
        print(f"\nCalculating {minutes}-minute cycling accessibility...")

        for category_name in service_categories:
            print(f"  Category: {category_name}")

            category_services = services[services["category"] == category_name]

            result = calculate_category_accessibility(
                G=G,
                origin_nodes=hexagons["origin_node"].values,
                service_nodes=category_services["service_node"].values,
                cutoff_m=cutoff_m,
                decay_beta_m=decay_beta_m,
            )

            hexagons[f"dist_{category_name}_{minutes}m"] = result["nearest_distance_m"].round(1)
            hexagons[f"score_{category_name}_{minutes}m"] = result["accessibility_score"]

        raw_col = f"bike_access_raw_{minutes}m"
        score_col = f"bike_access_score_{minutes}m"
        class_col = f"bike_access_class_{minutes}m"
        rank_col = f"bike_access_rank_{minutes}m"

        hexagons[raw_col] = sum(
            hexagons[f"score_{name}_{minutes}m"] * cfg["weight"]
            for name, cfg in service_categories.items()
        )

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

    return hexagons, services


# ------------------------------------------------------------
# 8. OUTPUTS AND MAPS
# ------------------------------------------------------------

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
    services: gpd.GeoDataFrame,
    municipality_boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    output_cols = [
        "hex_id",
        "origin_snap_m",
        "bike_access_score_5m",
        "bike_access_class_5m",
        "bike_access_rank_5m",
        "bike_access_raw_5m",
        "bike_access_score_15m",
        "bike_access_class_15m",
        "bike_access_rank_15m",
        "bike_access_raw_15m",
        *[c for c in hexagons.columns if c.startswith("dist_")],
        *[c for c in hexagons.columns if c.startswith("score_")],
        "geometry",
    ]

    hex_output = hexagons[output_cols].copy()

    if len(services):
        services.to_file(output_gpkg, layer="osm_service_points", driver="GPKG")

    hex_output.to_file(
        output_gpkg,
        layer="cycling_accessibility_500m_hexagons",
        driver="GPKG",
    )

    municipality.to_file(
        output_gpkg,
        layer="odense_municipality_districts",
        driver="GPKG",
    )

    print(f"GeoPackage saved to: {output_gpkg}")

    # Shapefile: shortened field names due to 10-character limitation.
    shp_output = hex_output.rename(
        columns={
            "bike_access_score_5m": "bike5_scr",
            "bike_access_class_5m": "bike5_cls",
            "bike_access_rank_5m": "bike5_rnk",
            "bike_access_raw_5m": "bike5_raw",
            "bike_access_score_15m": "bike15_scr",
            "bike_access_class_15m": "bike15_cls",
            "bike_access_rank_15m": "bike15_rnk",
            "bike_access_raw_15m": "bike15_raw",
            "origin_snap_m": "orig_snap",
        }
    )[
        [
            "hex_id",
            "orig_snap",
            "bike5_scr",
            "bike5_cls",
            "bike5_rnk",
            "bike5_raw",
            "bike15_scr",
            "bike15_cls",
            "bike15_rnk",
            "bike15_raw",
            "geometry",
        ]
    ]

    shp_path = output_shp_dir / "odense_cycling_accessibility_500m.shp"
    shp_output.to_file(shp_path, driver="ESRI Shapefile", encoding="UTF-8")
    print(f"Shapefile saved to: {shp_path}")

    plot_static_map(hex_output, municipality, municipality_boundary, minutes=5, output_path=output_map_5)
    plot_static_map(hex_output, municipality, municipality_boundary, minutes=15, output_path=output_map_15)
    plot_osm_basemap_map(hex_output, municipality, municipality_boundary, minutes=15)

    report_lines = validate_bikeability_index(hex_output)
    validation_report.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Validation report saved to: {validation_report}")

    summary_frames = []
    for minutes in threshold_minutes:
        class_col = f"bike_access_class_{minutes}m"
        score_col = f"bike_access_score_{minutes}m"
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
    summary_all.to_csv(output_dir / "cycling_accessibility_class_summary.csv", index=False)
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
    class_col = f"bike_access_class_{minutes}m"

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
        f"Odense Cycling Accessibility to Services ({minutes}-min OSM network)",
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
        title=f"Cycling accessibility to services ({minutes} min)",
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
    minutes: int = 15,
) -> None:
    try:
        import contextily as cx
    except ImportError:
        print("contextily is not installed; skipping OSM basemap map.")
        return

    colors = class_colors(minutes)
    class_col = f"bike_access_class_{minutes}m"

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
        "Odense Cycling Accessibility to Services\n\u00A0",
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
        title=f"Cycling accessibility",
        loc="lower left",
        frameon=True,
        fontsize=10,
        title_fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(output_map_osm_15, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"OSM basemap map saved to: {output_map_osm_15}")


# ------------------------------------------------------------
# 9. VALIDATION
# ------------------------------------------------------------

def validate_bikeability_index(hexagons: gpd.GeoDataFrame) -> list[str]:
    lines = ["Bikeability index validation", "=" * 40]
    weights = {k: v["weight"] for k, v in service_categories.items()}
    lines.append(f"Category weights sum: {sum(weights.values()):.4f} (expect 1.0)")
    lines.append(f"Cycling speed: {cycling_speed_m_per_s:.2f} m/s ({cycling_speed_m_per_s*3.6:.1f} km/h)")
    lines.append(f"Distance decay beta: {decay_beta_m:.0f} m")

    for minutes in threshold_minutes:
        lines.append("")
        lines.append(f"{minutes}-minute threshold")
        lines.append("-" * 30)
        lines.append(f"Maximum network distance: {max_cycling_distances_m[minutes]:.0f} m")

        score_cols = [f"score_{k}_{minutes}m" for k in weights]
        raw_col = f"bike_access_raw_{minutes}m"
        class_col = f"bike_access_class_{minutes}m"
        score_col = f"bike_access_score_{minutes}m"

        recomputed = sum(
            hexagons[c] * weights[k]
            for k, c in zip(weights, score_cols, strict=True)
        )
        diff = (recomputed - hexagons[raw_col]).abs().max()
        lines.append(f"Composite raw score max deviation: {diff:.6f} (expect 0)")

        for col in score_cols:
            bad = ((hexagons[col] < 0) | (hexagons[col] > 1)).sum()
            lines.append(f"  {col}: values outside [0,1] = {bad}")

        beyond = (hexagons[raw_col] <= 0).sum()
        lines.append(f"Hexagons beyond {minutes} min cutoff from all selected service categories: {beyond}")
        lines.append(f"Score range (0–100): {hexagons[score_col].min():.2f} – {hexagons[score_col].max():.2f}")
        lines.append("Class distribution:")
        for cls, n in hexagons[class_col].value_counts().items():
            sub = hexagons[hexagons[class_col] == cls]
            lines.append(f"  {cls}: {n} cells, mean score {sub[score_col].mean():.1f}")

    lines.append("")
    lines.append("Method notes:")
    lines.append("- Network: OSM bike network, undirected, metric CRS EPSG:25832")
    lines.append("- Access: multi-source Dijkstra from services")
    lines.append("- Thresholds: 5 and 15 min, using 15 km/h cycling speed")
    lines.append("- Decay: exp(-d/1800)")
    lines.append("- Index: weighted category scores, min-max scaled to 0–100")
    lines.append("- Classes: cutoff class = raw 0; others = quintiles among accessible cells")
    return lines


# ------------------------------------------------------------
# 10. MAP-ONLY MODE
# ------------------------------------------------------------

def map_only_from_gpkg() -> None:
    if not output_gpkg.exists():
        raise FileNotFoundError(f"No saved results at {output_gpkg}. Run the full pipeline first.")

    municipality = gpd.read_file(output_gpkg, layer="odense_municipality_districts").to_crs(target_crs)
    municipality_boundary = gpd.GeoDataFrame(
        geometry=[municipality.geometry.union_all()],
        crs=target_crs,
    )
    hex_output = gpd.read_file(output_gpkg, layer="cycling_accessibility_500m_hexagons").to_crs(target_crs)

    plot_static_map(hex_output, municipality, municipality_boundary, minutes=5, output_path=output_map_5)
    plot_static_map(hex_output, municipality, municipality_boundary, minutes=15, output_path=output_map_15)
    plot_osm_basemap_map(hex_output, municipality, municipality_boundary, minutes=15)

    report_lines = validate_bikeability_index(hex_output)
    validation_report.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Validation report: {validation_report}")


# ------------------------------------------------------------
# 11. MAIN
# ------------------------------------------------------------

def main() -> None:
    import sys

    if "--map-only" in sys.argv:
        map_only_from_gpkg()
        print("Finished (map-only).")
        return

    municipality, hexagons, municipality_boundary = load_inputs()
    boundary_wgs84 = municipality_boundary.to_crs("EPSG:4326").geometry.iloc[0]

    G = download_bike_network(municipality_boundary)
    services = download_all_services(boundary_wgs84)

    hexagons, services = compute_bikeability_index(G, hexagons, services)
    save_outputs(hexagons, municipality, services, municipality_boundary)

    print("Finished.")


if __name__ == "__main__":
    main()
