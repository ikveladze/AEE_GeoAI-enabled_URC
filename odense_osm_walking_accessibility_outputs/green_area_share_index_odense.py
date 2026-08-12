# ============================================================
# OSM-based Green-Area Share Index
# Odense Municipality, 500 m hexagonal cells
# Output: GeoPackage + Shapefile + map + validation report
# CRS: EPSG:25832
# ============================================================
#
# IMPORTANT:
# This model measures the share of each 500 m hexagonal cell covered by
# OSM-mapped green/open/natural land-cover features. It is not an accessibility
# index and does not use a travel-time threshold. It is an areal composition
# indicator: green area within cell / total cell area.
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

output_dir = SCRIPT_DIR / "odense_osm_green_area_share_outputs"
output_dir.mkdir(exist_ok=True)

output_gpkg = output_dir / "odense_green_area_share_500m.gpkg"
output_shp_dir = output_dir / "odense_green_area_share_500m_shp"
output_shp_dir.mkdir(exist_ok=True)

output_map = output_dir / "odense_green_area_share_500m_map.png"
output_map_osm = output_dir / "odense_green_area_share_500m_map_osm.png"
validation_report = output_dir / "green_area_share_validation.txt"

hex_alpha = 0.78

# ------------------------------------------------------------
# 2. GREEN-AREA DEFINITION
# ------------------------------------------------------------
# By default, this script focuses on green/open/natural areas relevant for
# urban environmental exposure. Agricultural production land is excluded from
# the main definition but can be included through include_agriculture=True.
#
# Recommended manuscript wording:
# "green-area share" should be interpreted as OSM-mapped green/open/natural
# land-cover share, not necessarily legally accessible public green space.

include_agriculture = False

green_tags_base = {
    "leisure": [
        "park",
        "garden",
        "nature_reserve",
        "recreation_ground",
        "common",
        "playground",
        "pitch",
        "sports_centre",
        "golf_course",
    ],
    "landuse": [
        "forest",
        "grass",
        "meadow",
        "recreation_ground",
        "village_green",
        "allotments",
        "orchard",
        "vineyard",
        "cemetery",
    ],
    "natural": [
        "wood",
        "grassland",
        "scrub",
        "heath",
        "wetland",
        "fell",
        "bare_rock",
    ],
    "boundary": [
        "national_park",
        "protected_area",
    ],
}

agriculture_tags = {
    "landuse": [
        "farmland",
        "farmyard",
        "pasture",
    ]
}

# ------------------------------------------------------------
# 3. INPUT LOADING
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
    hexagons["hex_id"] = [f"OD_GREEN_{i + 1:04d}" for i in range(len(hexagons))]
    hexagons["cell_area_m2"] = hexagons.geometry.area

    print(f"Number of clipped 500 m hexagons: {len(hexagons)}")
    return municipality, hexagons, municipality_boundary


# ------------------------------------------------------------
# 4. OSM GREEN-AREA EXTRACTION
# ------------------------------------------------------------

def make_green_tags() -> dict:
    tags = {k: list(v) for k, v in green_tags_base.items()}
    if include_agriculture:
        for key, values in agriculture_tags.items():
            tags.setdefault(key, [])
            tags[key].extend(values)
    return tags


def download_osm_green_areas(municipality_boundary: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    boundary_wgs84 = municipality_boundary.to_crs("EPSG:4326")
    boundary_polygon_wgs84 = boundary_wgs84.geometry.iloc[0]

    tags = make_green_tags()

    print("Downloading OSM green/open/natural area features...")
    print(f"include_agriculture = {include_agriculture}")

    features = ox.features_from_polygon(
        boundary_polygon_wgs84,
        tags=tags,
    )

    if features.empty:
        print("No OSM green-area features found.")
        return gpd.GeoDataFrame(
            columns=["green_type", "geometry"],
            geometry="geometry",
            crs=target_crs,
        )

    features = features.reset_index()
    features = features[features.geometry.notnull()].copy()
    features = features.to_crs(target_crs)

    # Keep polygonal area features only. Point features cannot contribute to
    # area share and line features should not be buffered unless a buffer-based
    # green-corridor model is explicitly intended.
    features = features[
        features.geometry.geom_type.isin(["Polygon", "MultiPolygon"])
    ].copy()

    if features.empty:
        print("No polygonal green-area features found after filtering.")
        return gpd.GeoDataFrame(
            columns=["green_type", "geometry"],
            geometry="geometry",
            crs=target_crs,
        )

    # Assign a simple diagnostic type based on the first matched OSM key.
    def infer_green_type(row):
        for key in ["leisure", "landuse", "natural", "boundary"]:
            if key in row and pd.notna(row[key]):
                return f"{key}={row[key]}"
        return "green_area"

    features["green_type"] = features.apply(infer_green_type, axis=1)

    # Repair invalid geometries if needed.
    features["geometry"] = features.geometry.make_valid()
    features = features[~features.geometry.is_empty].copy()

    # Clip to municipality boundary.
    features = gpd.clip(features[["green_type", "geometry"]], municipality_boundary)
    features = features.reset_index(drop=True)
    features["green_id"] = [f"OSM_GREEN_{i + 1:05d}" for i in range(len(features))]
    features["green_area_m2"] = features.geometry.area

    print(f"OSM polygonal green-area features after clipping: {len(features)}")
    print(f"Total OSM green area before union/dissolve: {features['green_area_m2'].sum()/1_000_000:.2f} km²")
    print(features["green_type"].value_counts().head(20))

    return features[["green_id", "green_type", "green_area_m2", "geometry"]]


# ------------------------------------------------------------
# 5. GREEN-AREA SHARE CALCULATION
# ------------------------------------------------------------

def compute_green_area_share(
    hexagons: gpd.GeoDataFrame,
    green_areas: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    hexagons = hexagons.copy()

    if green_areas.empty:
        hexagons["green_area_m2"] = 0.0
        hexagons["green_share"] = 0.0
        hexagons["green_share_pct"] = 0.0
        hexagons["green_class"] = "Very low"
        hexagons["green_rank"] = 1
        return hexagons

    # Dissolve/union green polygons to remove overlapping OSM features.
    # This is essential: without unioning, parks/nature_reserves/landuse polygons
    # that overlap each other can be double counted inside a hexagon.
    print("Dissolving OSM green areas to prevent overlap double-counting...")
    green_union_geom = green_areas.geometry.union_all()

    green_union = gpd.GeoDataFrame(
        geometry=[green_union_geom],
        crs=target_crs,
    )
    green_union["green_area_m2"] = green_union.geometry.area

    # Intersect hexagons with dissolved green polygon.
    print("Intersecting green areas with 500 m hexagons...")
    hex_for_overlay = hexagons[["hex_id", "cell_area_m2", "geometry"]].copy()
    intersections = gpd.overlay(
        hex_for_overlay,
        green_union[["geometry"]],
        how="intersection",
        keep_geom_type=True,
    )

    if intersections.empty:
        green_by_hex = pd.DataFrame({"hex_id": hexagons["hex_id"], "green_area_m2": 0.0})
    else:
        intersections["intersect_area_m2"] = intersections.geometry.area
        green_by_hex = (
            intersections.groupby("hex_id", as_index=False)
            .agg(green_area_m2=("intersect_area_m2", "sum"))
        )

    hexagons = hexagons.merge(green_by_hex, on="hex_id", how="left")
    hexagons["green_area_m2"] = hexagons["green_area_m2"].fillna(0.0)

    # Cap green area at cell area to protect against tiny numerical overlay errors.
    hexagons["green_area_m2"] = np.minimum(hexagons["green_area_m2"], hexagons["cell_area_m2"])

    hexagons["green_share"] = hexagons["green_area_m2"] / hexagons["cell_area_m2"]
    hexagons["green_share_pct"] = (hexagons["green_share"] * 100).round(2)

    # Relative quintile classification.
    # This makes the map comparable with previous accessibility maps as a
    # relative spatial diagnostic. For policy thresholds, replace with fixed
    # class breaks, e.g. 0–10, 10–25, 25–50, 50–75, 75–100%.
    ranks = pd.qcut(
        hexagons["green_share_pct"],
        q=5,
        labels=False,
        duplicates="drop",
    )

    class_labels = ["Very low", "Low", "Moderate", "High", "Very high"]
    rank_map = {b: i + 1 for i, b in enumerate(sorted(ranks.unique()))}
    hexagons["green_rank"] = ranks.map(rank_map).astype(int)

    max_rank = int(hexagons["green_rank"].max())
    class_map = {i + 1: class_labels[i] for i in range(max_rank)}
    hexagons["green_class"] = hexagons["green_rank"].map(class_map)

    print("Green-area share class distribution:")
    print(hexagons["green_class"].value_counts())

    return hexagons


# ------------------------------------------------------------
# 6. OUTPUTS AND MAPS
# ------------------------------------------------------------

def green_class_colors() -> dict[str, str]:
    return {
        "Very low": "#edf8e9",
        "Low": "#bae4b3",
        "Moderate": "#74c476",
        "High": "#31a354",
        "Very high": "#006d2c",
    }


def save_outputs(
    hexagons: gpd.GeoDataFrame,
    municipality: gpd.GeoDataFrame,
    green_areas: gpd.GeoDataFrame,
    municipality_boundary: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    output_cols = [
        "hex_id",
        "cell_area_m2",
        "green_area_m2",
        "green_share",
        "green_share_pct",
        "green_class",
        "green_rank",
        "geometry",
    ]

    hex_output = hexagons[output_cols].copy()

    if len(green_areas):
        green_areas.to_file(output_gpkg, layer="osm_green_area_features", driver="GPKG")

        green_union = gpd.GeoDataFrame(
            geometry=[green_areas.geometry.union_all()],
            crs=target_crs,
        )
        green_union["green_area_m2"] = green_union.geometry.area
        green_union.to_file(output_gpkg, layer="osm_green_area_union", driver="GPKG")

    hex_output.to_file(
        output_gpkg,
        layer="green_area_share_500m_hexagons",
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
            "cell_area_m2": "cell_m2",
            "green_area_m2": "green_m2",
            "green_share": "green_sh",
            "green_share_pct": "green_pct",
            "green_class": "green_cls",
            "green_rank": "green_rnk",
        }
    )[
        [
            "hex_id",
            "cell_m2",
            "green_m2",
            "green_sh",
            "green_pct",
            "green_cls",
            "green_rnk",
            "geometry",
        ]
    ]

    shp_path = output_shp_dir / "odense_green_area_share_500m.shp"
    shp_output.to_file(shp_path, driver="ESRI Shapefile", encoding="UTF-8")
    print(f"Shapefile saved to: {shp_path}")

    plot_static_map(hex_output, municipality, municipality_boundary, output_map)
    plot_osm_basemap_map(hex_output, municipality, municipality_boundary)

    report_lines = validate_green_area_share(hex_output, green_areas, municipality_boundary)
    validation_report.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Validation report saved to: {validation_report}")

    summary = (
        hex_output.groupby("green_class")
        .agg(
            n_cells=("hex_id", "count"),
            mean_green_pct=("green_share_pct", "mean"),
            min_green_pct=("green_share_pct", "min"),
            max_green_pct=("green_share_pct", "max"),
            total_green_km2=("green_area_m2", lambda s: s.sum() / 1_000_000),
        )
        .reset_index()
    )
    summary.to_csv(output_dir / "green_area_share_class_summary.csv", index=False)
    print(summary)

    return hex_output


def plot_static_map(
    hex_output: gpd.GeoDataFrame,
    municipality: gpd.GeoDataFrame,
    municipality_boundary: gpd.GeoDataFrame,
    output_path: Path,
) -> None:
    colors = green_class_colors()

    plot_gdf = hex_output.copy()
    plot_gdf["map_color"] = plot_gdf["green_class"].map(colors)

    fig, ax = plt.subplots(figsize=(14, 10))

    plot_gdf.plot(
        ax=ax,
        color=plot_gdf["map_color"],
        edgecolor="white",
        linewidth=0.15,
        alpha=0.96,
    )

    municipality.boundary.plot(ax=ax, linewidth=0.6, edgecolor="black", alpha=0.8)
    municipality_boundary.boundary.plot(ax=ax, linewidth=1.2, edgecolor="black")

    ax.set_title(
        "Odense Green-Area Share by 500 m Cell",
        fontsize=18,
        fontweight="bold",
    )
    ax.set_axis_off()

    legend_order = ["Very high", "High", "Moderate", "Low", "Very low"]
    legend_order = [label for label in legend_order if label in plot_gdf["green_class"].unique()]
    legend_patches = [
        mpatches.Patch(facecolor=colors[label], edgecolor="gray", label=label)
        for label in legend_order
    ]

    ax.legend(
        handles=legend_patches,
        title="Green-area share",
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
) -> None:
    try:
        import contextily as cx
    except ImportError:
        print("contextily is not installed; skipping OSM basemap map.")
        return

    colors = green_class_colors()

    hex_wm = hex_output.to_crs(web_mercator).copy()
    muni_wm = municipality.to_crs(web_mercator)
    boundary_wm = municipality_boundary.to_crs(web_mercator)

    hex_wm["map_color"] = hex_wm["green_class"].map(colors)

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
        "Odense Green-Area Share by 500 m Cell",
        fontsize=16,
        fontweight="bold",
    )
    ax.set_axis_off()

    legend_order = ["Very high", "High", "Moderate", "Low", "Very low"]
    legend_order = [label for label in legend_order if label in hex_wm["green_class"].unique()]
    legend_patches = [
        mpatches.Patch(facecolor=colors[label], edgecolor="gray", label=label, alpha=hex_alpha)
        for label in legend_order
    ]

    ax.legend(
        handles=legend_patches,
        title="Green-area share",
        loc="lower left",
        frameon=True,
        fontsize=10,
        title_fontsize=11,
    )

    plt.tight_layout()
    plt.savefig(output_map_osm, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"OSM basemap map saved to: {output_map_osm}")


# ------------------------------------------------------------
# 7. VALIDATION
# ------------------------------------------------------------

def validate_green_area_share(
    hex_output: gpd.GeoDataFrame,
    green_areas: gpd.GeoDataFrame,
    municipality_boundary: gpd.GeoDataFrame,
) -> list[str]:
    lines = ["Green-area share validation", "=" * 40]
    lines.append(f"include_agriculture: {include_agriculture}")
    lines.append(f"Number of 500 m cells: {len(hex_output)}")
    lines.append(f"Municipality area: {municipality_boundary.geometry.area.iloc[0] / 1_000_000:.2f} km²")

    if len(green_areas):
        dissolved_area = green_areas.geometry.union_all().area / 1_000_000
        lines.append(f"OSM green features: {len(green_areas)}")
        lines.append(f"Dissolved OSM green area: {dissolved_area:.2f} km²")
    else:
        lines.append("OSM green features: 0")
        lines.append("Dissolved OSM green area: 0.00 km²")

    total_cell_area = hex_output["cell_area_m2"].sum() / 1_000_000
    total_green_area = hex_output["green_area_m2"].sum() / 1_000_000
    weighted_green_share = (
        hex_output["green_area_m2"].sum() / hex_output["cell_area_m2"].sum() * 100
    )

    lines.append(f"Total clipped hexagon area: {total_cell_area:.2f} km²")
    lines.append(f"Total green area counted in cells: {total_green_area:.2f} km²")
    lines.append(f"Area-weighted green share: {weighted_green_share:.2f}%")

    bad_low = (hex_output["green_share_pct"] < -1e-9).sum()
    bad_high = (hex_output["green_share_pct"] > 100 + 1e-9).sum()
    lines.append(f"Cells with green_share_pct < 0: {bad_low}")
    lines.append(f"Cells with green_share_pct > 100: {bad_high}")

    lines.append(f"Green share range: {hex_output['green_share_pct'].min():.2f}% – {hex_output['green_share_pct'].max():.2f}%")
    lines.append("Class distribution:")
    for cls, n in hex_output["green_class"].value_counts().items():
        sub = hex_output[hex_output["green_class"] == cls]
        lines.append(
            f"  {cls}: {n} cells, mean green share {sub['green_share_pct'].mean():.2f}%"
        )

    lines.append("")
    lines.append("Method notes:")
    lines.append("- Data: OSM polygonal green/open/natural land-cover features")
    lines.append("- Geometry handling: polygons/multipolygons only; line and point features excluded")
    lines.append("- Overlap control: green polygons dissolved before intersection to avoid double counting")
    lines.append("- Indicator: green_area_m2 / clipped_cell_area_m2")
    lines.append("- Classes: quintiles of green_share_pct")
    lines.append("- Limitation: OSM completeness varies; indicator is not equivalent to official land cover or legally accessible public green space")
    return lines


# ------------------------------------------------------------
# 8. MAP-ONLY MODE
# ------------------------------------------------------------

def map_only_from_gpkg() -> None:
    if not output_gpkg.exists():
        raise FileNotFoundError(f"No saved results at {output_gpkg}. Run the full pipeline first.")

    municipality = gpd.read_file(output_gpkg, layer="odense_municipality_districts").to_crs(target_crs)
    municipality_boundary = gpd.GeoDataFrame(
        geometry=[municipality.geometry.union_all()],
        crs=target_crs,
    )
    hex_output = gpd.read_file(output_gpkg, layer="green_area_share_500m_hexagons").to_crs(target_crs)

    try:
        green_areas = gpd.read_file(output_gpkg, layer="osm_green_area_features").to_crs(target_crs)
    except Exception:
        green_areas = gpd.GeoDataFrame(columns=["green_type", "geometry"], geometry="geometry", crs=target_crs)

    plot_static_map(hex_output, municipality, municipality_boundary, output_map)
    plot_osm_basemap_map(hex_output, municipality, municipality_boundary)

    report_lines = validate_green_area_share(hex_output, green_areas, municipality_boundary)
    validation_report.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Validation report: {validation_report}")


# ------------------------------------------------------------
# 9. MAIN
# ------------------------------------------------------------

def main() -> None:
    import sys

    if "--map-only" in sys.argv:
        map_only_from_gpkg()
        print("Finished (map-only).")
        return

    municipality, hexagons, municipality_boundary = load_inputs()
    green_areas = download_osm_green_areas(municipality_boundary)
    hexagons = compute_green_area_share(hexagons, green_areas)
    save_outputs(hexagons, municipality, green_areas, municipality_boundary)

    print("Finished.")


if __name__ == "__main__":
    main()
