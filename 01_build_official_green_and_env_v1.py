#!/usr/bin/env python3
"""Rebuild green-area share and multisource environmental index from official data (v1).

Uses GeoDanmark composites + WDPA/§3 protected nature + heat/noise/air rasters.
Produces the same primary column names expected by Stage B fusion scripts.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
MAP_LAYERS = PROJECT_DIR / "Map Layers"
OFFICIAL = MAP_LAYERS / "Official Environmental"
TARGET_CRS = "EPSG:25832"
ROAD_BUFFER_M = 50.0

# Canonical analysis grid = walking accessibility hexes (n=1539), not the raw 2596 grid.
WALK_GPKG = SCRIPT_DIR / "odense_walking_accessibility_services_500m.gpkg"
MUNI_CANDIDATES = [
    MAP_LAYERS / "Odense_Municipality.shp",
    MAP_LAYERS / "Odense_Municipality_1.gpkg",
    MAP_LAYERS / "Odense_Municipality.gpkg",
]


def load_hex_muni():
    if not WALK_GPKG.exists():
        raise FileNotFoundError(f"Missing canonical hex gpkg: {WALK_GPKG}")
    hexes = gpd.read_file(WALK_GPKG, layer="walking_accessibility_500m_hexagons")
    try:
        muni = gpd.read_file(WALK_GPKG, layer="odense_municipality_districts")
    except Exception:
        muni_path = next(p for p in MUNI_CANDIDATES if p.exists())
        muni = gpd.read_file(muni_path)
    hexes = hexes.to_crs(TARGET_CRS)
    muni = muni.to_crs(TARGET_CRS)
    # Keep geometry + stable id for overlays; drop accessibility attrs
    keep = [c for c in hexes.columns if c in {"hex_id", "id", "geometry"} or c.startswith("row") or c.startswith("col")]
    if "hex_id" not in hexes.columns:
        hexes = hexes.reset_index(drop=True)
        hexes["hex_id"] = [f"OD_ENV_V1_{i:04d}" for i in range(len(hexes))]
    else:
        # ensure unique working ids for overlay (prefixes differ across indicators)
        hexes = hexes.copy()
        hexes["hex_id"] = [f"OD_ENV_V1_{i:04d}" for i in range(len(hexes))]
    hexes["cell_area_m2"] = hexes.geometry.area
    print(f"Using walking-grid hexes n={len(hexes)}")
    return hexes[["hex_id", "cell_area_m2", "geometry"]], muni


def overlay_share(hexes: gpd.GeoDataFrame, layer: gpd.GeoDataFrame, prefix: str) -> gpd.GeoDataFrame:
    layer = layer.to_crs(hexes.crs)
    if layer.empty:
        hexes[f"{prefix}_m2"] = 0.0
        hexes[f"{prefix}_share"] = 0.0
        hexes[f"{prefix}_pct"] = 0.0
        return hexes
    # dissolve to one geometry for speed when many parts
    if len(layer) > 1:
        geom = layer.unary_union
        layer = gpd.GeoDataFrame(geometry=[geom], crs=hexes.crs)
    inter = gpd.overlay(hexes[["hex_id", "geometry"]], layer[["geometry"]], how="intersection", keep_geom_type=False)
    inter["part_m2"] = inter.geometry.area
    sums = inter.groupby("hex_id")["part_m2"].sum()
    hexes[f"{prefix}_m2"] = hexes["hex_id"].map(sums).fillna(0.0)
    hexes[f"{prefix}_share"] = (hexes[f"{prefix}_m2"] / hexes["cell_area_m2"]).clip(0, 1)
    hexes[f"{prefix}_pct"] = hexes[f"{prefix}_share"] * 100.0
    return hexes


def road_buffer_share(hexes: gpd.GeoDataFrame, roads: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    roads = roads.to_crs(hexes.crs)
    # Keep major roads only (full vejmidte is ~63k segments)
    if "trafikart" in roads.columns:
        major = roads["trafikart"].astype(str).str.contains(
            "Motorvej|Motortrafikvej|Primærrute|Hovedrute|Sekundærrute", case=False, na=False
        )
        if "vejkategori" in roads.columns:
            major = major | roads["vejkategori"].astype(str).str.contains(
                "Motorvej|Hovedrute|Primær|Sekundær", case=False, na=False
            )
        roads = roads.loc[major]
    minx, miny, maxx, maxy = hexes.total_bounds
    roads = roads.cx[minx:maxx, miny:maxy]
    print(f"  major road segments for buffer: {len(roads)}")
    if roads.empty:
        hexes["road_buf_m2"] = 0.0
        hexes["road_buf_share"] = 0.0
        hexes["road_buf_pct"] = 0.0
        return hexes
    buf = roads.copy()
    buf["geometry"] = buf.buffer(ROAD_BUFFER_M)
    buf = gpd.GeoDataFrame(geometry=[buf.unary_union], crs=hexes.crs)
    return overlay_share(hexes, buf, "road_buf")


def zonal_raster_mean(hexes: gpd.GeoDataFrame, raster_path: Path, col: str) -> gpd.GeoDataFrame:
    hexes[col] = np.nan
    if not raster_path.exists():
        print(f"  skip missing raster {raster_path.name}")
        return hexes
    try:
        from rasterstats import zonal_stats
        import rasterio
    except ImportError:
        print(f"rasterstats/rasterio missing; skip {raster_path.name}")
        return hexes
    try:
        with rasterio.open(raster_path) as src:
            nodata = src.nodata if src.nodata is not None else -9999.0
            arr = src.read(1)
            affine = src.transform
        geoms = []
        for g in hexes.geometry:
            if g is None or g.is_empty:
                geoms.append(None)
            elif getattr(g, "geom_type", "") == "MultiPolygon" and len(g.geoms):
                geoms.append(max(g.geoms, key=lambda x: x.area))
            else:
                geoms.append(g)
        stats = zonal_stats(geoms, arr, affine=affine, stats=["mean"], nodata=nodata, all_touched=True)
        hexes[col] = [s["mean"] if s and s["mean"] is not None else np.nan for s in stats]
        print(f"  {col}: valid={int(hexes[col].notna().sum())} mean={hexes[col].mean()}")
    except Exception as e:
        print(f"  zonal failed for {raster_path.name}: {e}")
    return hexes


def minmax_norm(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.notna().sum() < 2 or s.max() == s.min():
        return pd.Series(np.nan, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def weighted_score(df: pd.DataFrame, weights: dict[str, float]) -> tuple[pd.Series, dict]:
    used = {}
    parts = []
    for col, w in weights.items():
        if col not in df.columns:
            continue
        v = df[col]
        if v.notna().sum() < 10 or v.nunique(dropna=True) < 2:
            continue
        used[col] = w
        parts.append(v.fillna(v.median()) * w)
    if not parts:
        return pd.Series(np.nan, index=df.index), {}
    total_w = sum(used.values())
    score = sum(parts) / total_w
    return score, {k: v / total_w for k, v in used.items()}


def plot_choropleth(hexes, muni, col, title, out_png, cmap="viridis"):
    try:
        if col not in hexes.columns or hexes[col].notna().sum() < 2:
            print(f"  skip plot {Path(out_png).name}: insufficient values")
            return
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        hexes.plot(column=col, cmap=cmap, linewidth=0.05, edgecolor="none", ax=ax, legend=True, legend_kwds={"shrink": 0.6})
        muni.boundary.plot(ax=ax, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        print(f"  wrote {Path(out_png).name}")
    except Exception as e:
        print(f"  plot failed {Path(out_png).name}: {e}")
        plt.close("all")


def main() -> None:
    print("Loading hex grid…")
    hexes, muni = load_hex_muni()
    print(f"n_hex={len(hexes)}")

    # --- Official land-cover shares ---
    natural = gpd.read_file(OFFICIAL / "official_landcover_natural.gpkg")
    blue = gpd.read_file(OFFICIAL / "official_blue.gpkg")
    built = gpd.read_file(OFFICIAL / "official_builtup.gpkg")
    roads = gpd.read_file(OFFICIAL / "official_roads.gpkg")

    print("Overlay natural…")
    hexes = overlay_share(hexes, natural, "green")  # use as green for Stage B green_share_pct path too
    hexes = overlay_share(hexes, natural, "official_natural")
    print("Overlay blue…")
    hexes = overlay_share(hexes, blue, "blue")
    print("Overlay built-up…")
    hexes = overlay_share(hexes, built, "builtup")
    print("Road buffers…")
    hexes = road_buffer_share(hexes, roads)

    # Protected / eco
    eco_path = OFFICIAL / "protected_nature.gpkg"
    if not eco_path.exists():
        eco_path = OFFICIAL / "ecological_health_vector.gpkg"
    if eco_path.exists():
        print("Overlay protected nature…")
        hexes = overlay_share(hexes, gpd.read_file(eco_path), "eco")
    else:
        hexes["eco_m2"] = 0.0
        hexes["eco_share"] = 0.0
        hexes["eco_pct"] = 0.0

    if (OFFICIAL / "noise_vector.gpkg").exists():
        print("Overlay noise vector…")
        hexes = overlay_share(hexes, gpd.read_file(OFFICIAL / "noise_vector.gpkg"), "noise")
    else:
        hexes["noise_m2"] = 0.0
        hexes["noise_share"] = 0.0
        hexes["noise_pct"] = 0.0

    # Rasters
    print("Zonal rasters…")
    hexes = zonal_raster_mean(hexes, OFFICIAL / "heat_raster.tif", "heat_mean")
    hexes = zonal_raster_mean(hexes, OFFICIAL / "noise_raster.tif", "noise_mean")
    hexes = zonal_raster_mean(hexes, OFFICIAL / "air_quality_raster.tif", "air_pollution_mean")

    # Norms for composite
    hexes["green_share_norm"] = minmax_norm(hexes["green_share"])
    hexes["blue_share_norm"] = minmax_norm(hexes["blue_share"])
    hexes["builtup_share_norm"] = minmax_norm(hexes["builtup_share"])
    hexes["builtup_inverse_norm"] = 1 - hexes["builtup_share_norm"]
    hexes["road_burden_norm"] = minmax_norm(hexes["road_buf_share"])
    hexes["road_burden_inverse_norm"] = 1 - hexes["road_burden_norm"]
    hexes["official_natural_share_norm"] = minmax_norm(hexes["official_natural_share"])
    hexes["eco_share_norm"] = minmax_norm(hexes["eco_share"])
    hexes["noise_share_norm"] = minmax_norm(hexes["noise_share"])
    hexes["noise_share_inverse_norm"] = 1 - hexes["noise_share_norm"]
    hexes["low_green_norm"] = 1 - hexes["green_share_norm"]
    hexes["low_blue_norm"] = 1 - hexes["blue_share_norm"]
    hexes["low_official_natural_norm"] = 1 - hexes["official_natural_share_norm"]
    hexes["low_eco_share_norm"] = 1 - hexes["eco_share_norm"]
    hexes["heat_mean_norm"] = minmax_norm(hexes["heat_mean"])
    hexes["heat_mean_inverse_norm"] = 1 - hexes["heat_mean_norm"]
    hexes["noise_mean_norm"] = minmax_norm(hexes["noise_mean"])
    hexes["noise_mean_inverse_norm"] = 1 - hexes["noise_mean_norm"]
    hexes["air_pollution_mean_norm"] = minmax_norm(hexes["air_pollution_mean"])
    hexes["air_pollution_mean_inverse_norm"] = 1 - hexes["air_pollution_mean_norm"]

    quality_w = {
        "green_share_norm": 0.18,
        "blue_share_norm": 0.08,
        "builtup_inverse_norm": 0.12,
        "road_burden_inverse_norm": 0.10,
        "official_natural_share_norm": 0.15,
        "eco_share_norm": 0.12,
        "noise_share_inverse_norm": 0.08,
        "heat_mean_inverse_norm": 0.09,
        "noise_mean_inverse_norm": 0.08,
    }
    burden_w = {
        "low_green_norm": 0.12,
        "builtup_share_norm": 0.18,
        "road_burden_norm": 0.15,
        "low_official_natural_norm": 0.10,
        "low_eco_share_norm": 0.10,
        "noise_share_norm": 0.10,
        "heat_mean_norm": 0.12,
        "noise_mean_norm": 0.08,
        "air_pollution_mean_norm": 0.05,
    }
    hexes["env_quality_raw"], qw = weighted_score(hexes, quality_w)
    hexes["env_burden_raw"], bw = weighted_score(hexes, burden_w)
    hexes["env_quality_score"] = minmax_norm(hexes["env_quality_raw"]) * 100
    hexes["env_burden_score"] = minmax_norm(hexes["env_burden_raw"]) * 100
    # Aliases used elsewhere
    hexes["green_share_pct"] = hexes["green_pct"]

    # --- Write green share package ---
    green_dir = SCRIPT_DIR / "odense_osm_green_area_share_outputs"
    green_dir.mkdir(exist_ok=True)
    green_gpkg = green_dir / "odense_green_area_share_500m.gpkg"
    if green_gpkg.exists():
        green_gpkg.unlink()
    green_out = hexes[["hex_id", "cell_area_m2", "green_m2", "green_share", "green_share_pct", "green_pct", "geometry"]].copy()
    green_out["green_id"] = green_out["hex_id"]
    green_out.to_file(green_gpkg, layer="green_area_share_500m_hexagons", driver="GPKG")
    muni.to_file(green_gpkg, layer="odense_municipality_districts", driver="GPKG")
    natural.to_crs(TARGET_CRS).to_file(green_gpkg, layer="official_natural_features", driver="GPKG")
    plot_choropleth(hexes, muni, "green_share_pct", "Green-area share (GeoDanmark natural) %", green_dir / "odense_green_area_share_500m_map.png", "YlGn")
    (green_dir / "green_area_share_validation.txt").write_text(
        f"v1 official green\nn={len(hexes)}\nmean_green_pct={hexes['green_share_pct'].mean():.2f}\nsource=official_landcover_natural.gpkg\n"
    )

    # --- Write environmental package ---
    env_dir = SCRIPT_DIR / "odense_multisource_environmental_index_outputs"
    env_dir.mkdir(exist_ok=True)
    env_gpkg = env_dir / "odense_multisource_environmental_index_500m.gpkg"
    if env_gpkg.exists():
        env_gpkg.unlink()
    hexes.to_file(env_gpkg, layer="environmental_index_500m_hexagons", driver="GPKG")
    muni.to_file(env_gpkg, layer="odense_municipality_districts", driver="GPKG")
    natural.to_crs(TARGET_CRS).to_file(env_gpkg, layer="official_natural_features", driver="GPKG")
    blue.to_crs(TARGET_CRS).to_file(env_gpkg, layer="official_blue_features", driver="GPKG")
    built.to_crs(TARGET_CRS).to_file(env_gpkg, layer="official_builtup_features", driver="GPKG")

    plot_choropleth(hexes, muni, "env_quality_score", "Environmental quality (official v1)", env_dir / "odense_environmental_quality_index_500m_map.png", "YlGnBu")
    plot_choropleth(hexes, muni, "env_burden_score", "Environmental burden (official v1)", env_dir / "odense_environmental_burden_index_500m_map.png", "YlOrRd")
    plot_choropleth(hexes, muni, "builtup_pct", "Built-up share % (GeoDanmark)", env_dir / "odense_builtup_share_500m_map.png", "Greys")

    (env_dir / "multisource_environmental_index_validation.txt").write_text(
        "\n".join(
            [
                "Official-first environmental index v1",
                f"n_hex={len(hexes)}",
                f"mean_green_pct={hexes['green_pct'].mean():.2f}",
                f"mean_blue_pct={hexes['blue_pct'].mean():.2f}",
                f"mean_builtup_pct={hexes['builtup_pct'].mean():.2f}",
                f"mean_road_buf_pct={hexes['road_buf_pct'].mean():.2f}",
                f"mean_eco_pct={hexes['eco_pct'].mean():.2f}",
                f"quality_weights={qw}",
                f"burden_weights={bw}",
                "sources: official_landcover_natural, official_blue, official_builtup, official_roads,",
                "protected_nature/ecological_health_vector, noise_vector, heat/noise/air rasters",
                "air_quality_raster is interim Open-Meteo/CAMS proxy pending DCE",
            ]
        )
    )
    print("Done. Green + env written.")
    print(f"green mean%={hexes['green_share_pct'].mean():.2f} builtup%={hexes['builtup_pct'].mean():.2f}")


if __name__ == "__main__":
    main()
