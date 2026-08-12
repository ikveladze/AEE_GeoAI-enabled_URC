#!/usr/bin/env python3
"""Build hex-ready demographic source grid from Map Layers/Demographic.

Priority (total population only):
  1. DST Kvadratnet extranet 1 km population-density interval cells
     (dst_kvadratnet_extranet/odense_dst_kvadratnet_1km_2019.geojson
      or km1bef.js filtered to kommune 461)
  2. Fallback: odense_1km_population_density_classes.gpkg / StatBank-style class map

Control total:
  KOMM_TOT_21 = 205,987 (Basisdata_o / README extract), used to scale class midpoints.

Excluded (not total-population hex drivers):
  - nonwestern / ikkevestlig / turkey (migration subset)
  - demographic_grid.gpkg (invalid inflated pop_total)
  - OSM place population points (place tags, not a grid)
  - StatBank municipality/postcode/parish tables (no hex geometry here; used for validation only)
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import shape

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEMO_DIR = PROJECT_DIR / "Map Layers" / "Demographic"
OUT_GPKG = DEMO_DIR / "dst_kvadratnet_population.gpkg"
OUT_CSV = DEMO_DIR / "odense_basisdata_o" / "odense_1km_pop_estimate_v1.csv"
OUT_REPORT = DEMO_DIR / "odense_basisdata_o" / "demographic_source_v1_validation.txt"

EXTRANET_DIR = DEMO_DIR / "dst_kvadratnet_extranet"
EXTRANET_GEOJSON = EXTRANET_DIR / "odense_dst_kvadratnet_1km_2019.geojson"
EXTRANET_JS = EXTRANET_DIR / "km1bef.js"
DENSITY_CLASS_GPKG = DEMO_DIR / "odense_basisdata_o" / "odense_1km_population_density_classes.gpkg"
README_JSON = DEMO_DIR / "odense_basisdata_o" / "README_odense_demographics_extract.json"
STATBANK_SUMMARY = DEMO_DIR / "statbank_odense" / "odense_demographic_summary.json"

# DST extranet legend (grid.js): value 1..6 → persons/km² class
VALUE_TO_CLASS = {
    1: "1-19",
    2: "20-49",
    3: "50-99",
    4: "100-149",
    5: "150-399",
    6: "400 >",
}
CLASS_MID = {
    "1-19": 10.0,
    "20-49": 34.5,
    "50-99": 74.5,
    "100-149": 124.5,
    "150-399": 274.5,
    "400 >": 500.0,
    "400-": 500.0,
    "400>": 500.0,
}


def municipality_control_total() -> tuple[float, str]:
    if README_JSON.exists():
        meta = json.loads(README_JSON.read_text(encoding="utf-8"))
        tot = float(meta.get("municipality_KOMM_TOT_21", {}).get("pop_total", 205_987))
        return tot, "KOMM_TOT_21 via README_odense_demographics_extract.json"
    return 205_987.0, "default KOMM_TOT_21"


def class_to_mid(val) -> float:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return np.nan
    s = str(val).strip()
    return CLASS_MID.get(s, CLASS_MID.get(s.replace("400-", "400 >"), np.nan))


def load_from_extranet_geojson() -> gpd.GeoDataFrame | None:
    if not EXTRANET_GEOJSON.exists():
        return None
    g = gpd.read_file(EXTRANET_GEOJSON).to_crs("EPSG:25832")
    g["source"] = g.get(
        "source",
        "DST Kvadratnet extranet (odense_dst_kvadratnet_1km_2019.geojson)",
    )
    return g


def load_from_km1bef_js() -> gpd.GeoDataFrame | None:
    if not EXTRANET_JS.exists():
        return None
    text = EXTRANET_JS.read_text(encoding="utf-8", errors="replace")
    text = text.split("=", 1)[1].strip().rstrip(";")
    text = re.sub(r",\s*([}\]])", r"\1", text)  # trailing commas
    obj = json.loads(text)
    rows = []
    for f in obj["features"]:
        props = f.get("properties", {})
        if str(props.get("kom")) != "461":
            continue
        val = int(props.get("value"))
        code = str(props.get("code"))
        cls = VALUE_TO_CLASS[val]
        geom = shape(f["geometry"])
        rows.append(
            {
                "code": code,
                "kom": "461",
                "interval_group": val,
                "kvadrat_id": f"1km_{code}",
                "pop_density_class_per_km2": cls,
                "dens_mid_per_km2": CLASS_MID[cls],
                "main_district": "Odense",
                "geometry": geom,
            }
        )
    if not rows:
        return None
    g = gpd.GeoDataFrame(rows, crs="EPSG:25832")
    g["source"] = "DST Kvadratnet extranet km1bef.js (Befolkning 1. jan. 2019 interval groups)"
    g["source_url"] = "https://www.dst.dk/websites/extranet/kvadratnet/index.html"
    return g


def load_from_density_classes() -> gpd.GeoDataFrame:
    dens = gpd.read_file(DENSITY_CLASS_GPKG).to_crs("EPSG:25832")
    dens_col = "pop_density_class_per_km2"
    dens["dens_mid_per_km2"] = dens[dens_col].map(class_to_mid)
    if "kvadrat_id" in dens.columns:
        dens = dens.sort_values("dens_mid_per_km2", ascending=False).drop_duplicates("kvadrat_id", keep="first")
    dens["source"] = "odense_1km_population_density_classes.gpkg"
    dens["main_district"] = dens.get("main_district", "Odense")
    return dens


def main() -> None:
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    (DEMO_DIR / "odense_basisdata_o").mkdir(parents=True, exist_ok=True)

    control_total, control_src = municipality_control_total()
    statbank_note = ""
    if STATBANK_SUMMARY.exists():
        sb = json.loads(STATBANK_SUMMARY.read_text(encoding="utf-8"))
        statbank_note = (
            f"StatBank FOLK1A context: pop={sb.get('population_total')} "
            f"({sb.get('period_population')}); age_0_14={sb.get('age_0_14')}; "
            f"age_65plus={sb.get('age_65_plus')} — municipality scale only, not mapped to hex."
        )

    dens = load_from_extranet_geojson()
    build_path = "extranet_geojson"
    if dens is None or dens.empty:
        dens = load_from_km1bef_js()
        build_path = "km1bef_js"
    if dens is None or dens.empty:
        dens = load_from_density_classes()
        build_path = "density_classes_fallback"

    dens = dens.to_crs("EPSG:25832")
    if "pop_density_class_per_km2" not in dens.columns and "interval_group" in dens.columns:
        dens["pop_density_class_per_km2"] = dens["interval_group"].map(VALUE_TO_CLASS)
    if "dens_mid_per_km2" not in dens.columns:
        dens["dens_mid_per_km2"] = dens["pop_density_class_per_km2"].map(class_to_mid)

    dens["cell_area_km2"] = dens.geometry.area / 1e6
    # DST 1 km squares are ~1 km²; use class midpoint as persons/km² × area
    dens["pop_raw"] = dens["dens_mid_per_km2"].fillna(0) * dens["cell_area_km2"]

    if "kvadrat_id" in dens.columns:
        dens = dens.sort_values("pop_raw", ascending=False).drop_duplicates("kvadrat_id", keep="first")

    raw_sum = float(dens["pop_raw"].sum())
    scale = control_total / raw_sum if raw_sum > 0 else 1.0
    dens["pop_total"] = dens["pop_raw"] * scale
    dens["age_0_14"] = np.nan
    dens["age_65plus"] = np.nan

    # Drop any migration fields if present
    drop_mig = [
        c
        for c in dens.columns
        if any(k in c.lower() for k in ("nonwestern", "ikkevest", "turkey", "migrant", "immigrant"))
    ]
    dens = dens.drop(columns=drop_mig, errors="ignore")

    wanted = [
        "kvadrat_id",
        "main_district",
        "pop_density_class_per_km2",
        "dens_mid_per_km2",
        "pop_total",
        "age_0_14",
        "age_65plus",
        "interval_group",
        "code",
        "kom",
        "source",
        "source_url",
        "geometry",
    ]
    keep = [c for c in wanted if c in dens.columns]
    out = dens[keep].copy()
    out = out.loc[:, ~out.columns.duplicated()]

    if OUT_GPKG.exists():
        OUT_GPKG.unlink()
    out.to_file(OUT_GPKG, layer="population_1km", driver="GPKG")
    pd.DataFrame(out.drop(columns="geometry")).to_csv(OUT_CSV, index=False)

    report = [
        "Demographic source grid (total population) — rebuilt from Map Layers/Demographic",
        f"build_path={build_path}",
        f"cells={len(out)}",
        f"raw_pop_sum={raw_sum:,.1f}",
        f"control_total={control_total:,.0f} ({control_src})",
        f"scale={scale:.6f}",
        f"scaled_pop_sum={out['pop_total'].sum():,.1f}",
        f"density_class_counts={out['pop_density_class_per_km2'].value_counts().to_dict()}",
        "USED:",
        f"  - DST Kvadratnet 1 km density intervals ({build_path})",
        f"  - Municipal control total {control_total:,.0f}",
        "NOT USED for hex population:",
        "  - demographic_grid.gpkg (invalid inflated counts)",
        "  - Odense_PopDencity_Points / osm place points (place tags)",
        "  - nonwestern / ikkevestlig / turkey (migration subset)",
        "  - StatBank / OMRP age-sex tables (no reliable hex crosswalk in folder)",
        statbank_note,
        f"written={OUT_GPKG}",
    ]
    OUT_REPORT.write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report))


if __name__ == "__main__":
    main()
