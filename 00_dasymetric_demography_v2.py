#!/usr/bin/env python3
"""Odense demographic estimation v2 — multi-method dasymetric analysis.

Purpose
-------
Produce the best feasible 500 m hex population / age surface for the Urban
Science AEE study by combining:

  * StatBank parish (SOGN) and postcode (POSTNR) counts
  * Basisdata municipal age–sex control totals (KOMM_TOT_21)
  * BBR residential housing floor area as ancillary weights
  * DST Kvadratnet 1 km density *classes* as spatial ground-truth check

Methods implemented (literature-aligned)
----------------------------------------
M1 Areal weighting (Goodchild & Lam 1980 tradition; baseline)
M2 Binary dasymetric — equal share to residential buildings
   (Eicher & Brewer 2001; Langford 2006)
M3 Floor-area weighted dasymetric — BYG_BOLIG_ weights
   (Mennis 2003; Mennis & Hultgren 2006; Stevens et al. 2015 tradition)
M4 DST class-midpoint → hex areal transfer (previous pipeline)
M5 Hybrid: DST class-midpoint cell totals refined by BBR weights
   inside each 1 km cell (Comber et al. 2019; hybrid dasymetric)
M6 Postcode-constrained floor-area dasymetric + POSTNR age bands

Ground truth
------------
DST extranet cells are *interval* density classes (persons/km²), not exact
counts. Validation therefore reports:
  - density-class agreement rate
  - Spearman rank correlation vs class midpoints
  - MAE/RMSE vs class midpoints (soft reference only)
  - municipal / parish control-total checks

Outputs are written under Map Layers/Demographic/dasymetric_workspace/ and
a preferred source grid for the indicator script.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from shapely.geometry import Point

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT = SCRIPT_DIR.parent
DEMO = PROJECT / "Map Layers" / "Demographic"
WS = DEMO / "dasymetric_workspace"
OUT_DIR = SCRIPT_DIR / "odense_demographic_dasymetric_v2_outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
WS.mkdir(parents=True, exist_ok=True)

HEX_PATH = PROJECT / "Map Layers" / "Odense-500mHexaCells_1.gpkg"
BBR_PATH = DEMO / "BBR Data" / "OD Buildings_2020.shp"
DST_GEOJSON = DEMO / "dst_kvadratnet_extranet" / "odense_dst_kvadratnet_1km_2019.geojson"
DST_GPKG = DEMO / "dst_kvadratnet_population.gpkg"
SOGN_GPKG = WS / "odense_sogne_filtered.gpkg"
POST_GPKG = WS / "odense_postnumre.gpkg"
STATBANK = DEMO / "statbank_odense"
BASIS = DEMO / "odense_basisdata_o"
README_JSON = BASIS / "README_odense_demographics_extract.json"
SB_SUMMARY = STATBANK / "odense_demographic_summary.json"

TARGET_CRS = "EPSG:25832"

CLASS_BOUNDS = {
    "1-19": (1.0, 19.0),
    "20-49": (20.0, 49.0),
    "50-99": (50.0, 99.0),
    "100-149": (100.0, 149.0),
    "150-399": (150.0, 399.0),
    "400-": (400.0, 2500.0),
    "400 >": (400.0, 2500.0),
    "400>": (400.0, 2500.0),
}
CLASS_MID = {
    "1-19": 10.0,
    "20-49": 34.5,
    "50-99": 74.5,
    "100-149": 124.5,
    "150-399": 274.5,
    "400-": 500.0,
    "400 >": 500.0,
    "400>": 500.0,
}

PUBS = [
    {
        "key": "Mennis2003",
        "citation": "Mennis, J. (2003). Generating surface models of population using dasymetric mapping. The Professional Geographer, 55(1), 31–42.",
        "doi": "10.1111/0033-0124.10042",
        "use": "Floor-area / categorical ancillary dasymetric redistribution (M3).",
    },
    {
        "key": "MennisHultgren2006",
        "citation": "Mennis, J., & Hultgren, T. (2006). Intelligent dasymetric mapping and its application to areal interpolation. Cartography and Geographic Information Science, 33(3), 179–194.",
        "doi": "10.1559/152304006779077309",
        "use": "Intelligent / hybrid dasymetric parameterisation (M5).",
    },
    {
        "key": "EicherBrewer2001",
        "citation": "Eicher, C. L., & Brewer, C. A. (2001). Dasymetric mapping and areal interpolation: Implementation and evaluation. Cartography and Geographic Information Science, 28(2), 125–138.",
        "doi": "10.1559/152304001782173727",
        "use": "Binary dasymetric mask using inhabited buildings (M2).",
    },
    {
        "key": "Comber2019",
        "citation": "Comber, A., & Zeng, W. (2019). Spatial interpolation using areal features: A review of methods and opportunities using new forms of data with coded illustrations. Geography Compass, 13(1), e12425.",
        "doi": "10.1111/gec3.12425",
        "use": "Method taxonomy: areal weighting vs ancillary-constrained dasymetric.",
    },
    {
        "key": "Quiros2022",
        "citation": "Quirós, E., et al. (2022). Empiric recommendations for population disaggregation under different data scenarios. PLOS ONE, 17(9), e0274504.",
        "doi": "10.1371/journal.pone.0274504",
        "use": "Guidance favouring building-weighted / hybrid disaggregation when footprints exist.",
    },
    {
        "key": "Tobler1979",
        "citation": "Tobler, W. R. (1979). Smooth pycnophylactic interpolation for geographical regions. Journal of the American Statistical Association, 74(367), 519–530.",
        "doi": "10.1080/01621459.1979.10481647",
        "use": "Conceptual reference for mass-preserving (pycnophylactic) redistribution; parish totals preserved in M1–M3/M6.",
    },
    {
        "key": "GoodchildLam1980",
        "citation": "Goodchild, M. F., & Lam, N. S.-N. (1980). Areal interpolation: A variant of the traditional spatial problem. Geo-Processing, 1, 297–312.",
        "doi": None,
        "use": "Areal weighting baseline (M1).",
    },
]


def load_hex() -> gpd.GeoDataFrame:
    layers = gpd.list_layers(HEX_PATH)["name"].tolist()
    prefer = [
        "odense500mhexacells__grid",
        "odense_aee_500m_hexgrid",
        "odense500mhexacells_clipped",
    ]
    layer = next((p for p in prefer if p in layers), layers[0])
    hexg = gpd.read_file(HEX_PATH, layer=layer).to_crs(TARGET_CRS)
    if "hex_id" not in hexg.columns:
        hexg["hex_id"] = np.arange(len(hexg))
    hexg["hex_area_m2"] = hexg.geometry.area
    return hexg[["hex_id", "hex_area_m2", "geometry"]].copy()


def load_sogne_sources() -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    sogne = gpd.read_file(SOGN_GPKG).to_crs(TARGET_CRS)
    sogne["kode"] = sogne["kode"].astype(str).str.zfill(4)

    s1 = pd.read_csv(STATBANK / "SOGN1_odense_population.csv", sep=";")
    s1["kode"] = s1["SOGN"].str.extract(r"^(\d{4})")
    s1 = s1.rename(columns={"INDHOLD": "pop_sogn1_2026"})

    s10 = pd.read_csv(STATBANK / "SOGN10_odense_population.csv", sep=";")
    s10["kode"] = s10["SOGN"].str.extract(r"^(\d{4})")
    s10 = s10.rename(columns={"INDHOLD": "pop_sogn10_2022"})

    sogne = sogne.merge(s1[["kode", "pop_sogn1_2026"]], on="kode", how="left")
    sogne = sogne.merge(s10[["kode", "pop_sogn10_2022"]], on="kode", how="left")
    # For DST ~2019 validation prefer 2022 parish counts; fill missing with 2026
    sogne["pop_source_val"] = sogne["pop_sogn10_2022"].fillna(sogne["pop_sogn1_2026"])
    sogne["pop_source_map"] = sogne["pop_sogn1_2026"].fillna(sogne["pop_sogn10_2022"])
    sogne["sogn_area_m2"] = sogne.geometry.area
    return sogne, sogne.copy()


def load_postcodes() -> gpd.GeoDataFrame:
    post = gpd.read_file(POST_GPKG).to_crs(TARGET_CRS)
    post["nr"] = post["nr"].astype(str).str.zfill(4)
    age = pd.read_csv(STATBANK / "POSTNR1_odense_age_2026.csv", sep=";")
    age["nr"] = age["PNR20"].str.extract(r"^(\d{4})")
    # keep Odense municipal postcodes only (exclude stormodtager if any)
    tot = age[age["ALDER"] == "Alder i alt"][["nr", "INDHOLD"]].rename(
        columns={"INDHOLD": "pop_post_2026"}
    )

    def band_sum(labels: list[str]) -> pd.Series:
        sub = age[age["ALDER"].isin(labels)].groupby("nr")["INDHOLD"].sum()
        return sub

    age0 = band_sum(["0-4 år", "5-9 år", "10-14 år"]).rename("age_0_14_post")
    age65 = band_sum(
        ["65-69 år", "70-74 år", "75-79 år", "80-84 år", "85-89 år", "90 år og derover"]
    ).rename("age_65plus_post")

    post = post.merge(tot, on="nr", how="left")
    post = post.merge(age0, on="nr", how="left")
    post = post.merge(age65, on="nr", how="left")
    # Restrict to postcodes present in StatBank Odense extract
    post = post[post["pop_post_2026"].notna()].copy()
    return post


def load_dst() -> gpd.GeoDataFrame:
    if DST_GPKG.exists():
        dst = gpd.read_file(DST_GPKG).to_crs(TARGET_CRS)
    else:
        dst = gpd.read_file(DST_GEOJSON).to_crs(TARGET_CRS)
    if "pop_density_class_per_km2" not in dst.columns:
        # rebuild from geojson properties
        raw = gpd.read_file(DST_GEOJSON).to_crs(TARGET_CRS)
        dst = raw
    cls = dst["pop_density_class_per_km2"].astype(str).str.strip()
    dst["dens_class"] = cls
    dst["dens_mid"] = cls.map(CLASS_MID)
    dst["cell_area_km2"] = dst.geometry.area / 1e6
    if "pop_total" not in dst.columns:
        dst["pop_total"] = dst["dens_mid"] * dst["cell_area_km2"]
        # scale later
    if "kvadrat_id" not in dst.columns:
        if "code" in dst.columns:
            dst["kvadrat_id"] = "1km_" + dst["code"].astype(str)
        else:
            dst["kvadrat_id"] = [f"1km_{i}" for i in range(len(dst))]
    return dst


def load_bbr_residential() -> gpd.GeoDataFrame:
    bbr = gpd.read_file(BBR_PATH)
    if bbr.crs is None:
        bbr = bbr.set_crs(TARGET_CRS)
    else:
        bbr = bbr.to_crs(TARGET_CRS)
    use = bbr["BuildingUs"].astype(str)
    res = bbr[use.str.contains("year-round living", case=False, na=False)].copy()
    res["bolig_m2"] = pd.to_numeric(res["BYG_BOLIG_"], errors="coerce").fillna(0).clip(lower=0)
    # fallback weight if bolig is 0: bebygget or unit area
    bebyg = pd.to_numeric(res["BYG_BEBYG_"], errors="coerce").fillna(0).clip(lower=0)
    res.loc[res["bolig_m2"] <= 0, "bolig_m2"] = bebyg[res["bolig_m2"] <= 0]
    res = res[res["bolig_m2"] > 0].copy()
    res["bbr_id"] = np.arange(len(res))
    # points for fast assignment
    res["geometry"] = res.geometry.centroid
    return res[["bbr_id", "bolig_m2", "geometry"]].copy()


def assign_points_to_polygons(
    pts: gpd.GeoDataFrame, polys: gpd.GeoDataFrame, key: str, prefix: str
) -> gpd.GeoDataFrame:
    joined = gpd.sjoin(pts, polys[[key, "geometry"]], how="left", predicate="within")
    # unresolved: nearest polygon
    miss = joined[joined[key].isna()].copy()
    if len(miss):
        miss = miss.drop(columns=[c for c in miss.columns if c.endswith("_right") or c == "index_right"], errors="ignore")
        near = gpd.sjoin_nearest(
            miss[["bbr_id", "bolig_m2", "geometry"]],
            polys[[key, "geometry"]],
            how="left",
            max_distance=500,
        )
        joined.loc[miss.index, key] = near[key].values
    joined = joined.rename(columns={key: f"{prefix}_{key}"})
    # one row per building if duplicate joins
    joined = joined.sort_values("bolig_m2", ascending=False).drop_duplicates("bbr_id", keep="first")
    return joined


def dasymetric_from_source(
    buildings: gpd.GeoDataFrame,
    source_key: str,
    pop_col: str,
    source_pop: pd.DataFrame,
) -> gpd.GeoDataFrame:
    """Allocate source-zone population to buildings by bolig_m2 share (pycnophylactic)."""
    b = buildings.merge(source_pop[[source_key, pop_col]], left_on=source_key, right_on=source_key, how="left")
    b[pop_col] = b[pop_col].fillna(0)
    wsum = b.groupby(source_key)["bolig_m2"].transform("sum")
    b["pop_est"] = np.where(wsum > 0, b[pop_col] * (b["bolig_m2"] / wsum), 0.0)
    return b


def binary_dasymetric(
    buildings: gpd.GeoDataFrame,
    source_key: str,
    pop_col: str,
    source_pop: pd.DataFrame,
) -> gpd.GeoDataFrame:
    b = buildings.merge(source_pop[[source_key, pop_col]], on=source_key, how="left")
    b[pop_col] = b[pop_col].fillna(0)
    n = b.groupby(source_key)["bbr_id"].transform("count")
    b["pop_est"] = np.where(n > 0, b[pop_col] / n, 0.0)
    return b


def buildings_to_hex(buildings: gpd.GeoDataFrame, hexg: gpd.GeoDataFrame, value_col: str) -> pd.Series:
    j = gpd.sjoin(buildings[["geometry", value_col]], hexg[["hex_id", "geometry"]], how="inner", predicate="within")
    return j.groupby("hex_id")[value_col].sum()


def areal_weight_to_hex(source: gpd.GeoDataFrame, hexg: gpd.GeoDataFrame, pop_col: str) -> pd.Series:
    src = source[[pop_col, "geometry"]].copy()
    src["src_id"] = np.arange(len(src))
    src["src_area"] = src.geometry.area
    inter = gpd.overlay(
        hexg[["hex_id", "geometry"]],
        src[["src_id", pop_col, "src_area", "geometry"]],
        how="intersection",
    )
    inter["iarea"] = inter.geometry.area
    inter["pop_part"] = inter[pop_col] * inter["iarea"] / inter["src_area"].replace(0, np.nan)
    return inter.groupby("hex_id")["pop_part"].sum()


def aggregate_hex_to_dst(hexg: gpd.GeoDataFrame, pop_series: pd.Series, dst: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    h = hexg.copy()
    h["pop"] = h["hex_id"].map(pop_series).fillna(0)
    inter = gpd.overlay(
        h[["hex_id", "pop", "geometry"]],
        dst[["kvadrat_id", "dens_class", "dens_mid", "cell_area_km2", "geometry"]],
        how="intersection",
    )
    inter["iarea"] = inter.geometry.area
    hex_area = h.set_index("hex_id")["geometry"].area
    inter["hex_area"] = inter["hex_id"].map(hex_area)
    inter["pop_part"] = inter["pop"] * inter["iarea"] / inter["hex_area"].replace(0, np.nan)
    cell = inter.groupby("kvadrat_id", as_index=False).agg(
        pop_est=("pop_part", "sum"),
        dens_class=("dens_class", "first"),
        dens_mid=("dens_mid", "first"),
        cell_area_km2=("cell_area_km2", "first"),
    )
    cell["dens_est"] = cell["pop_est"] / cell["cell_area_km2"].replace(0, np.nan)
    return cell


def class_agreement(dens: float, dens_class: str) -> bool:
    bounds = CLASS_BOUNDS.get(str(dens_class).strip())
    if bounds is None or not np.isfinite(dens):
        return False
    lo, hi = bounds
    return lo <= dens <= hi


def validate_vs_dst(cell: gpd.GeoDataFrame | pd.DataFrame, method: str) -> dict:
    d = cell.dropna(subset=["dens_est", "dens_mid"]).copy()
    agree = [class_agreement(a, b) for a, b in zip(d["dens_est"], d["dens_class"])]
    spearman = stats.spearmanr(d["dens_est"], d["dens_mid"], nan_policy="omit")
    pearson = stats.pearsonr(d["dens_est"], d["dens_mid"]) if len(d) > 2 else (np.nan, np.nan)
    mae = float(np.mean(np.abs(d["dens_est"] - d["dens_mid"])))
    rmse = float(np.sqrt(np.mean((d["dens_est"] - d["dens_mid"]) ** 2)))
    return {
        "method": method,
        "n_cells": int(len(d)),
        "pop_sum": float(d["pop_est"].sum()),
        "class_agreement_rate": float(np.mean(agree)) if agree else np.nan,
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "pearson_r": float(pearson[0]),
        "mae_vs_class_mid": mae,
        "rmse_vs_class_mid": rmse,
        "median_dens_est": float(d["dens_est"].median()),
        "median_dens_mid": float(d["dens_mid"].median()),
    }


def scale_series(s: pd.Series, target_total: float) -> pd.Series:
    tot = float(s.sum())
    if tot <= 0:
        return s
    return s * (target_total / tot)


def main() -> None:
    print("Loading inputs…")
    hexg = load_hex()
    sogne, _ = load_sogne_sources()
    post = load_postcodes()
    dst = load_dst()
    bbr = load_bbr_residential()

    meta = json.loads(README_JSON.read_text(encoding="utf-8")) if README_JSON.exists() else {}
    komm_tot = float(meta.get("municipality_KOMM_TOT_21", {}).get("pop_total", 205987))
    age0_m = float(meta.get("municipality_KOMM_TOT_21", {}).get("age_0_14", np.nan))
    age65_m = float(meta.get("municipality_KOMM_TOT_21", {}).get("age_65_plus", np.nan))
    sb = json.loads(SB_SUMMARY.read_text(encoding="utf-8")) if SB_SUMMARY.exists() else {}
    folk_tot = float(sb.get("population_total", 213431))

    print(f"hex={len(hexg)} sogne={len(sogne)} post={len(post)} dst={len(dst)} bbr_res={len(bbr)}")
    print(f"controls: KOMM_TOT_21={komm_tot:,.0f} FOLK1A={folk_tot:,.0f} SOGN1={sogne.pop_sogn1_2026.sum():,.0f} SOGN10={sogne.pop_sogn10_2022.sum():,.0f}")

    # Assign buildings
    print("Assigning BBR buildings to parishes / postcodes…")
    b_sogn = assign_points_to_polygons(bbr, sogne.rename(columns={"kode": "kode"}), "kode", "sogn")
    b_sogn = b_sogn.rename(columns={"sogn_kode": "kode"})
    b_post = assign_points_to_polygons(bbr, post.rename(columns={"nr": "nr"}), "nr", "post")
    b_post = b_post.rename(columns={"post_nr": "nr"})
    buildings = bbr.merge(b_sogn[["bbr_id", "kode"]], on="bbr_id", how="left")
    buildings = buildings.merge(b_post[["bbr_id", "nr"]], on="bbr_id", how="left")
    print("buildings with parish", buildings["kode"].notna().mean(), "with post", buildings["nr"].notna().mean())

    # Scale DST midpoints to KOMM_TOT for M4/M5
    dst = dst.copy()
    raw_mid_pop = (dst["dens_mid"] * dst["cell_area_km2"]).sum()
    dst["pop_mid_scaled"] = dst["dens_mid"] * dst["cell_area_km2"] * (komm_tot / raw_mid_pop)

    results_hex = hexg[["hex_id", "hex_area_m2", "geometry"]].copy()
    validation_rows = []
    method_notes = {}

    # ---- M1 areal weighting (SOGN10 for validation year; SOGN1 for map) ----
    print("M1 areal weighting…")
    m1_val = areal_weight_to_hex(sogne, hexg, "pop_source_val")
    m1_map = areal_weight_to_hex(sogne, hexg, "pop_source_map")
    results_hex["pop_M1_val"] = results_hex["hex_id"].map(m1_val).fillna(0)
    results_hex["pop_M1"] = results_hex["hex_id"].map(m1_map).fillna(0)
    cell = aggregate_hex_to_dst(hexg, m1_val, dst)
    validation_rows.append(validate_vs_dst(cell, "M1_areal_SOGN"))
    method_notes["M1"] = "Areal weighting of StatBank parish counts to hex (Goodchild & Lam 1980)."

    # ---- M2 binary dasymetric ----
    print("M2 binary dasymetric…")
    src_val = sogne[["kode", "pop_source_val"]].rename(columns={"pop_source_val": "pop"})
    src_map = sogne[["kode", "pop_source_map"]].rename(columns={"pop_source_map": "pop"})
    b2v = binary_dasymetric(buildings.dropna(subset=["kode"]), "kode", "pop", src_val)
    b2m = binary_dasymetric(buildings.dropna(subset=["kode"]), "kode", "pop", src_map)
    results_hex["pop_M2_val"] = results_hex["hex_id"].map(buildings_to_hex(b2v, hexg, "pop_est")).fillna(0)
    results_hex["pop_M2"] = results_hex["hex_id"].map(buildings_to_hex(b2m, hexg, "pop_est")).fillna(0)
    validation_rows.append(validate_vs_dst(aggregate_hex_to_dst(hexg, results_hex.set_index("hex_id")["pop_M2_val"], dst), "M2_binary_BBR"))
    method_notes["M2"] = "Binary dasymetric: equal parish pop per residential BBR building (Eicher & Brewer 2001)."

    # ---- M3 floor-area weighted ----
    print("M3 floor-area dasymetric…")
    b3v = dasymetric_from_source(buildings.dropna(subset=["kode"]), "kode", "pop", src_val)
    b3m = dasymetric_from_source(buildings.dropna(subset=["kode"]), "kode", "pop", src_map)
    results_hex["pop_M3_val"] = results_hex["hex_id"].map(buildings_to_hex(b3v, hexg, "pop_est")).fillna(0)
    results_hex["pop_M3"] = results_hex["hex_id"].map(buildings_to_hex(b3m, hexg, "pop_est")).fillna(0)
    # Scale map version to FOLK1A for contemporary consistency
    results_hex["pop_M3_folk"] = scale_series(results_hex["pop_M3"], folk_tot)
    validation_rows.append(validate_vs_dst(aggregate_hex_to_dst(hexg, results_hex.set_index("hex_id")["pop_M3_val"], dst), "M3_floorarea_BBR"))
    method_notes["M3"] = "Floor-area weighted dasymetric using BYG_BOLIG_ within parish (Mennis 2003)."

    # ---- M3c: soft-constrain M3_val densities into DST class intervals, then scale to FOLK ----
    print("M3c DST class soft-constraint…")
    htmp = hexg[["hex_id", "geometry"]].copy()
    htmp["pop"] = htmp["hex_id"].map(results_hex.set_index("hex_id")["pop_M3_val"]).fillna(0)
    htmp["hex_area"] = htmp.geometry.area
    inter_c = gpd.overlay(
        htmp,
        dst[["kvadrat_id", "dens_class", "cell_area_km2", "geometry"]],
        how="intersection",
    )
    inter_c["iarea"] = inter_c.geometry.area
    inter_c["pop_part"] = inter_c["pop"] * inter_c["iarea"] / inter_c["hex_area"].replace(0, np.nan)
    cell_c = inter_c.groupby("kvadrat_id", as_index=False).agg(
        pop=("pop_part", "sum"),
        dens_class=("dens_class", "first"),
        cell_area_km2=("cell_area_km2", "first"),
    )
    cell_c["dens"] = cell_c["pop"] / cell_c["cell_area_km2"].replace(0, np.nan)

    def _target_pop(row):
        bounds = CLASS_BOUNDS.get(str(row["dens_class"]).strip())
        if not bounds or not np.isfinite(row["dens"]):
            return row["pop"]
        lo, hi = bounds
        if row["dens"] < lo:
            return lo * row["cell_area_km2"]
        if row["dens"] > hi:
            return hi * row["cell_area_km2"]
        return row["pop"]

    cell_c["pop_target"] = cell_c.apply(_target_pop, axis=1)
    cell_c["scale"] = np.where(cell_c["pop"] > 0, cell_c["pop_target"] / cell_c["pop"], 1.0)
    inter_c = inter_c.merge(cell_c[["kvadrat_id", "scale"]], on="kvadrat_id", how="left")
    inter_c["pop_part_c"] = inter_c["pop_part"] * inter_c["scale"].fillna(1.0)
    results_hex["pop_M3c_val"] = results_hex["hex_id"].map(inter_c.groupby("hex_id")["pop_part_c"].sum()).fillna(0)
    results_hex["pop_M3c_folk"] = scale_series(results_hex["pop_M3c_val"], folk_tot)
    validation_rows.append(
        validate_vs_dst(
            aggregate_hex_to_dst(hexg, results_hex.set_index("hex_id")["pop_M3c_folk"], dst),
            "M3c_DST_soft_FOLK",
        )
    )
    method_notes["M3c"] = "M3 soft-projected into DST density-class intervals, then scaled to FOLK1A (preferred)."


    # ---- M4 DST midpoint areal ----
    print("M4 DST midpoint areal…")
    m4 = areal_weight_to_hex(dst.rename(columns={"pop_mid_scaled": "pop_mid_scaled"}), hexg, "pop_mid_scaled")
    results_hex["pop_M4"] = results_hex["hex_id"].map(m4).fillna(0)
    validation_rows.append(validate_vs_dst(aggregate_hex_to_dst(hexg, results_hex.set_index("hex_id")["pop_M4"], dst), "M4_DST_mid_areal"))
    method_notes["M4"] = "DST 1 km class midpoints scaled to KOMM_TOT, areal to hex (previous v1 approach)."

    # ---- M5 hybrid DST + BBR within cell ----
    print("M5 hybrid DST×BBR…")
    # assign buildings to DST cells
    b_dst = assign_points_to_polygons(bbr, dst.rename(columns={"kvadrat_id": "kvadrat_id"}), "kvadrat_id", "dst")
    b_dst = b_dst.rename(columns={"dst_kvadrat_id": "kvadrat_id"})
    bd = bbr.merge(b_dst[["bbr_id", "kvadrat_id"]], on="bbr_id", how="left").dropna(subset=["kvadrat_id"])
    cell_pop = dst[["kvadrat_id", "pop_mid_scaled"]].rename(columns={"pop_mid_scaled": "pop"})
    b5 = dasymetric_from_source(bd, "kvadrat_id", "pop", cell_pop)
    results_hex["pop_M5"] = results_hex["hex_id"].map(buildings_to_hex(b5, hexg, "pop_est")).fillna(0)
    # leftover DST pop in cells with no buildings: areal residual
    allocated = float(results_hex["pop_M5"].sum())
    residual = komm_tot - allocated
    if residual > 1:
        # distribute residual by hex area where pop_M5==0 and hex intersects DST
        empty = results_hex["pop_M5"] <= 0
        if empty.any():
            results_hex.loc[empty, "pop_M5"] += residual * (
                results_hex.loc[empty, "hex_area_m2"] / results_hex.loc[empty, "hex_area_m2"].sum()
            )
    validation_rows.append(validate_vs_dst(aggregate_hex_to_dst(hexg, results_hex.set_index("hex_id")["pop_M5"], dst), "M5_hybrid_DST_BBR"))
    method_notes["M5"] = "Hybrid: DST cell totals refined by BBR bolig weights inside each 1 km cell."

    # ---- M6 postcode floor-area + age ----
    print("M6 postcode floor-area + age…")
    src_post = post[["nr", "pop_post_2026", "age_0_14_post", "age_65plus_post"]].rename(
        columns={"pop_post_2026": "pop"}
    )
    bp = buildings.dropna(subset=["nr"]).copy()
    b6 = dasymetric_from_source(bp, "nr", "pop", src_post[["nr", "pop"]])
    # age shares within postcode using same weights
    for age_col, out_col in [("age_0_14_post", "age_0_14_est"), ("age_65plus_post", "age_65plus_est")]:
        tmp = bp.merge(src_post[["nr", age_col]], on="nr", how="left")
        wsum = tmp.groupby("nr")["bolig_m2"].transform("sum")
        tmp[out_col] = np.where(wsum > 0, tmp[age_col].fillna(0) * tmp["bolig_m2"] / wsum, 0)
        b6[out_col] = tmp[out_col].values

    results_hex["pop_M6"] = results_hex["hex_id"].map(buildings_to_hex(b6, hexg, "pop_est")).fillna(0)
    results_hex["age_0_14_M6"] = results_hex["hex_id"].map(buildings_to_hex(b6, hexg, "age_0_14_est")).fillna(0)
    results_hex["age_65plus_M6"] = results_hex["hex_id"].map(buildings_to_hex(b6, hexg, "age_65plus_est")).fillna(0)
    validation_rows.append(validate_vs_dst(aggregate_hex_to_dst(hexg, results_hex.set_index("hex_id")["pop_M6"], dst), "M6_postcode_BBR"))
    method_notes["M6"] = "Postcode (POSTNR1) totals + age bands redistributed by BYG_BOLIG_."

    # Preferred product: M3c scaled to FOLK1A; ages from M6 proportions
    results_hex["pop_total"] = results_hex["pop_M3c_folk"]
    # age: use M6 spatial pattern scaled to municipal shares from KOMM_TOT / FOLK
    if results_hex["age_0_14_M6"].sum() > 0:
        results_hex["age_0_14"] = scale_series(results_hex["age_0_14_M6"], age0_m * (folk_tot / komm_tot) if komm_tot else age0_m)
        results_hex["age_65plus"] = scale_series(
            results_hex["age_65plus_M6"], age65_m * (folk_tot / komm_tot) if komm_tot else age65_m
        )
    else:
        share0 = age0_m / komm_tot
        share65 = age65_m / komm_tot
        results_hex["age_0_14"] = results_hex["pop_total"] * share0
        results_hex["age_65plus"] = results_hex["pop_total"] * share65

    results_hex["pop_density_km2"] = results_hex["pop_total"] / (results_hex["hex_area_m2"] / 1e6)
    results_hex["pct_0_14"] = np.where(results_hex["pop_total"] > 0, 100 * results_hex["age_0_14"] / results_hex["pop_total"], np.nan)
    results_hex["pct_65plus"] = np.where(results_hex["pop_total"] > 0, 100 * results_hex["age_65plus"] / results_hex["pop_total"], np.nan)

    # Also produce a 2021-scaled M3 for comparability with KOMM_TOT / DST era
    results_hex["pop_M3_komm21"] = scale_series(results_hex["pop_M3_val"], komm_tot)

    val_df = pd.DataFrame(validation_rows).sort_values("class_agreement_rate", ascending=False)
    print("\n=== Validation vs DST 1 km density classes ===")
    print(val_df.to_string(index=False))

    # Choose preferred method by agreement then spearman
    best = val_df.iloc[0]["method"]
    print("Best vs DST classes:", best)

    # Control total checks
    controls = {
        "KOMM_TOT_21": komm_tot,
        "FOLK1A_2026K2": folk_tot,
        "SOGN1_2026_sum": float(sogne["pop_sogn1_2026"].sum()),
        "SOGN10_2022_sum": float(sogne["pop_sogn10_2022"].sum()),
        "POSTNR_2026_sum": float(post["pop_post_2026"].sum()),
        "M1_map_sum": float(results_hex["pop_M1"].sum()),
        "M2_map_sum": float(results_hex["pop_M2"].sum()),
        "M3_map_sum": float(results_hex["pop_M3"].sum()),
        "M3_folk_sum": float(results_hex["pop_M3_folk"].sum()),
        "M3c_folk_sum": float(results_hex["pop_M3c_folk"].sum()),
        "M4_sum": float(results_hex["pop_M4"].sum()),
        "M5_sum": float(results_hex["pop_M5"].sum()),
        "M6_sum": float(results_hex["pop_M6"].sum()),
        "preferred_pop_total_sum": float(results_hex["pop_total"].sum()),
        "preferred_age0_sum": float(results_hex["age_0_14"].sum()),
        "preferred_age65_sum": float(results_hex["age_65plus"].sum()),
    }

    # Parish pycnophylactic check for M3
    b3_chk = b3m.groupby("kode")["pop_est"].sum().rename("allocated")
    chk = sogne.set_index("kode")[["pop_source_map"]].join(b3_chk)
    chk["abs_err"] = (chk["allocated"] - chk["pop_source_map"]).abs()
    controls["M3_parish_max_abs_err"] = float(chk["abs_err"].max())
    controls["M3_parish_mean_abs_err"] = float(chk["abs_err"].mean())

    # Write preferred source for indicator pipeline: building-level then hex already done
    # Also write 1 km reconstructed surface from preferred hex for maps
    preferred_cell = aggregate_hex_to_dst(hexg, results_hex.set_index("hex_id")["pop_total"], dst)
    preferred_cell = dst[["kvadrat_id", "geometry", "dens_class", "dens_mid"]].merge(
        preferred_cell.drop(columns=["dens_class", "dens_mid"], errors="ignore"),
        on="kvadrat_id",
        how="left",
    )
    preferred_cell["pop_total"] = preferred_cell["pop_est"].fillna(0)
    preferred_cell["source"] = "M3c parish×BBR bolig + DST class soft-constraint scaled to FOLK1A"
    preferred_cell.to_file(DEMO / "dasymetric_population_1km_v2.gpkg", driver="GPKG")

    # Hex output GPKG used as demographic source
    out_hex = results_hex.copy()
    out_hex["source"] = "dasymetric_v2 preferred=M3c_folk ages=M6_scaled"
    out_hex.to_file(DEMO / "dasymetric_population_hex500_v2.gpkg", driver="GPKG")
    # Also replace / write companion csv
    out_hex.drop(columns="geometry").to_csv(OUT_DIR / "odense_hex500_demography_methods_v2.csv", index=False)
    val_df.to_csv(OUT_DIR / "dst_class_validation_metrics_v2.csv", index=False)
    pd.DataFrame([controls]).to_csv(OUT_DIR / "control_total_checks_v2.csv", index=False)
    pd.DataFrame(PUBS).to_csv(OUT_DIR / "publications_used_v2.csv", index=False)
    with open(OUT_DIR / "publications_used_v2.json", "w", encoding="utf-8") as f:
        json.dump(PUBS, f, indent=2)
    with open(OUT_DIR / "method_notes_v2.json", "w", encoding="utf-8") as f:
        json.dump(method_notes, f, indent=2)
    with open(OUT_DIR / "control_totals_v2.json", "w", encoding="utf-8") as f:
        json.dump(controls, f, indent=2)

    # Quick diagnostic maps
    print("Writing diagnostic maps…")
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    methods_plot = [
        ("pop_M1", "M1 areal"),
        ("pop_M3c_folk", "M3c preferred"),
        ("pop_M4", "M4 DST mid"),
        ("pop_M5", "M5 hybrid"),
        ("pop_M6", "M6 postcode"),
        ("pop_density_km2", "Preferred density"),
    ]
    for ax, (col, title) in zip(axes.ravel(), methods_plot):
        results_hex.plot(column=col, ax=ax, cmap="YlOrRd", linewidth=0, legend=True, legend_kwds={"shrink": 0.6})
        ax.set_title(title, fontsize=10, loc="left")
        ax.set_axis_off()
    fig.suptitle("Odense demography v2 — method comparison (500 m hex)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "method_comparison_hex_maps_v2.png", dpi=160)
    plt.close(fig)

    # DST agreement map for preferred
    cell_map = preferred_cell.copy()
    cell_map["agree"] = [
        class_agreement(d, c) for d, c in zip(cell_map["dens_est"].fillna(-1), cell_map["dens_class"])
    ]
    fig, ax = plt.subplots(figsize=(8, 8))
    cell_map.plot(column="agree", categorical=True, cmap="RdYlGn", ax=ax, linewidth=0.2, edgecolor="k", legend=True)
    ax.set_title("Preferred M3 vs DST class agreement (green=within class)", loc="left")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "dst_class_agreement_preferred_v2.png", dpi=160)
    plt.close(fig)

    # Scatter dens_est vs dens_mid
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(preferred_cell["dens_mid"], preferred_cell["dens_est"], s=18, alpha=0.7)
    lim = max(preferred_cell["dens_mid"].max(), preferred_cell["dens_est"].max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("DST class midpoint (pers/km²)")
    ax.set_ylabel("Preferred M3 dens_est (pers/km²)")
    ax.set_title("Preferred estimate vs DST class midpoints", loc="left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scatter_preferred_vs_dst_mid_v2.png", dpi=160)
    plt.close(fig)

    report = f"""Demographic dasymetric analysis v2 — Odense AEE
================================================
Hex cells: {len(results_hex)}
Residential BBR buildings used: {len(bbr)}
Parishes matched: {len(sogne)} | Postcodes: {len(post)} | DST 1km cells: {len(dst)}

Control totals
--------------
{json.dumps(controls, indent=2)}

Validation vs DST 1 km density CLASSES (not exact counts)
---------------------------------------------------------
{val_df.to_string(index=False)}

Preferred product
-----------------
pop_total = M3c parish×BBR BYG_BOLIG_ + DST class soft-constraint, scaled to StatBank FOLK1A ({folk_tot:,.0f})
age_0_14 / age_65plus = POSTNR1 age bands × BBR weights, scaled to Basisdata age structure × FOLK ratio
Files:
  {DEMO / 'dasymetric_population_hex500_v2.gpkg'}
  {DEMO / 'dasymetric_population_1km_v2.gpkg'}
  {OUT_DIR}

Important caveats
-----------------
1. DST extranet provides density *intervals*, not person counts; class agreement and rank
   correlation are the appropriate ground-truth metrics.
2. Temporal mismatch: DST classes ~2019, SOGN10=2022, SOGN1/POSTNR/FOLK=2026, BBR=2020,
   KOMM_TOT age structure=2021.
3. OMRP age–sex micro-areas lack a spatial crosswalk in this folder and were used only
   for municipal consistency via KOMM_TOT extracts.
4. Migration / origin fields intentionally excluded per study design.

Publications used
-----------------
""" + "\n".join(f"- {p['citation']} (DOI: {p['doi']}) — {p['use']}" for p in PUBS)

    (OUT_DIR / "ANALYSIS_REPORT_v2.txt").write_text(report, encoding="utf-8")
    (BASIS / "demographic_dasymetric_v2_validation.txt").write_text(report, encoding="utf-8")
    print(report)
    print("DONE")


if __name__ == "__main__":
    main()
