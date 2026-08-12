#!/usr/bin/env python3
"""Regenerate ALL manuscript geographic maps with standardised styling."""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from map_style_utils import (  # noqa: E402
    plot_categorical_hex_map,
    plot_custom_colored_hex_map,
    plot_numeric_hex_map,
)

TARGET_CRS = "EPSG:25832"


def muni_from(gpkg: Path) -> gpd.GeoDataFrame:
    import pyogrio

    layers = [r[0] for r in pyogrio.list_layers(gpkg)]
    if "odense_municipality_districts" in layers:
        return gpd.read_file(gpkg, layer="odense_municipality_districts").to_crs(TARGET_CRS)
    walk = SCRIPT_DIR / "odense_walking_accessibility_services_500m.gpkg"
    return gpd.read_file(walk, layer="odense_municipality_districts").to_crs(TARGET_CRS)


def access_colors(present: set[str]) -> dict[str, str]:
    all_c = {
        "Very low": "#fff7bc",
        "Low": "#fec44f",
        "Moderate": "#fe9929",
        "High": "#ec7014",
        "Very high": "#cc0000",
        "Beyond 5 min cutoff": "#d9d9d9",
        "Beyond 10 min cutoff": "#d9d9d9",
        "Beyond 15 min cutoff": "#d9d9d9",
        "Beyond 20 min cutoff": "#d9d9d9",
    }
    out = {k: v for k, v in all_c.items() if k in present}
    for p in present:
        out.setdefault(p, "#d9d9d9")
    return out


def density_colors() -> dict[str, str]:
    return {
        "Very low": "#ffffcc",
        "Low": "#c2e699",
        "Moderate": "#78c679",
        "High": "#31a354",
        "Very high": "#006837",
        "Not available": "#d9d9d9",
    }


def vuln_colors() -> dict[str, str]:
    return {
        "Very low": "#ffffcc",
        "Low": "#fed976",
        "Moderate": "#fd8d3c",
        "High": "#f03b20",
        "Very high": "#bd0026",
        "Not available": "#d9d9d9",
    }


def quality_colors() -> dict[str, str]:
    # Sequential green → light orange (high → low quality); no red / burnt orange
    # Matched to Figure 4A manuscript reference palette
    return {
        "Very high": "#1a9850",
        "High": "#91cf60",
        "Moderate": "#d9ef8b",
        "Low": "#fee08b",
        "Very low": "#fdae61",
        "Not available": "#d9d9d9",
    }


def burden_colors() -> dict[str, str]:
    return {
        "Very low": "#ffffcc",
        "Low": "#fed976",
        "Moderate": "#fd8d3c",
        "High": "#f03b20",
        "Very high": "#bd0026",
    }


def green_colors() -> dict[str, str]:
    return {
        "Very low": "#f7fcf5",
        "Low": "#c7e9c0",
        "Moderate": "#74c476",
        "High": "#238b45",
        "Very high": "#00441b",
    }


def quintile_class(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    try:
        ranks = pd.qcut(s.rank(method="first"), q=5, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series(["Not available"] * len(s), index=s.index)
    labels = ["Very low", "Low", "Moderate", "High", "Very high"]
    uniq = sorted(pd.Series(ranks).dropna().unique())
    mapping = {u: labels[i] for i, u in enumerate(uniq)}
    return pd.Series(ranks, index=s.index).map(mapping).fillna("Not available")


def palette_for(values: list[str], cmap_name: str = "tab10") -> dict[str, str]:
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    return {v: matplotlib.colors.to_hex(cmap(i % cmap.N)) for i, v in enumerate(values)}


def cat(hexes, muni, col, colors, title, legend, out, order=None, legend_loc="lower left", legend_offset_mm=(0.0, 0.0)):
    present = set(hexes[col].dropna().astype(str).unique())
    use = {k: colors.get(k, "#d9d9d9") for k in present}
    for k in present:
        use.setdefault(k, "#d9d9d9")
    plot_categorical_hex_map(
        hexes,
        muni,
        col,
        use,
        title,
        legend,
        out,
        legend_order=order,
        legend_loc=legend_loc,
        legend_offset_mm=legend_offset_mm,
    )


def main() -> None:
    print("=== Regenerating all manuscript maps (left titles; OSM only if filename has 'osm') ===")

    # Walking
    gpkg = SCRIPT_DIR / "odense_walking_accessibility_services_500m.gpkg"
    hexes = gpd.read_file(gpkg, layer="walking_accessibility_500m_hexagons").to_crs(TARGET_CRS)
    muni = muni_from(gpkg)
    colors = access_colors(set(hexes["walk_access_class"].astype(str)))
    for out in [
        SCRIPT_DIR / "odense_walking_accessibility_services_500m_map.png",
        SCRIPT_DIR / "odense_walking_accessibility_services_500m_map_osm.png",
    ]:
        cat(
            hexes,
            muni,
            "walk_access_class",
            colors,
            "Odense Walking Accessibility to Services",
            "Walking accessibility",
            out,
            ["Very high", "High", "Moderate", "Low", "Very low", "Beyond 15 min cutoff"],
        )

    # Cycling
    bike_dir = SCRIPT_DIR / "odense_osm_cycling_accessibility_outputs"
    bike_gpkg = bike_dir / "odense_cycling_accessibility_services_500m.gpkg"
    hex_b = gpd.read_file(bike_gpkg, layer="cycling_accessibility_500m_hexagons").to_crs(TARGET_CRS)
    muni_b = muni_from(bike_gpkg)
    for minutes, col, outs in [
        (5, "bike_access_class_5m", [bike_dir / "odense_cycling_accessibility_services_500m_map_5min.png"]),
        (
            15,
            "bike_access_class_15m",
            [
                bike_dir / "odense_cycling_accessibility_services_500m_map_15min.png",
                bike_dir / "odense_cycling_accessibility_services_500m_map_15min_osm.png",
            ],
        ),
    ]:
        colors = access_colors(set(hex_b[col].astype(str)))
        for out in outs:
            cat(hex_b, muni_b, col, colors, f"Odense Cycling Accessibility to Services ({minutes} min)", "Cycling accessibility", out)

    # PT
    pt_dir = SCRIPT_DIR / "odense_osm_public_transport_accessibility_outputs"
    pt_gpkg = pt_dir / "odense_public_transport_stop_accessibility_500m.gpkg"
    hex_p = gpd.read_file(pt_gpkg, layer="pt_stop_accessibility_500m_hexagons").to_crs(TARGET_CRS)
    muni_p = muni_from(pt_gpkg)
    for minutes, col, outs in [
        (10, "pt_access_class_10m", [pt_dir / "odense_public_transport_stop_accessibility_500m_map_10min.png"]),
        (
            20,
            "pt_access_class_20m",
            [
                pt_dir / "odense_public_transport_stop_accessibility_500m_map_20min.png",
                pt_dir / "odense_public_transport_stop_accessibility_500m_map_20min_osm.png",
            ],
        ),
    ]:
        colors = access_colors(set(hex_p[col].astype(str)))
        for out in outs:
            cat(hex_p, muni_p, col, colors, f"Odense Public Transport Stop Accessibility ({minutes} min)", "PT stop accessibility", out)

    # Green
    green_dir = SCRIPT_DIR / "odense_osm_green_area_share_outputs"
    green_gpkg = green_dir / "odense_green_area_share_500m.gpkg"
    hex_g = gpd.read_file(green_gpkg, layer="green_area_share_500m_hexagons").to_crs(TARGET_CRS)
    muni_g = muni_from(green_gpkg)
    hex_g = hex_g.copy()
    hex_g["green_share_class"] = quintile_class(hex_g["green_share_pct"])
    for out in [
        green_dir / "odense_green_area_share_500m_map.png",
        green_dir / "odense_green_area_share_500m_map_osm.png",
    ]:
        cat(hex_g, muni_g, "green_share_class", green_colors(), "Odense Green-Area Share by 500 m Cell", "Green-area share", out)

    # Env quality / burden
    env_dir = SCRIPT_DIR / "odense_multisource_environmental_index_outputs"
    env_gpkg = env_dir / "odense_multisource_environmental_index_500m.gpkg"
    hex_e = gpd.read_file(env_gpkg, layer="environmental_index_500m_hexagons").to_crs(TARGET_CRS)
    muni_e = muni_from(env_gpkg)
    hex_e = hex_e.copy()
    qcol = "env_quality_score" if "env_quality_score" in hex_e.columns else "env_quality"
    bcol = "env_burden_score" if "env_burden_score" in hex_e.columns else "env_burden"
    hex_e["env_quality_class"] = quintile_class(hex_e[qcol])
    hex_e["env_burden_class"] = quintile_class(hex_e[bcol])
    for out in [
        env_dir / "odense_environmental_quality_index_500m_map.png",
        env_dir / "odense_environmental_quality_index_500m_map_osm.png",
    ]:
        cat(hex_e, muni_e, "env_quality_class", quality_colors(), "Environmental Quality Index", "Environmental quality", out)
    cat(hex_e, muni_e, "env_burden_class", burden_colors(), "Odense Environmental Burden Index by 500 m Cell", "Environmental burden", env_dir / "odense_environmental_burden_index_500m_map.png")
    # Component maps (plain / no OSM unless filename has osm)
    for col, title, out in [
        ("builtup_pct", "Odense Built-up Share by 500 m Cell", env_dir / "odense_builtup_share_500m_map.png"),
        ("green_share_pct", "Odense Green Share (Environmental Index Input) by 500 m Cell", env_dir / "odense_green_share_env_500m_map.png"),
        ("road_buf_pct", "Odense Major-Road Buffer Share by 500 m Cell", env_dir / "odense_road_buffer_share_500m_map.png"),
    ]:
        if col in hex_e.columns:
            hex_e[f"{col}_class"] = quintile_class(hex_e[col])
            cmap = green_colors() if "green" in col else (burden_colors() if "road" in col else density_colors())
            cat(hex_e, muni_e, f"{col}_class", cmap, title, col.replace("_", " "), out)

    # Demography
    demo_dir = SCRIPT_DIR / "odense_demographic_indicator_outputs"
    demo_gpkg = demo_dir / "odense_demographic_indicators_500m.gpkg"
    hex_d = gpd.read_file(demo_gpkg, layer="demographic_indicators_500m_hexagons").to_crs(TARGET_CRS)
    muni_d = muni_from(demo_gpkg)
    for col, colors, title, legend, out in [
        ("pop_density_class", density_colors(), "Odense Population Density by 500 m Cell", "Population density", demo_dir / "odense_population_density_500m_map.png"),
        ("age_vulnerability_class", vuln_colors(), "Odense Age Vulnerability by 500 m Cell", "Age vulnerability", demo_dir / "odense_age_vulnerability_500m_map.png"),
        ("socioeconomic_vulnerability_class", vuln_colors(), "Odense Socio-economic Vulnerability by 500 m Cell", "Socio-economic vulnerability", demo_dir / "odense_socioeconomic_vulnerability_500m_map.png"),
        ("demo_vulnerability_class", vuln_colors(), "Demographic vulnerability index", "Demographic vulnerability", demo_dir / "odense_demographic_vulnerability_index_500m_map.png"),
        ("demo_vulnerability_class", vuln_colors(), "Demographic vulnerability index", "Demographic vulnerability", demo_dir / "odense_demographic_vulnerability_index_500m_map_osm.png"),
    ]:
        if col in hex_d.columns:
            cat(hex_d, muni_d, col, colors, title, legend, out)

    # Typology
    typ_dir = SCRIPT_DIR / "odense_geoai_functional_urc_typology_outputs"
    typ_gpkg = typ_dir / "odense_geoai_functional_urc_typology_500m.gpkg"
    hex_t = gpd.read_file(typ_gpkg, layer="functional_urc_typology_500m_hexagons").to_crs(TARGET_CRS)
    muni_t = muni_from(typ_gpkg)
    type_col = "functional_urc_type" if "functional_urc_type" in hex_t.columns else "functional_urc_class"
    from geoai_functional_urc_typology_odense import make_typology_palette

    color_map = make_typology_palette(hex_t[type_col].tolist())
    types = sorted(hex_t[type_col].dropna().astype(str).unique().tolist())
    hex_t = hex_t.copy()
    hex_t["map_color"] = hex_t[type_col].map(color_map).fillna("#d9d9d9")
    handles = [mpatches.Patch(facecolor=color_map[t], edgecolor="gray", label=t) for t in types]
    plot_custom_colored_hex_map(
        hex_t,
        muni_t,
        "map_color",
        "Functional Urban-Rural Continuum Typology",
        [],  # no legend on this manuscript figure
        "",
        typ_dir / "odense_functional_urc_typology_map.png",
        use_basemap=False,
    )
    plot_custom_colored_hex_map(
        hex_t,
        muni_t,
        "map_color",
        "GeoAI-derived Functional Urban-Rural Continuum Typology",
        handles,
        "Functional URC type",
        typ_dir / "odense_geoai_functional_urc_typology_map.png",
        use_basemap=False,
        legend_loc="lower right",
    )
    plot_numeric_hex_map(
        hex_t,
        muni_t,
        "functional_urc_score",
        "Odense Functional Urban–Rural Continuum Score",
        typ_dir / "odense_functional_urc_score_map.png",
        cmap="viridis",
        legend_label="URC score (0 urban → 100 rural)",
    )

    # Benchmarking
    bench_dir = SCRIPT_DIR / "odense_urc_benchmarking_uncertainty_outputs"
    bench_gpkg = bench_dir / "odense_urc_benchmarking_uncertainty_500m.gpkg"
    hex_u = gpd.read_file(bench_gpkg, layer="urc_benchmarking_uncertainty_500m_hexagons").to_crs(TARGET_CRS)
    muni_u = muni_from(bench_gpkg)
    plot_numeric_hex_map(hex_u, muni_u, "urc_type_uncertainty", "Functional URC: GMM Membership Uncertainty", bench_dir / "odense_urc_gmm_membership_uncertainty_map.png", cmap="magma", legend_label="Uncertainty")
    plot_numeric_hex_map(hex_u, muni_u, "boundary_heterogeneity", "Functional URC: Boundary Heterogeneity", bench_dir / "odense_urc_boundary_heterogeneity_map.png", cmap="magma", legend_label="Boundary heterogeneity")
    plot_numeric_hex_map(hex_u, muni_u, "scenario_stability", "Odense Functional URC: Scenario Stability", bench_dir / "odense_urc_scenario_stability_map.png", cmap="viridis", legend_label="Scenario stability")
    agreement_colors = {
        "Agreement with dominant K-means pairing": "#a6cee3",  # light blue
        "Algorithmically unstable": "#1f78b4",  # dark blue
        "Not available": "#d9d9d9",
    }
    cat(
        hex_u,
        muni_u,
        "gmm_kmeans_agreement_class",
        agreement_colors,
        "Functional URC: GMM–K-means Agreement",
        "Agreement",
        bench_dir / "odense_urc_gmm_kmeans_agreement_map.png",
        order=["Agreement with dominant K-means pairing", "Algorithmically unstable"],
        legend_loc="lower right",
        legend_offset_mm=(0.0, 5.0),
    )
    dens_col = "population_density_quintile" if "population_density_quintile" in hex_u.columns else "benchmark_morphology_density_only"
    dens_vals = sorted(hex_u[dens_col].dropna().astype(str).unique())
    dens_colors = {**density_colors(), **palette_for(dens_vals, "YlGn")}
    cat(hex_u, muni_u, dens_col, dens_colors, "Odense Benchmark Classification: Population Density Quintiles", "Density class", bench_dir / "odense_urc_density_benchmark_map.png")

    # Mismatch
    mm_dir = SCRIPT_DIR / "odense_urc_mismatch_outputs"
    mm_gpkg = mm_dir / "odense_urc_conventional_vs_aee_mismatch_500m.gpkg"
    hex_m = gpd.read_file(mm_gpkg, layer="conventional_vs_aee_mismatch_500m").to_crs(TARGET_CRS)
    muni_m = muni_from(mm_gpkg)
    for col, title, out in [
        ("mismatch_category", "Odense URC Mismatch: Conventional Classification vs AEE Functional Typology", mm_dir / "odense_urc_conventional_vs_aee_mismatch_map.png"),
        ("agreement_class", "Odense URC Agreement / Mismatch Map", mm_dir / "odense_urc_agreement_map.png"),
        ("conventional_class", "Odense Conventional Urban–Rural Classification", mm_dir / "odense_conventional_urban_rural_classification_map.png"),
        ("aee_functional_class", "Odense AEE Functional URC Classification", mm_dir / "odense_aee_functional_urc_class_map.png"),
    ]:
        vals = sorted(hex_m[col].dropna().astype(str).unique())
        cat(hex_m, muni_m, col, palette_for(vals, "tab10"), title, col.replace("_", " "), out)
    plot_numeric_hex_map(hex_m, muni_m, "mismatch_intensity", "Odense URC Mismatch Intensity", mm_dir / "odense_urc_mismatch_intensity_map.png", cmap="YlOrRd", legend_label="Mismatch intensity")

    # PCA / UMAP cluster maps
    feat_dir = SCRIPT_DIR / "odense_aee_feature_space_outputs"
    feat_gpkg = feat_dir / "odense_aee_feature_space_500m.gpkg"
    hex_f = gpd.read_file(feat_gpkg, layer="aee_feature_space_500m_hexagons").to_crs(TARGET_CRS)
    muni_f = muni_from(feat_gpkg)
    for col, title, out in [
        ("cluster_pca", "Odense AEE PCA clusters by 500 m cell", feat_dir / "odense_aee_pca_cluster_map.png"),
        ("cluster_umap", "Odense AEE UMAP clusters by 500 m cell", feat_dir / "odense_aee_umap_cluster_map.png"),
        ("cluster_feature", "Odense AEE clusters in full standardised feature space", feat_dir / "odense_aee_feature_cluster_map.png"),
    ]:
        vals = sorted(hex_f[col].dropna().astype(str).unique())
        cat(hex_f, muni_f, col, palette_for(vals, "tab10"), title, "Cluster", out)

    print("=== Map regeneration complete ===")


if __name__ == "__main__":
    main()
