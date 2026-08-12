#!/usr/bin/env python3
"""Regenerate all manuscript choropleth maps with standard cartography."""
from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pyogrio

matplotlib.use("Agg")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from map_cartography import plot_class_choropleth_osm, plot_continuous_choropleth_osm

TARGET = "EPSG:25832"

SEQ = {"Very low": "#ffffcc", "Low": "#c2e699", "Moderate": "#78c679", "High": "#31a354", "Very high": "#006837", "Not available": "#d9d9d9"}
WARM = {
    "Beyond 15 min cutoff": "#e0e0e0", "Beyond 5 min cutoff": "#e0e0e0",
    "Beyond 10 min cutoff": "#e0e0e0", "Beyond 20 min cutoff": "#e0e0e0",
    "Very low": "#fff7bc", "Low": "#fec44f", "Moderate": "#fe9929", "High": "#ec7014", "Very high": "#cc0000",
}
VULN = {"Very low": "#ffffcc", "Low": "#fed976", "Moderate": "#fd8d3c", "High": "#f03b20", "Very high": "#bd0026", "Not available": "#d9d9d9"}
QUALITY = {"Very low": "#a50026", "Low": "#f46d43", "Moderate": "#fee08b", "High": "#66bd63", "Very high": "#006837"}
BURDEN = {"Very low": "#006837", "Low": "#66bd63", "Moderate": "#fee08b", "High": "#f46d43", "Very high": "#a50026"}
MISMATCH = {
    "Agreement": "#2ca25f",
    "Soft mismatch: conventional more urban than AEE": "#fee08b",
    "Strong mismatch: conventional more urban than AEE": "#fdae61",
    "Soft mismatch: AEE more urban-functional than conventional": "#abd9e9",
    "Strong mismatch: AEE more urban-functional than conventional": "#2c7bb6",
    "Mismatch": "#d7191c",
    "Not available": "#d9d9d9",
}
CONV = {
    "Conventional urban core": "#7b3294",
    "Conventional urban": "#c2a5cf",
    "Conventional suburban/peri-urban": "#a6dba0",
    "Conventional rural/peripheral": "#008837",
    "Not available": "#d9d9d9",
}
AEE = {
    "Functional urban core": "#7b3294",
    "Functional urban / local centre": "#c2a5cf",
    "Functional transition zone": "#fdb863",
    "Functionally vulnerable transition zone": "#e66101",
    "Functional green fringe / rural": "#5aae61",
    "Functional peripheral rural": "#1b7837",
    "Not available": "#d9d9d9",
}


def load(gpkg: Path, layer: str):
    hexes = gpd.read_file(gpkg, layer=layer).to_crs(TARGET)
    try:
        muni = gpd.read_file(gpkg, layer="odense_municipality_districts").to_crs(TARGET)
    except Exception:
        muni = gpd.GeoDataFrame(geometry=[hexes.geometry.union_all()], crs=TARGET)
    bnd = gpd.GeoDataFrame(geometry=[muni.geometry.union_all()], crs=TARGET)
    return hexes, muni, bnd


def class_map(gpkg, layer, col, colors, title, legend, outs):
    if not gpkg.exists():
        print("SKIP", gpkg)
        return
    hexes, muni, bnd = load(gpkg, layer)
    if col not in hexes.columns:
        print("SKIP col", col, gpkg.name)
        return
    # add any unexpected labels
    for lab in hexes[col].dropna().astype(str).unique():
        if lab not in colors:
            colors = dict(colors)
            colors[lab] = "#999999"
    for out in outs:
        plot_class_choropleth_osm(hexes, muni, bnd, col, colors, title, legend, out)


def cont_map(gpkg, layer, col, title, legend, out, cmap="viridis"):
    if not gpkg.exists():
        print("SKIP", gpkg)
        return
    hexes, muni, bnd = load(gpkg, layer)
    if col not in hexes.columns:
        print("SKIP col", col)
        return
    plot_continuous_choropleth_osm(hexes, muni, bnd, col, title, legend, out, cmap=cmap)


def auto_palette(series):
    vals = sorted(series.dropna().astype(str).unique())
    try:
        cmap = plt.colormaps.get_cmap("tab10" if len(vals) <= 10 else "tab20").resampled(max(len(vals), 1))
    except Exception:
        cmap = plt.cm.get_cmap("tab10" if len(vals) <= 10 else "tab20", max(len(vals), 1))
    return {v: mcolors.to_hex(cmap(i)) for i, v in enumerate(vals)}


def ensure_quintile_class(hexes, value_col: str, class_col: str):
    """Add quintile class column in-place if missing (for official-env GPKGs without classes)."""
    import numpy as np
    import pandas as pd

    if class_col in hexes.columns:
        return hexes
    if value_col not in hexes.columns:
        return hexes
    s = pd.to_numeric(hexes[value_col], errors="coerce")
    labels = ["Very low", "Low", "Moderate", "High", "Very high"]
    try:
        ranks = pd.qcut(s.rank(method="first"), 5, labels=False) + 1
    except Exception:
        ranks = pd.Series(np.nan, index=hexes.index)
    hexes = hexes.copy()
    hexes[class_col] = ranks.map({1: labels[0], 2: labels[1], 3: labels[2], 4: labels[3], 5: labels[4]}).fillna("Not available")
    return hexes


def class_map_with_quintile(gpkg, layer, value_col, class_col, colors, title, legend, outs):
    if not gpkg.exists():
        print("SKIP", gpkg)
        return
    hexes, muni, bnd = load(gpkg, layer)
    hexes = ensure_quintile_class(hexes, value_col, class_col)
    if class_col not in hexes.columns:
        print("SKIP col", class_col, gpkg.name)
        return
    for lab in hexes[class_col].dropna().astype(str).unique():
        if lab not in colors:
            colors = dict(colors)
            colors[lab] = "#999999"
    for out in outs:
        plot_class_choropleth_osm(hexes, muni, bnd, class_col, colors, title, legend, out)


def main():
    # Accessibility / env / demo
    class_map(
        SCRIPT_DIR / "odense_walking_accessibility_services_500m.gpkg",
        "walking_accessibility_500m_hexagons", "walk_access_class", WARM,
        "Odense Walking Accessibility to Services", "Walking accessibility",
        [SCRIPT_DIR / "odense_walking_accessibility_services_500m_map.png",
         SCRIPT_DIR / "odense_walking_accessibility_services_500m_map_osm.png"],
    )
    bike = SCRIPT_DIR / "odense_osm_cycling_accessibility_outputs" / "odense_cycling_accessibility_services_500m.gpkg"
    class_map(bike, "cycling_accessibility_500m_hexagons", "bike_access_class_5m", WARM,
              "Odense Cycling Accessibility to Services (5 min)", "Cycling accessibility (5 min)",
              [SCRIPT_DIR / "odense_osm_cycling_accessibility_outputs" / "odense_cycling_accessibility_services_500m_map_5min.png"])
    class_map(bike, "cycling_accessibility_500m_hexagons", "bike_access_class_15m", WARM,
              "Odense Cycling Accessibility to Services (15 min)", "Cycling accessibility (15 min)",
              [SCRIPT_DIR / "odense_osm_cycling_accessibility_outputs" / "odense_cycling_accessibility_services_500m_map_15min.png",
               SCRIPT_DIR / "odense_osm_cycling_accessibility_outputs" / "odense_cycling_accessibility_services_500m_map_15min_osm.png"])
    pt = SCRIPT_DIR / "odense_osm_public_transport_accessibility_outputs" / "odense_public_transport_stop_accessibility_500m.gpkg"
    class_map(pt, "pt_stop_accessibility_500m_hexagons", "pt_access_class_10m", WARM,
              "Odense Public-Transport Stop Accessibility (10 min)", "PT stop accessibility (10 min)",
              [SCRIPT_DIR / "odense_osm_public_transport_accessibility_outputs" / "odense_public_transport_stop_accessibility_500m_map_10min.png"])
    class_map(pt, "pt_stop_accessibility_500m_hexagons", "pt_access_class_20m", WARM,
              "Odense Public-Transport Stop Accessibility (20 min)", "PT stop accessibility (20 min)",
              [SCRIPT_DIR / "odense_osm_public_transport_accessibility_outputs" / "odense_public_transport_stop_accessibility_500m_map_20min.png",
               SCRIPT_DIR / "odense_osm_public_transport_accessibility_outputs" / "odense_public_transport_stop_accessibility_500m_map_20min_osm.png"])
    green = SCRIPT_DIR / "odense_osm_green_area_share_outputs" / "odense_green_area_share_500m.gpkg"
    class_map_with_quintile(
        green, "green_area_share_500m_hexagons", "green_share_pct", "green_class", SEQ,
        "Odense Green-Area Share by 500 m Cell", "Green-area share",
        [SCRIPT_DIR / "odense_osm_green_area_share_outputs" / "odense_green_area_share_500m_map.png",
         SCRIPT_DIR / "odense_osm_green_area_share_outputs" / "odense_green_area_share_500m_map_osm.png"],
    )
    env = SCRIPT_DIR / "odense_multisource_environmental_index_outputs" / "odense_multisource_environmental_index_500m.gpkg"
    class_map_with_quintile(
        env, "environmental_index_500m_hexagons", "env_quality_score", "env_quality_class", QUALITY,
        "Odense Environmental Quality Index by 500 m Cell", "Environmental quality",
        [SCRIPT_DIR / "odense_multisource_environmental_index_outputs" / "odense_environmental_quality_index_500m_map.png",
         SCRIPT_DIR / "odense_multisource_environmental_index_outputs" / "odense_environmental_quality_index_500m_map_osm.png"],
    )
    class_map_with_quintile(
        env, "environmental_index_500m_hexagons", "env_burden_score", "env_burden_class", BURDEN,
        "Odense Environmental Burden Index by 500 m Cell", "Environmental burden",
        [SCRIPT_DIR / "odense_multisource_environmental_index_outputs" / "odense_environmental_burden_index_500m_map.png"],
    )

    demo = SCRIPT_DIR / "odense_demographic_indicator_outputs" / "odense_demographic_indicators_500m.gpkg"
    class_map(demo, "demographic_indicators_500m_hexagons", "pop_density_class", SEQ,
              "Odense Population Density by 500 m Cell", "Population density",
              [SCRIPT_DIR / "odense_demographic_indicator_outputs" / "odense_population_density_500m_map.png"])
    class_map(demo, "demographic_indicators_500m_hexagons", "demo_vulnerability_class", VULN,
              "Odense Demographic Vulnerability Index by 500 m Cell", "Demographic vulnerability",
              [SCRIPT_DIR / "odense_demographic_indicator_outputs" / "odense_demographic_vulnerability_index_500m_map.png",
               SCRIPT_DIR / "odense_demographic_indicator_outputs" / "odense_demographic_vulnerability_index_500m_map_osm.png"])

    # Typology
    typ = SCRIPT_DIR / "odense_geoai_functional_urc_typology_outputs" / "odense_geoai_functional_urc_typology_500m.gpkg"
    if typ.exists():
        layer = pyogrio.list_layers(typ)[0][0]
        g = gpd.read_file(typ, layer=layer)
        label_col = "functional_urc_type" if "functional_urc_type" in g.columns else "functional_urc_class"
        class_map(typ, layer, label_col, auto_palette(g[label_col]),
                  "Odense GeoAI-derived Functional Urban–Rural Continuum Typology", "Functional URC type",
                  [SCRIPT_DIR / "odense_geoai_functional_urc_typology_outputs" / "odense_geoai_functional_urc_typology_map.png"])
        cont_map(typ, layer, "functional_urc_score",
                 "Odense Functional Urban–Rural Continuum Score", "Functional URC score",
                 SCRIPT_DIR / "odense_geoai_functional_urc_typology_outputs" / "odense_functional_urc_score_map.png")

    # Mismatch
    mm = SCRIPT_DIR / "odense_urc_mismatch_outputs" / "odense_urc_conventional_vs_aee_mismatch_500m.gpkg"
    if mm.exists():
        layer = pyogrio.list_layers(mm)[0][0]
        g = gpd.read_file(mm, layer=layer)
        outs = SCRIPT_DIR / "odense_urc_mismatch_outputs"
        mismatch_col = "mismatch_class" if "mismatch_class" in g.columns else (
            "mismatch_category" if "mismatch_category" in g.columns else None
        )
        if mismatch_col:
            pal = dict(MISMATCH)
            for lab in g[mismatch_col].dropna().astype(str).unique():
                if lab not in pal:
                    pal[lab] = "#999999"
            class_map(mm, layer, mismatch_col, pal,
                      "Odense Conventional vs AEE Functional URC Mismatch", "Mismatch class",
                      [outs / "odense_urc_conventional_vs_aee_mismatch_map.png"])
        if "agreement_class" in g.columns or "agreement_label" in g.columns:
            col = "agreement_class" if "agreement_class" in g.columns else "agreement_label"
            class_map(mm, layer, col, {"Agreement": "#2ca25f", "Mismatch": "#d7191c", "Not available": "#d9d9d9"},
                      "Odense Conventional vs AEE Agreement", "Agreement",
                      [outs / "odense_urc_agreement_map.png"])
        if "conventional_class" in g.columns:
            class_map(mm, layer, "conventional_class", CONV,
                      "Odense Conventional Urban–Rural Classification", "Conventional class",
                      [outs / "odense_conventional_urban_rural_classification_map.png"])
        if "aee_functional_class" in g.columns:
            # extend palette with observed labels
            pal = dict(AEE)
            for lab in g["aee_functional_class"].dropna().astype(str).unique():
                if lab not in pal:
                    pal[lab] = "#999999"
            class_map(mm, layer, "aee_functional_class", pal,
                      "Odense AEE Functional Urban–Rural Continuum Classes", "AEE functional class",
                      [outs / "odense_aee_functional_urc_class_map.png"])
        if "mismatch_intensity" in g.columns:
            cont_map(mm, layer, "mismatch_intensity",
                     "Odense Conventional vs AEE Mismatch Intensity", "Mismatch intensity",
                     outs / "odense_urc_mismatch_intensity_map.png", cmap="magma")

    # Benchmarking
    bn = SCRIPT_DIR / "odense_urc_benchmarking_uncertainty_outputs" / "odense_urc_benchmarking_uncertainty_500m.gpkg"
    if bn.exists():
        layer = pyogrio.list_layers(bn)[0][0]
        g = gpd.read_file(bn, layer=layer)
        outs = SCRIPT_DIR / "odense_urc_benchmarking_uncertainty_outputs"
        for col, title, out, kind, cmap in [
            ("urc_type_uncertainty", "Odense GMM Membership Uncertainty", outs / "odense_urc_gmm_membership_uncertainty_map.png", "cont", "YlOrRd"),
            ("boundary_heterogeneity", "Odense Boundary Heterogeneity", outs / "odense_urc_boundary_heterogeneity_map.png", "cont", "plasma"),
            ("scenario_stability", "Odense Scenario Stability", outs / "odense_urc_scenario_stability_map.png", "cont", "viridis"),
        ]:
            if col in g.columns:
                cont_map(bn, layer, col, title, col.replace("_", " ").title(), out, cmap=cmap)
        for col, title, out in [
            ("gmm_kmeans_agreement_class", "Odense GMM–K-means Agreement", outs / "odense_urc_gmm_kmeans_agreement_map.png"),
            ("gmm_kmeans_agreement", "Odense GMM–K-means Agreement", outs / "odense_urc_gmm_kmeans_agreement_map.png"),
            ("scenario_stability_class", "Odense Scenario Stability Classes", outs / "odense_urc_scenario_stability_map.png"),
            ("boundary_heterogeneity_class", "Odense Boundary Heterogeneity Classes", outs / "odense_urc_boundary_heterogeneity_map.png"),
        ]:
            if col in g.columns:
                class_map(bn, layer, col, auto_palette(g[col]), title, col.replace("_", " ").title(), [out])
        # Density morphology benchmark (categorical labels from dedicated benchmark run)
        dens_col = next((c for c in ("density_benchmark_class", "benchmark_morphology_density_only") if c in g.columns), None)
        if dens_col:
            class_map(bn, layer, dens_col, auto_palette(g[dens_col]),
                      "Odense Density Benchmark Comparison", "Density benchmark",
                      [outs / "odense_urc_density_benchmark_map.png"])

    # Feature-space cluster maps
    fs = SCRIPT_DIR / "odense_aee_feature_space_outputs" / "odense_aee_feature_space_500m.gpkg"
    if fs.exists():
        layer = pyogrio.list_layers(fs)[0][0]
        g = gpd.read_file(fs, layer=layer)
        outs = SCRIPT_DIR / "odense_aee_feature_space_outputs"
        for col, outname, title in [
            ("cluster_pca", "odense_aee_pca_cluster_map.png", "Odense AEE PCA Feature-Space Clusters"),
            ("cluster_umap", "odense_aee_umap_cluster_map.png", "Odense AEE UMAP Feature-Space Clusters"),
            ("cluster_feature", "odense_aee_feature_cluster_map.png", "Odense AEE Feature Clusters"),
        ]:
            if col in g.columns:
                class_map(fs, layer, col, auto_palette(g[col].astype(str)), title, "Cluster", [outs / outname])

    print("Standard map regeneration finished.")


if __name__ == "__main__":
    main()
