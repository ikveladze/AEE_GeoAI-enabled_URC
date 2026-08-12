#!/usr/bin/env python3
"""Shared manuscript cartography for Odense AEE / URC maps.

Standards:
  - OpenStreetMap basemap under semi-transparent hexagons
  - North arrow in upper-right of the map view
  - Numeric representative-fraction scale (e.g. 1:10 000) at lower-right
  - Standard OSM copyright attribution at lower-right
  - No 'OSM' secondary titles / subtitles
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

WEB_MERCATOR = "EPSG:3857"
HEX_ALPHA = 0.55
OSM_ATTRIBUTION = "© OpenStreetMap contributors"


def clean_title(title: str) -> str:
    """Remove OSM / basemap secondary title fragments."""
    t = title.replace("\n ", "\n").strip()
    drop_phrases = [
        "(OSM-based)",
        "(OSM basemap)",
        "(OSM land-cover features + OSM basemap)",
        "(OSM environmental features + OSM basemap)",
        "(multi-source indicators + OSM basemap)",
        "(5-min OSM network)",
        "(15-min OSM network)",
        "(10-min OSM walking network)",
        "(20-min OSM walking network)",
        "-min OSM walking network)",
        "-min OSM network)",
        "+ OSM basemap",
        "OSM basemap",
        "OSM-based",
        "OSM network",
        "OSM walking network",
    ]
    for p in drop_phrases:
        t = t.replace(p, "")
    lines = [ln.strip(" -–") for ln in t.split("\n") if ln.strip(" -–")]
    return "\n".join(lines)


def add_osm_basemap(ax, crs=WEB_MERCATOR) -> None:
    """Add OSM tiles; ignore environment proxy settings that often break tile downloads."""
    import os

    import contextily as cx
    import requests

    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        os.environ.pop(key, None)

    _orig_request = requests.Session.request

    def _request_no_proxy(self, method, url, **kwargs):
        kwargs.setdefault("proxies", {"http": None, "https": None})
        kwargs.setdefault("timeout", 30)
        return _orig_request(self, method, url, **kwargs)

    requests.Session.request = _request_no_proxy  # type: ignore[method-assign]

    sources = [cx.providers.OpenStreetMap.Mapnik]
    try:
        sources.append(cx.providers.OpenStreetMap.DE)
    except Exception:
        pass
    try:
        sources.append(cx.providers.CartoDB.Positron)
    except Exception:
        pass

    last_err = None
    for source in sources:
        try:
            cx.add_basemap(ax, crs=crs, source=source, zoom="auto", attribution=False)
            return
        except Exception as exc:
            last_err = exc
            print(f"Basemap source failed: {exc}")
    print(f"WARNING: could not load OSM basemap ({last_err}); continuing without tiles.")


def add_north_arrow(ax, x: float = 0.93, y: float = 0.93, size: float = 0.06) -> None:
    """North arrow in axes fraction coordinates (upper right)."""
    arrow = FancyArrowPatch(
        (x, y - size),
        (x, y),
        transform=ax.transAxes,
        color="black",
        arrowstyle="-|>",
        mutation_scale=18,
        linewidth=1.6,
        zorder=20,
    )
    ax.add_patch(arrow)
    ax.text(
        x,
        y + 0.012,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        zorder=21,
        bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.85),
    )


def _nice_rf(rf: float) -> int:
    """Round representative fraction to a cartographically conventional value."""
    if rf <= 0 or not np.isfinite(rf):
        return 10000
    candidates = [
        1000, 2000, 2500, 5000, 7500, 10000, 15000, 20000, 25000,
        30000, 40000, 50000, 75000, 100000, 150000, 200000, 250000, 500000,
    ]
    return int(min(candidates, key=lambda c: abs(c - rf)))


def add_scalebar(ax, fig=None) -> None:
    """Graphical scale bar + representative fraction at lower-right of the map view."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    width_m = abs(xmax - xmin)
    y_mid = 0.5 * (ymin + ymax)
    lat = np.degrees(2 * np.arctan(np.exp(y_mid / 6378137.0)) - np.pi / 2)
    metres_per_map_unit = np.cos(np.radians(lat))  # Web Mercator → ground metres
    ground_width_m = width_m * metres_per_map_unit

    # Choose a nice round bar length (~1/5 of map width)
    target = ground_width_m / 5.0
    candidates = [500, 1000, 2000, 2500, 5000, 7500, 10000, 15000, 20000, 25000]
    bar_m = min(candidates, key=lambda c: abs(c - target))
    bar_map_units = bar_m / metres_per_map_unit

    # Place bar in axes fraction coords near lower-right
    x1, x0 = 0.98, 0.98
    # convert desired ground length to axes fraction of current xlim
    frac = bar_map_units / max(width_m, 1e-9)
    x0 = max(0.55, 0.98 - frac)

    y_bar = 0.065
    # background plate
    ax.add_patch(
        mpatches.FancyBboxPatch(
            (x0 - 0.02, 0.035),
            (0.98 - (x0 - 0.02)),
            0.055,
            transform=ax.transAxes,
            boxstyle="round,pad=0.01",
            facecolor="white",
            edgecolor="#333333",
            linewidth=0.6,
            alpha=0.92,
            zorder=20,
            clip_on=False,
        )
    )
    # dual-tone bar
    mid = (x0 + 0.98) / 2
    ax.plot([x0, mid], [y_bar, y_bar], transform=ax.transAxes, color="black", linewidth=4, solid_capstyle="butt", zorder=21, clip_on=False)
    ax.plot([mid, 0.98], [y_bar, y_bar], transform=ax.transAxes, color="white", linewidth=4, solid_capstyle="butt", zorder=21, clip_on=False)
    ax.plot([x0, 0.98], [y_bar, y_bar], transform=ax.transAxes, color="black", linewidth=1.0, zorder=22, clip_on=False)
    for x in (x0, mid, 0.98):
        ax.plot([x, x], [y_bar - 0.008, y_bar + 0.008], transform=ax.transAxes, color="black", linewidth=1.0, zorder=22, clip_on=False)

    if bar_m >= 1000:
        label = f"0    {bar_m/2000:g}    {bar_m/1000:g} km"
    else:
        label = f"0    {bar_m//2}    {bar_m} m"
    ax.text(0.98, 0.078, label, transform=ax.transAxes, ha="right", va="bottom", fontsize=8, zorder=23)

    if fig is None:
        fig = ax.figure
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    map_width_m_on_page = max(bbox.width * 0.0254, 1e-6)
    rf_nice = _nice_rf(ground_width_m / map_width_m_on_page)
    ax.text(
        0.98,
        0.040,
        f"1:{rf_nice:,}".replace(",", " "),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#333333",
        zorder=23,
    )


def add_osm_attribution(ax) -> None:
    """Standard OSM copyright at lower-right (below scale)."""
    ax.text(
        0.98,
        0.012,
        OSM_ATTRIBUTION,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#222222",
        zorder=22,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85),
    )


def finalize_map_furniture(ax, fig=None) -> None:
    add_north_arrow(ax)
    add_scalebar(ax, fig=fig)
    add_osm_attribution(ax)


def plot_class_choropleth_osm(
    hex_gdf: gpd.GeoDataFrame,
    municipality: gpd.GeoDataFrame,
    municipality_boundary: gpd.GeoDataFrame,
    class_col: str,
    colors: dict[str, str],
    title: str,
    legend_title: str,
    output_path: Path,
    legend_order: list[str] | None = None,
    figsize: tuple[float, float] = (14, 10),
    hex_alpha: float = HEX_ALPHA,
) -> None:
    """Standard choropleth map with OSM basemap, N arrow, RF scale, attribution."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hex_wm = hex_gdf.to_crs(WEB_MERCATOR).copy()
    muni_wm = municipality.to_crs(WEB_MERCATOR)
    boundary_wm = municipality_boundary.to_crs(WEB_MERCATOR)
    hex_wm["map_color"] = hex_wm[class_col].map(colors).fillna("#d9d9d9")

    fig, ax = plt.subplots(figsize=figsize)
    boundary_wm.plot(ax=ax, facecolor="none", edgecolor="none")
    add_osm_basemap(ax, crs=WEB_MERCATOR)

    hex_wm.plot(
        ax=ax,
        color=hex_wm["map_color"],
        edgecolor="white",
        linewidth=0.2,
        alpha=hex_alpha,
        zorder=5,
    )
    muni_wm.boundary.plot(ax=ax, linewidth=0.5, edgecolor="#333333", alpha=0.85, zorder=6)
    boundary_wm.boundary.plot(ax=ax, linewidth=1.2, edgecolor="black", zorder=7)

    ax.set_title(clean_title(title), fontsize=16, fontweight="bold", pad=10)
    ax.set_axis_off()

    if legend_order is None:
        legend_order = [
            "Very high", "High", "Moderate", "Low", "Very low",
            "Beyond 15 min cutoff", "Beyond 5 min cutoff",
            "Beyond 10 min cutoff", "Beyond 20 min cutoff", "Not available",
        ]
    present = set(hex_wm[class_col].dropna().astype(str).unique().tolist())
    order = [x for x in legend_order if x in present]
    for x in sorted(present):
        if x not in order:
            order.append(x)

    patches = [
        mpatches.Patch(facecolor=colors.get(lab, "#d9d9d9"), edgecolor="gray", label=lab, alpha=hex_alpha)
        for lab in order
    ]
    ax.legend(
        handles=patches,
        title=legend_title,
        loc="lower left",
        frameon=True,
        fontsize=9,
        title_fontsize=10,
        framealpha=0.92,
    )

    finalize_map_furniture(ax, fig=fig)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Map saved: {output_path}")


def plot_continuous_choropleth_osm(
    hex_gdf: gpd.GeoDataFrame,
    municipality: gpd.GeoDataFrame,
    municipality_boundary: gpd.GeoDataFrame,
    value_col: str,
    title: str,
    legend_label: str,
    output_path: Path,
    cmap: str = "viridis",
    figsize: tuple[float, float] = (14, 10),
    hex_alpha: float = HEX_ALPHA,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    hex_wm = hex_gdf.to_crs(WEB_MERCATOR).copy()
    muni_wm = municipality.to_crs(WEB_MERCATOR)
    boundary_wm = municipality_boundary.to_crs(WEB_MERCATOR)

    fig, ax = plt.subplots(figsize=figsize)
    boundary_wm.plot(ax=ax, facecolor="none", edgecolor="none")
    add_osm_basemap(ax, crs=WEB_MERCATOR)

    hex_wm.plot(
        ax=ax,
        column=value_col,
        cmap=cmap,
        linewidth=0.15,
        edgecolor="white",
        alpha=hex_alpha,
        legend=True,
        legend_kwds={"label": legend_label, "shrink": 0.65, "pad": 0.01},
        zorder=5,
    )
    muni_wm.boundary.plot(ax=ax, linewidth=0.5, edgecolor="#333333", alpha=0.85, zorder=6)
    boundary_wm.boundary.plot(ax=ax, linewidth=1.2, edgecolor="black", zorder=7)

    ax.set_title(clean_title(title), fontsize=16, fontweight="bold", pad=10)
    ax.set_axis_off()
    finalize_map_furniture(ax, fig=fig)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Map saved: {output_path}")
