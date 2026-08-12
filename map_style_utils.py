"""Shared manuscript map styling for Odense AEE / URC figures.

Rules:
  - Titles are left-aligned
  - OSM basemap + copyright only when the *filename* contains ``osm``
    (e.g. ``..._map_osm.png``). Paths/folders named ``odense_osm_*`` do not count.
  - Non-OSM maps: plain white background, no basemap attribution
  - North arrow (upper right) on all geographic maps
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

WEB_MERCATOR = "EPSG:3857"
HEX_ALPHA_OSM = 0.58
HEX_ALPHA_PLAIN = 0.92
OSM_COPYRIGHT = "© OpenStreetMap contributors"
BOUNDARY_LW = 1.15


def filename_wants_osm_basemap(output_path: Path | str) -> bool:
    """True only if the output *file name* contains 'osm' (not parent folders)."""
    return "osm" in Path(output_path).stem.lower()


def clean_map_title(title: str) -> str:
    """Remove OSM / basemap secondary phrases from titles."""
    t = " ".join(str(title).replace("\n", " ").split())
    drop_phrases = [
        "(OSM-based)",
        "(OSM basemap)",
        "+ OSM basemap",
        "OSM basemap",
        "(multi-source indicators + OSM basemap)",
        "multi-source indicators + OSM basemap",
        "-min OSM network",
        "-min OSM walking network",
        "OSM network",
        "OSM walking network",
    ]
    for p in drop_phrases:
        t = t.replace(p, "")
    t = t.replace("()", "").replace("  ", " ").strip(" -–|")
    t = t.replace("Services (5)", "Services (5 min)")
    t = t.replace("Services (15)", "Services (15 min)")
    t = t.replace("Accessibility (10)", "Accessibility (10 min)")
    t = t.replace("Accessibility (20)", "Accessibility (20 min)")
    return " ".join(t.split())


def set_left_title(ax, title: str, fontsize: float = 16) -> None:
    ax.set_title(
        title,
        fontsize=fontsize,
        fontweight="bold",
        pad=12,
        loc="left",
        ha="left",
    )


def add_north_arrow(
    ax,
    x: float = 0.94,
    y: float = 0.92,
    size: float = 0.06,
    white_outline: bool = False,
    outline_mm: float = 0.5,
) -> None:
    """Simple north arrow in axes fraction coordinates (upper right).

    When ``white_outline`` is True (OSM basemap maps), draw a white halo of
    ``outline_mm`` millimetres around the arrow and "N" so they stay readable.
    """
    from matplotlib import patheffects as pe

    # Matplotlib linewidth is in points; 1 pt = 1/72 in; 1 mm = 72/25.4 pt
    outline_pt = outline_mm * (72.0 / 25.4)
    black_lw = 1.6

    if white_outline:
        # Underlay: thicker white arrow (more reliable than path effects on FancyArrowPatch)
        white_arrow = FancyArrowPatch(
            (x, y - size),
            (x, y),
            transform=ax.transAxes,
            arrowstyle="-|>",
            mutation_scale=14 + outline_pt * 1.5,
            linewidth=black_lw + 2 * outline_pt,
            color="white",
            zorder=10,
        )
        ax.add_patch(white_arrow)

    arrow = FancyArrowPatch(
        (x, y - size),
        (x, y),
        transform=ax.transAxes,
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=black_lw,
        color="black",
        zorder=11,
    )
    ax.add_patch(arrow)

    txt = ax.text(
        x,
        y + 0.012,
        "N",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color="black",
        zorder=11,
    )
    if white_outline:
        # Text halo: stroke width ≈ full outline on each side of glyph
        txt.set_path_effects(
            [pe.withStroke(linewidth=2 * outline_pt + 0.8, foreground="white")]
        )


def add_osm_copyright(ax, text: str = OSM_COPYRIGHT) -> None:
    """Standard OSM attribution, lower-right corner of the map view."""
    ax.text(
        0.985,
        0.015,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#222222",
        zorder=10,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.75),
    )


def add_basemap(ax, crs) -> None:
    import contextily as cx

    try:
        cx.add_basemap(ax, crs=crs, source=cx.providers.OpenStreetMap.Mapnik, zoom="auto", attribution=False)
    except Exception as exc:
        print(f"OSM Mapnik basemap failed ({exc}); trying Carto Positron.")
        cx.add_basemap(ax, crs=crs, source=cx.providers.CartoDB.Positron, attribution=False)


def _prepare_frame(
    hexes: gpd.GeoDataFrame,
    municipality: gpd.GeoDataFrame,
    output_path: Path,
    use_basemap: bool | None,
    alpha: float | None,
    figsize: tuple[float, float],
):
    output_path = Path(output_path)
    if use_basemap is None:
        use_basemap = filename_wants_osm_basemap(output_path)
    if alpha is None:
        alpha = HEX_ALPHA_OSM if use_basemap else HEX_ALPHA_PLAIN

    hex_wm = hexes.to_crs(WEB_MERCATOR).copy()
    muni_wm = municipality.to_crs(WEB_MERCATOR)
    boundary = gpd.GeoDataFrame(geometry=[muni_wm.geometry.union_all()], crs=muni_wm.crs)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    boundary.plot(ax=ax, facecolor="none", edgecolor="none")
    if use_basemap:
        add_basemap(ax, hex_wm.crs)

    return fig, ax, hex_wm, muni_wm, boundary, output_path, use_basemap, alpha


def _finish_map(fig, ax, title: str, output_path: Path, use_basemap: bool) -> Path:
    set_left_title(ax, title)
    ax.set_axis_off()
    add_north_arrow(ax, white_outline=use_basemap, outline_mm=0.5)
    if use_basemap:
        add_osm_copyright(ax)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Map saved: {output_path} (basemap={'OSM' if use_basemap else 'none'})")
    return output_path


def plot_categorical_hex_map(
    hexes: gpd.GeoDataFrame,
    municipality: gpd.GeoDataFrame,
    class_col: str,
    colors: dict[str, str],
    title: str,
    legend_title: str,
    output_path: Path,
    legend_order: Sequence[str] | None = None,
    alpha: float | None = None,
    figsize: tuple[float, float] = (14, 10),
    use_basemap: bool | None = None,
    legend_loc: str = "lower left",
    legend_offset_mm: tuple[float, float] = (0.0, 0.0),
) -> Path:
    """Classified choropleth; OSM basemap if filename contains 'osm'."""
    title = clean_map_title(title)
    fig, ax, hex_wm, muni_wm, boundary, output_path, use_basemap, alpha = _prepare_frame(
        hexes, municipality, output_path, use_basemap, alpha, figsize
    )

    hex_wm["map_color"] = hex_wm[class_col].map(colors).fillna("#d9d9d9")
    hex_wm.plot(ax=ax, color=hex_wm["map_color"], edgecolor="white", linewidth=0.18, alpha=alpha)
    muni_wm.boundary.plot(ax=ax, linewidth=0.55, edgecolor="#222222", alpha=0.9)
    boundary.boundary.plot(ax=ax, linewidth=BOUNDARY_LW, edgecolor="black")

    if legend_order is None:
        legend_order = ["Very high", "High", "Moderate", "Low", "Very low", "Not available"]
    present = set(hex_wm[class_col].dropna().astype(str).unique())
    order = [x for x in legend_order if x in present]
    for extra in sorted(present):
        if extra not in order:
            order.append(extra)

    patches = [
        mpatches.Patch(facecolor=colors.get(lab, "#d9d9d9"), edgecolor="gray", label=lab, alpha=alpha)
        for lab in order
    ]
    if patches:
        # Convert mm offset to axes-fraction using figure size (1 in = 25.4 mm)
        dx_mm, dy_mm = legend_offset_mm
        fx, fy = figsize
        dx = dx_mm / (25.4 * fx)
        dy = dy_mm / (25.4 * fy)
        # Anchor positions matching common loc keywords (axes fraction)
        anchors = {
            "lower left": (0.0, 0.0),
            "lower right": (1.0, 0.0),
            "upper left": (0.0, 1.0),
            "upper right": (1.0, 1.0),
            "center left": (0.0, 0.5),
            "center right": (1.0, 0.5),
        }
        base = anchors.get(legend_loc, (0.0, 0.0))
        bbox = (base[0] + dx, base[1] + dy)
        ax.legend(
            handles=patches,
            title=legend_title,
            loc=legend_loc,
            bbox_to_anchor=bbox,
            frameon=True,
            fontsize=9,
            title_fontsize=10,
        )

    return _finish_map(fig, ax, title, output_path, use_basemap)


def plot_numeric_hex_map(
    hexes: gpd.GeoDataFrame,
    municipality: gpd.GeoDataFrame,
    value_col: str,
    title: str,
    output_path: Path,
    cmap: str = "viridis",
    alpha: float | None = None,
    figsize: tuple[float, float] = (14, 10),
    legend_label: str | None = None,
    use_basemap: bool | None = None,
) -> Path:
    """Continuous choropleth; OSM basemap if filename contains 'osm'."""
    title = clean_map_title(title)
    fig, ax, hex_wm, muni_wm, boundary, output_path, use_basemap, alpha = _prepare_frame(
        hexes, municipality, output_path, use_basemap, alpha, figsize
    )

    hex_wm.plot(
        ax=ax,
        column=value_col,
        cmap=cmap,
        linewidth=0.12,
        edgecolor="white",
        alpha=alpha,
        legend=True,
        legend_kwds={"label": legend_label or value_col, "shrink": 0.55},
    )
    muni_wm.boundary.plot(ax=ax, linewidth=0.55, edgecolor="#222222", alpha=0.9)
    boundary.boundary.plot(ax=ax, linewidth=BOUNDARY_LW, edgecolor="black")

    return _finish_map(fig, ax, title, output_path, use_basemap)


def plot_custom_colored_hex_map(
    hexes: gpd.GeoDataFrame,
    municipality: gpd.GeoDataFrame,
    color_col: str,
    title: str,
    legend_handles: list,
    legend_title: str,
    output_path: Path,
    alpha: float | None = None,
    figsize: tuple[float, float] = (14, 10),
    use_basemap: bool | None = None,
    legend_loc: str = "lower left",
) -> Path:
    """Hex map where `color_col` already holds hex color strings."""
    title = clean_map_title(title)
    fig, ax, hex_wm, muni_wm, boundary, output_path, use_basemap, alpha = _prepare_frame(
        hexes, municipality, output_path, use_basemap, alpha, figsize
    )

    hex_wm.plot(ax=ax, color=hex_wm[color_col], edgecolor="white", linewidth=0.18, alpha=alpha)
    muni_wm.boundary.plot(ax=ax, linewidth=0.55, edgecolor="#222222", alpha=0.9)
    boundary.boundary.plot(ax=ax, linewidth=BOUNDARY_LW, edgecolor="black")

    if legend_handles:
        for h in legend_handles:
            try:
                h.set_alpha(alpha)
            except Exception:
                pass
        ax.legend(
            handles=legend_handles,
            title=legend_title,
            loc=legend_loc,
            frameon=True,
            fontsize=9,
            title_fontsize=10,
        )

    return _finish_map(fig, ax, title, output_path, use_basemap)
