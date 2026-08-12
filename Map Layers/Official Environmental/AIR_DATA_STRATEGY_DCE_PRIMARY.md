# Air pollution data strategy (AEE Odense)

**Decision (2026-07-29):** Use **DCE “Luften på din vej” / AirGIS** as the **primary** air indicator. Use Copernicus CAMS / NASA–Sentinel-5P only as **validation / context**, not as the main clustering input.

## Primary (to obtain)

| Item | Detail |
|------|--------|
| Product | Luften på din vej 2.0 (Air Quality at Your Street) |
| Provider | DCE – Danish Centre for Environment and Energy, Aarhus University |
| Portal | https://lpdv.spatialsuite.dk/spatialmap |
| Preferred variables | Annual mean **NO₂** (primary); optionally **PM2.5** |
| Year | 2019 (v2.0) unless newer extract available |
| Model | DEHM/UBM/AirGIS + OSPM |
| Target format for analysis | GeoTIFF or address points → hex mean → `air_quality_raster.tif` (EPSG:25832) |
| Polarity | Higher = worse air quality (burden) |

## Temporary placeholder (until DCE arrives)

| File | Role |
|------|------|
| `air_quality_raster.tif` (Open-Meteo / CAMS) | Interim only; **do not** claim as DCE in manuscript |
| `air_quality_pm25_raster.tif` | Same |

When DCE data arrive: replace `air_quality_raster.tif`, re-run environmental index + typology, update Appendix A1.

## Validation / context (not main GMM features)

| Source | Use in paper |
|--------|----------------|
| Copernicus CAMS / Open-Meteo | Mention that regional patterns are consistent with street-level DCE gradients (optional map/scatter in Appendix) |
| Sentinel-5P / NASA column NO₂ | Optional Discussion: satellite columns support urban–rural NO₂ contrast at coarser scale |
| EEA air pollution topic pages | Policy/context citations only |

## Manuscript wording (suggested)

> Street-level annual mean NO₂ was obtained from DCE Air Quality at Your Street / AirGIS (Luften på din vej). Coarser Copernicus Atmosphere (CAMS) and satellite NO₂ products were consulted only for contextual comparison and are not used as primary clustering inputs, because their spatial resolution is insufficient to resolve intra-municipal variation at the 500 m hexagon scale.

## Contact

Request research extract for **Odense Municipality** (kommunekode 0461 / bounding box EPSG:25832). See `DCE_AIR_DATA_REQUEST_EMAIL.txt`.
