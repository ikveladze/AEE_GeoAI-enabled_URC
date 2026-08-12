# Data licensing and attribution (Odense AEE / URC reproducibility package)

This package mixes open geodata, municipal extracts, and derived research products.
**Always check the original provider licence before commercial reuse or bulk redistribution.**

## Attribution by source

| Source | Used for | Typical attribution / notes |
|--------|----------|-----------------------------|
| OpenStreetMap contributors | Walk/bike networks, POIs, PT stops | © OpenStreetMap contributors — ODbL. Retrieved via OSMnx/Overpass. |
| GeoDanmark (Datafordeler / Dataforsyningen / Klimadatastyrelsen) | Natural, blue, built-up, roads | Danish public geodata terms apply; cite GeoDanmark / GEODKV. |
| UNEP-WCMC Protected Planet (WDPA) | Protected areas | Cite WDPA / Protected Planet as required by UNEP-WCMC. |
| Landbrugsstyrelsen (LBST) | §3 nature | Danish agricultural/nature geodata terms. |
| Vejdirektoratet | State-road noise Lden 2022 | Cite Danish Road Directorate noise mapping. |
| ESA CLIM4cities / Zenodo | Summer land-surface temperature | Cite Zenodo dataset DOI `10.5281/zenodo.20863040` and ESA as applicable. |
| Open-Meteo / Copernicus CAMS | Interim NO₂ proxy | Interim only; not a substitute for Danish national air mapping. |
| Municipal Basisdata_o (`KOMM_TOT_21`) | Population control total 2021 | Municipal register extract — redistribute only under your institutional rights. |
| Population density class map | Morphology / density estimate | Project GIS layer with DST-style density classes — not paid DST Kvadratnet counts. |

## What we redistribute here

- **Analysis-ready composites** and **hex-level derived indicators** needed to reproduce the paper’s Stage B results.
- Scripts and Appendix A1 documentation.

## What we do not claim

- We do not claim ownership of OSM, GeoDanmark, WDPA, LBST, Vejdirektoratet, or ESA products.
- Population density on hexes is a **research estimate** (class midpoints scaled to municipal total), not Statistics Denmark official Kvadratnet person counts.
- Air quality in v1 is an **interim proxy**.

## Recommended citation

Cite the *MDPI Urban Science* article (when published) and this repository (commit hash or release tag).
