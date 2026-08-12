# Appendix Table A1 — Indicator data sources (v1 re-analysis, Odense)

**Analysis folder:** `odense_osm_walking_accessibility_outputs_1`  
**Common framework:** Odense Municipality; **500 m hexagons**, **n = 1,539**; CRS **EPSG:25832**; municipal area **304.37 km²**  
**Table revised:** 2026-07-29  

**Notes**
1. Walking accessibility is computed at **15 min only** (no 5-min walking indicator in the code).  
2. GeoDanmark layers were retrieved via **Datafordeler WFS (GEODKV)** / **Dataforsyningen**. **Klimadatastyrelsen** is the authority for Datafordeler and co-produces GeoDanmark with the municipalities (Klimadatastyrelsen platform).  
3. Population density is **not** paid DST Kvadratnet person counts; it is a **dasymetric estimate** from StatBank parish/postcode counts + BBR housing floor area, soft-checked against DST 1 km density classes (see Exposure row).  
4. Exact OSM Overpass retrieval timestamp was not stored in code; accessibility GPKGs date from the **2026-06-01** analysis run (OSMnx live download).

---

| Domain | Final indicator | Definition | Exact dataset and layer | Provider | Year/download date | Native geometry/resolution | Hexagon-level construction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Accessibility | Walking service accessibility, 15 min | Decayed walking-network access to weighted daily-service categories (food retail, health, education/childcare, food/drink, civic/daily services, recreation) | OpenStreetMap Denmark via OSMnx (`network_type=walk`; POI/polygon tags in Appendix A1 variable dictionary); study clip = Odense municipality | OpenStreetMap / Geofabrik community data; retrieved with OSMnx/Overpass | OSM snapshot at retrieval; analysis GPKG **2026-06-01** (re-used in v1, 2026-07-29) | Line network + point/polygon destinations | Multi-source Dijkstra from services to hex centroids; \(e^{-d/700}\); walking speed **1.34 m/s (4.8 km/h)**; **15 min** cutoff (≈1,206 m); category-weighted composite min–max scaled 0–100 |
| Accessibility | Cycling service accessibility, 5 min | Decayed cycling-network access to the same service taxonomy | OSM via OSMnx (`network_type=bike`); same destination tags as walking | OpenStreetMap; OSMnx/Overpass | Analysis GPKG **2026-06-01** (re-used 2026-07-29) | Line bicycle network + destinations | Nearest-service network distance; \(e^{-d/1800}\); cycling speed **15 km/h**; **5 min** cutoff (≈1,250 m); min–max 0–100 |
| Accessibility | Cycling service accessibility, 15 min | Same, with 15-min cutoff | Same as above | Same | Same | Same | Same calculation with **15 min** cutoff (≈3,750 m) |
| Accessibility | PT stop-access potential, 10/20 min | First-/last-mile walking access to public-transport stops/stations (not timetable journey times) | OSM public transport features via OSMnx (bus stop, platform, rail/tram station, transport hub tags; Appendix A1) | OpenStreetMap; OSMnx/Overpass | Analysis GPKG **2026-06-01** (re-used 2026-07-29) | Point (stop/station) features; walk network | Walking-network access to stops; speed **1.34 m/s**; \(\beta=1000\) m; cutoffs **10 min** (≈804 m) and **20 min** (≈1,608 m); 50/50 all-stops vs category-weighted composite |
| Environment | Green-cover share | Proportion of cell covered by official natural/green land-cover classes | GeoDanmark composite `official_landcover_natural.gpkg` (from `geodanmark_odense_official.gpkg`: skov, vådområde, kratbevoksning, and related natural classes; see download summary) | Agency for Data Supply and Infrastructure / **GeoDanmark** via **Datafordeler** (GEODKV WFS); portal: Dataforsyningen / Klimadatastyrelsen | Product currency per GeoDanmark registration; **downloaded 2026-07-29** | Polygon (dissolved multipolygon for Odense clip) | Intersected green/natural area ÷ clipped hexagon area × 100 (`green_share_pct`) |
| Environment | Blue-cover share | Proportion of cell covered by water bodies | GeoDanmark composite `official_blue.gpkg` (sø, bassin, and related water features from GEODKV) | GeoDanmark / Datafordeler (GEODKV); Dataforsyningen / Klimadatastyrelsen portal | **Downloaded 2026-07-29** | Polygon | Intersected blue area ÷ hexagon area × 100 (`blue_pct`) |
| Environment | Protected/open-land share | Share of cell in protected / §3 nature | Merged protected layer `protected_nature.gpkg` / `ecological_health_vector.gpkg` = **WDPA** (Protected Planet, DNK extract) ∪ **LBST §3 / areal** (`paragraf3_protected_nature.gpkg`, `lbst_odense_official.gpkg`) | UNEP-WCMC / Protected Planet; Landbrugsstyrelsen (LBST) via geodata.fvm.dk | WDPA DNK extract **Jul 2026**; LBST pull **2026-07-29** | Polygon | Intersected protected area ÷ hexagon area × 100 (`eco_pct`); mean ≈7.6% |
| Environment | Built-up intensity | Built-up share of each cell | GeoDanmark composite `official_builtup.gpkg` (bygning, bykerne, høj-/lavbebyggelse, erhverv, etc.) | GeoDanmark / Datafordeler (GEODKV); Dataforsyningen / Klimadatastyrelsen portal | **Downloaded 2026-07-29** | Polygon | Intersected built-up area ÷ hexagon area × 100 (`builtup_pct`) |
| Environment | Major-road exposure | Share of cell within buffer of selected major roads | GeoDanmark `official_roads.gpkg` (`vejmidte`); filtered to major `trafikart`/`vejkategori` (motorway, hovedrute, etc.) | GeoDanmark / Datafordeler (GEODKV) | **Downloaded 2026-07-29** | Line | **50 m** buffer around major road centre-lines; buffered area ∩ hex ÷ hex area × 100 (`road_buf_pct`) |
| Environment (supporting) | Summer land-surface temperature | Mean summer LST used in environmental quality/burden composites | `heat_raster.tif` from ESA **CLIM4cities** Sentinel-3 SLSTR downscaled LST (Zenodo 10.5281/zenodo.20863040) | ESA / Zenodo CLIM4cities | Scenes **Jun–Aug 2020–2023**; processed **2026-07-29** | Raster ~**100 m**, EPSG:25832 | Zonal mean per hex (`heat_mean`); LST not air temperature |
| Environment (supporting) | Road-traffic noise (Lden) | State-road noise exposure | `noise_vector.gpkg` / `noise_raster.tif` — Vejdirektoratet **Støjkortlægning af statsveje 2022** | Vejdirektoratet | Mapping year **2022**; downloaded **2026-07-29** | Polygon bands + 50 m raster | Vector area share and/or zonal mean dB (`noise_share`, `noise_mean`); **state roads only** |
| Environment (supporting, interim) | Air pollution proxy (NO₂) | Interim regional NO₂ surface (not primary DCE product) | `air_quality_raster.tif` — Open-Meteo Air Quality API (CAMS-based) | Open-Meteo / Copernicus CAMS (interim) | Pulled **2026-07-29** | Raster ~hex-grid scale | Zonal mean (`air_pollution_mean`); **to be replaced by DCE “Luften på din vej” / AirGIS** |
| Exposure | Population density | Dasymetric population concentration (morphology/equity proxy) | Preferred `dasymetric_population_hex500_v2.gpkg`: StatBank **SOGN** parish counts × BBR residential **`BYG_BOLIG_`** weights; soft-constrained to DST Kvadratnet extranet 1 km density classes; scaled to StatBank **FOLK1A ≈ 213,431**. Age 0–14 / 65+ from **POSTNR1** × BBR. DST classes used as ground-truth check (not paid person counts). | Statistics Denmark StatBank (SOGN/POSTNR/FOLK); BBR 2020 buildings; DST Kvadratnet extranet classes; DAWA/DAGI parish/postcode geometries; Basisdata KOMM_TOT_21 age structure | SOGN/FOLK **2026**; BBR **2020**; DST classes ~**2019**; KOMM_TOT ages **2021**; hex product **2026-07-31** | Parish/postcode polygons + building points → 500 m hex | Floor-area-weighted dasymetric within parish (pycnophylactic); DST class soft-constraint; density = persons / hex km² |
| Exposure | Non-Western-origin count (proxy) | *Excluded from v1/v2 analysis by design* | `odense_1km_ikkevestlig_turkey_2021.gpkg` retained in folder only | Municipal/Basisdata_o migration GIS extract (Odense) | AAR_2021 | 1 km grid | **Not used** in hex indicators, vulnerability, or clustering |

---

## Not available at hex level (documented only at municipality)

| Indicator | Available as | Source | Year |
| --- | --- | --- | --- |
| Fine official age–sex / income / education at hex | Municipality / postcode / parish only (no paid 100 m grid) | Basisdata KOMM_TOT_21; StatBank FOLK/POSTNR/HFUDD/INDKP; OMRP stock without geometry | 2021–2026 |
| Paid DST Kvadratnet persons (100 m) | Not obtained (commercial) | Statistics Denmark Kvadratnet Natbefolkning | — |

---

## File locations (v1 / demography v2)

- Accessibility GPKGs: `odense_osm_walking_accessibility_outputs_1/` (+ cycling/PT subfolders)  
- Official env: `Map Layers/Official Environmental/`  
- Demographics (preferred): `Map Layers/Demographic/dasymetric_population_hex500_v2.gpkg`  
- Demographics (methods): `odense_demographic_dasymetric_v2_outputs/`  
- Full pipeline notes: `odense_osm_walking_accessibility_outputs_1/STATUS_V1.md`
