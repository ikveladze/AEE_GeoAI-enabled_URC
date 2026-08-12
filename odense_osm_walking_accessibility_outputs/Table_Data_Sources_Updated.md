# Table — Primary data sources (updated, Odense AEE / functional URC study)

**Common framework:** Odense Municipality; 500 m hexagons (*n* = 1,539); CRS EPSG:25832  
**Table revised:** 2026-08-01  
**Purpose:** Manuscript-ready overview of input data layers (companion to Appendix Table A1).

---

## Primary data layers (manuscript overview)

| Data layer | Details | Source | Year / reference date |
| --- | --- | --- | --- |
| BBR buildings | Address-linked building points with residential floor area (`BYG_BOLIG_`) used as dasymetric weights | Danish Building and Dwelling Register (BBR); local extract `OD Buildings_2020` | 2020 |
| GeoDanmark buildings / built-up | Building footprints and built-up land-cover classes (bygning, bykerne, høj-/lavbebyggelse, erhverv, etc.) | Klimadatastyrelsen & municipalities (GeoDanmark); Datafordeler / Dataforsyningen | Product currency ~2023; downloaded 2026-07-29 |
| GeoDanmark land use / land cover | Natural/green classes (skov, vådområde, kratbevoksning, …), water (sø, bassin), and related land-cover polygons | Klimadatastyrelsen & municipalities (GeoDanmark); Datafordeler / Dataforsyningen | Product currency ~2023; downloaded 2026-07-29 |
| GeoDanmark road network | Road centre-lines (`vejmidte`) and attributes; major roads used for exposure buffers | Klimadatastyrelsen & municipalities (GeoDanmark); Datafordeler / Dataforsyningen | Product currency ~2023; downloaded 2026-07-29 |
| OpenStreetMap (OSM) | Walking/cycling networks, service amenities (POIs), public-transport stops/stations | OpenStreetMap contributors; retrieved with OSMnx/Overpass (Geofabrik community extracts as fallback) | Snapshot at retrieval; accessibility analysis GPKGs 2026-06-01 |
| StatBank parish / postcode population | Parish (SOGN) and postcode (POSTNR) population counts for dasymetric allocation; FOLK1A municipal control total | Statistics Denmark StatBank | SOGN/FOLK extracts 2022–2026; FOLK1A ≈ 213,431 |
| DST Kvadratnet density classes | 1 km density *interval classes* (persons/km²) used as soft spatial constraint / validation (not paid person counts) | Statistics Denmark Kvadratnet extranet | ~2019 |
| Protected / §3 nature | Protected areas (WDPA) and Danish §3 / areal nature polygons | UNEP-WCMC Protected Planet (DNK); Landbrugsstyrelsen (LBST) via geodata.fvm.dk | WDPA Jul 2026; LBST pull 2026-07-29 |
| Summer land-surface temperature | Mean summer LST for environmental composites | ESA CLIM4cities Sentinel-3 SLSTR downscaled LST (Zenodo) | Scenes Jun–Aug 2020–2023; processed 2026-07-29 |
| State-road traffic noise (Lden) | Noise bands along state roads | Vejdirektoratet — Støjkortlægning af statsveje | Mapping year 2022; downloaded 2026-07-29 |
| Air pollution proxy (NO₂) | Interim regional NO₂ surface (to be replaced by DCE AirGIS if available) | Open-Meteo Air Quality API (CAMS-based) | Pulled 2026-07-29 |

---

## Derived demographic product (not a raw download)

| Data layer | Details | Construction | Reference date |
| --- | --- | --- | --- |
| Dasymetric population (500 m hex) | Preferred population surface for density and age shares | StatBank SOGN × BBR `BYG_BOLIG_` weights; soft-constrained to DST 1 km density classes; scaled to FOLK1A; ages from POSTNR1 × BBR | Hex product 2026-07-31 |

---

## Explicitly excluded from v1/v2 analysis

| Data layer | Reason |
| --- | --- |
| Non-Western-origin / migration origin fields | Excluded by design from indicators, vulnerability, and clustering |

---

## Notes for the manuscript text

1. GeoDanmark is maintained by **Klimadatastyrelsen** with the municipalities and was accessed via **Datafordeler** / **Dataforsyningen** (Klimadatastyrelsen platform).  
2. Population density is a **dasymetric estimate**, not paid DST Kvadratnet person counts.  
3. Public-transport accessibility is **stop-access potential** (OSM), not GTFS timetable routing.  
4. For full indicator definitions and hex-level construction, see `Appendix_Table_A1_Data_Sources_v1.md`.
