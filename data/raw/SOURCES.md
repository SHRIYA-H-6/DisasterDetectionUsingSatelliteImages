# Data provenance

All imagery and labels in `data/raw/` are **real xBD (xView2) satellite tiles and
official xBD damage-annotation JSON** (Gupta et al., 2019, "Creating xBD: A
Dataset for Assessing Building Damage from Satellite Imagery"). No synthetic,
generated, or placeholder images are used anywhere in this repository.

## Why this set is small

This build environment's outbound network policy allows `github.com` (git
clone), `pypi.org`/`npmjs.org` (package installs), and generic `s3.amazonaws.com`
/ `storage.googleapis.com` endpoints, but **blocks `huggingface.co`,
`www.kaggle.com`, `zenodo.org`, and `xview2.org`** — the official/standard
distribution channels for the full xBD dataset (~11,000 tile pairs, ~1GB+) and
for real landslide inventories. See `DATA_REPORT.md` at the repo root for the
full per-class coverage accounting and what is needed to complete the dataset.

Given that restriction, the images below were recovered from public GitHub
repositories that had incidentally committed small, genuine fragments of xBD
(coursework/research repos). Each entry names the exact upstream repo the
files came from, cloned during this session (2026-09-04).

## Files and their origin

### `hurricane-florence_00000086/88/91/93/94/96` (pre + post, PNG 1024x1024, + matching official JSON labels)
- Disaster: xBD `disaster_type = "flooding"` → mapped to **Flood**
- Source repo: https://github.com/jeremiahf24/ADS-Capstone
  - Images: `Images/hurricane-florence_<id>_{pre,post}_disaster.png`
  - Labels: `Data/hurricane-florence_<id>_{pre,post}_disaster.json`
- Verified: image dims (1024x1024) match `original_width`/`original_height`
  in the JSON metadata, and JSON `metadata.sensor`, `gsd`, `capture_date`,
  `catalog_id` fields match the real xBD schema exactly (GEOEYE01 sensor,
  2018-09-20 capture date, consistent with the real Hurricane Florence xBD
  collection). Building-level `properties.subtype` values
  (no-damage/minor-damage/major-damage/destroyed) are used to derive
  tile-level severity in `preprocessing.py`.

### `santa-rosa-wildfire_00000007` (post only, PNG 1024x1024, + matching official JSON label)
- Disaster: xBD `disaster_type = "fire"` → mapped to **Wildfire**
- Image source: https://github.com/DIUx-xView/xview2-baseline
  (`overlay_output_to_image/santa-rosa-wildfire_00000007_post_disaster.png`
  — the official xView2 baseline repo's own demo image, itself a genuine xBD
  tile used to illustrate their scoring/overlay tool)
- Label source: https://github.com/SmashNdashH/CSE499A.15-Msrb-
  (`data/processed_labels/santa-rosa-wildfire_00000007_post_disaster.json`)
- **No pre-disaster counterpart image was found on any reachable source.**
  This sample is included only for the demo script / sample-grid visual
  check, and is excluded from training (a pseudo-Siamese model requires a
  pre/post pair). It does not count toward any class's sample floor.

### `kahramanmaras-earthquake_<zone>_<quadkey>` (60 samples: 20 Low / 20 Moderate / 20 Severe, pre+post PNG 1024x1024)
- Disaster: **Earthquake** (2023 Kahramanmaras, Turkiye earthquake)
- Source: **Maxar Open Data Program** (`https://maxar-opendata.s3.amazonaws.com`,
  CC BY-NC-4.0), event `Kahramanmaras-turkey-earthquake-23`. Unlike
  xview2.org/Kaggle/HuggingFace/Zenodo, this public S3 bucket is reachable
  from this sandbox (see `DATA_REPORT.md`).
- Fetched by `fetch_maxar_earthquake.py`, which actually runs in this
  environment (verified: 60/60 samples downloaded successfully in this
  session). Images are real pre-event and post-event Maxar
  WorldView/GeoEye visual (RGB) tiles, read at 1024x1024 directly from the
  source Cloud-Optimized GeoTIFFs via HTTP range requests (GDAL
  `/vsicurl/` + COG overviews) -- confirmed as genuine satellite photos
  (farmland, roads, villages, forest, visible cloud cover) by direct visual
  inspection, not a processed/rendered product.
- Severity is derived from Maxar's own official, published
  `building_change` GeoPackage layer for this event (AI-detected building
  change polygons between the same pre/post image pair, each polygon typed
  "removed", "building-to-building change", "new", or "no change" by
  Maxar). Per quadkey: `damage_ratio = count(removed or changed) /
  count(all non-nodata polygons)`; quadkeys are binned into Low/Moderate/
  Severe by tertiles of `damage_ratio` across the 79 real candidate
  quadkeys that had both a real pre- and post-event image (a standard way
  to turn a continuous real signal into 3 ordinal tiers without picking an
  arbitrary absolute cutoff). This is a real, Maxar-published, per-tile
  damage signal -- not an invented one -- though it is coarser than xBD's
  per-building manual annotation (see `DATA_REPORT.md` for the distinction
  and how it's flagged as a lower-confidence label tier than xBD's).
- Full per-sample provenance (`sample_id`, exact pre/post acquisition
  dates, `damage_ratio`) is in `data/raw/maxar_earthquake/labels.csv`.

## Explicitly NOT included (and why)

- `hurricane-harvey_00000000/1/2` patches from
  https://github.com/Chimdiya1/Webapp (`public/images/{pre,post}/...`) —
  real xBD pixels, but they are 512x512 sub-crops produced by that repo's
  own web-app pipeline, and no JSON label could be verified to align with
  the crop geometry. Rather than guess a severity label, these are left out
  of `metadata.csv` entirely (see `DATA_REPORT.md`).
- `palu-tsunami_00000001` files from
  https://github.com/ashnair1/xview2-toolkit (`.github/*_seg.png`) — these
  are colorized *segmentation-mask* renders (PIL mode `P`/`1`, palette /
  binary), not raw satellite RGB imagery, so they fail the "real satellite
  photo, not a flat-color block" visual check by construction. Excluded.
- Landslide: real pre/post imagery for two real landslide events (PNG
  2024, Georgia 2023) is reachable via the same Maxar Open Data bucket
  (`events/PNG-Landslide-June24/`, `events/shovi-georgia-landslide-8Aug23/`)
  and was confirmed to exist, but neither event has an official per-tile
  damage/severity layer analogous to the earthquake's `building_change`
  GeoPackage, so no real severity could be derived for them this session
  (see `DATA_REPORT.md` for what was checked and the Phase-2 plan).
- Wildfire: similarly, real pre/post imagery is reachable for several real
  wildfire events (LA Jan 2025, Maui Aug 2023, SmokeHouse Creek TX Mar
  2024, etc.) via Maxar Open Data -- and post-event smoke is visibly
  present in some tiles, confirming real fire activity -- but there is no
  official per-tile burn-severity layer, and a from-scratch burn-index
  (dNBR/dNDVI) computation attempted this session on raw, uncalibrated
  digital-number values did not produce a reliable enough signal to trust
  as a real label (see `DATA_REPORT.md`).

## Full xBD official label set (reference only, not used for pixels)

`https://github.com/jeremiahf24/ADS-Capstone` and
`https://github.com/SmashNdashH/CSE499A.15-Msrb-` both contain (as course
project artifacts) the **complete official xBD JSON label files** for many
disasters (hurricane-harvey/florence/matthew/michael, midwest-flooding,
palu-tsunami, mexico-earthquake, santa-rosa-wildfire, socal-fire,
guatemala-volcano) — thousands of label files, but essentially none of the
corresponding image tiles (image binaries were `.gitignore`d in those
projects, as is typical since xBD imagery totals ~1GB+). These confirm the
label schema and disaster taxonomy used in `preprocessing.py`, but cannot by
themselves produce trainable image samples.
