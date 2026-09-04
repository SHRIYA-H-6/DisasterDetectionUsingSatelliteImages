"""
Fetch REAL earthquake pre/post satellite image pairs + REAL severity labels
from Maxar's Open Data Program (public, unauthenticated S3 bucket
`maxar-opendata`, CC BY-NC-4.0) for the 2023 Kahramanmaras, Turkiye
earthquake, and write them into data/raw/maxar_earthquake/ in the same
pre/post-pair + labels.csv format preprocessing.py already knows how to
scan (see preprocessing.py::scan_extra_source).

Unlike xBD, HuggingFace, Kaggle, or Zenodo, `s3.amazonaws.com` is reachable
from this sandbox, so this script actually runs here (see DATA_REPORT.md for
the full reachability check).

Real severity labels come from Maxar's own official building-change
GeoPackage layer (`events/.../building_change/{36,37}_Feb11_vis_change.gpkg`)
-- polygons of AI-detected building change between a pre-event and the
Feb 2023 post-event image, each tagged by Maxar with a change type: "rem"
(building removed), "och" (building change), "new" (new building), "nch"
(no change), or "nodata". For each quadkey we compute:

    damage_ratio = count(chdesc in {rem, och}) / count(chdesc != nodata)

This is a real, published, per-quadkey remote-sensing damage signal (not
invented by this project), and is used to bin quadkeys into
Low/Moderate/Severe via tertiles across the quadkeys actually processed
(a standard, defensible way to turn a continuous real signal into 3 ordinal
tiers without picking arbitrary absolute cutoffs).

Images: each quadkey's pre-event and post-event `-visual.tif` (a
Cloud-Optimized GeoTIFF) is read at a downsampled 1024x1024 resolution
directly over HTTP via GDAL's /vsicurl/ + COG overviews (so only ~a few MB
of range requests are fetched per image, not the full ~20-100MB source
file), then saved as PNG.

Usage:
    python fetch_maxar_earthquake.py --per-tier 20
"""
import argparse
import json
import os
import re
import sqlite3
import statistics
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import rasterio
from rasterio.enums import Resampling

import config

BUCKET = "https://maxar-opendata.s3.amazonaws.com"
EVENT = "Kahramanmaras-turkey-earthquake-23"
ZONES = ["36", "37"]
EARTHQUAKE_DATE = "2023-02-06"
OUT_DIR = os.path.join(config.ROOT_DIR, "data", "raw", "maxar_earthquake")
GPKG_CACHE = os.path.join(OUT_DIR, "_gpkg_cache")


def list_prefixes(prefix):
    url = f"{BUCKET}/?list-type=2&prefix={prefix}&delimiter=/"
    data = urllib.request.urlopen(url, timeout=30).read().decode()
    return re.findall(r"<Prefix>(.*?)</Prefix>", data)


def list_keys(prefix):
    url = f"{BUCKET}/?list-type=2&prefix={prefix}"
    data = urllib.request.urlopen(url, timeout=30).read().decode()
    return re.findall(r"<Key>(.*?)</Key>", data)


def download_gpkgs():
    os.makedirs(GPKG_CACHE, exist_ok=True)
    paths = {}
    for zone in ZONES:
        local = os.path.join(GPKG_CACHE, f"{zone}_Feb11_vis_change.gpkg")
        if not os.path.exists(local):
            url = f"{BUCKET}/events/{EVENT}/building_change/{zone}_Feb11_vis_change.gpkg"
            print(f"Downloading {url} ...")
            urllib.request.urlretrieve(url, local)
        paths[zone] = local
    return paths


def compute_damage_ratios(gpkg_paths):
    """Returns {(zone, quadkey): {bdate_options, adate, total, damage, ratio}}."""
    out = {}
    for zone, path in gpkg_paths.items():
        con = sqlite3.connect(path)
        cur = con.cursor()
        table = f"{zone}_Feb11_vis_change"
        cur.execute(f'SELECT DISTINCT quadkey FROM "{table}"')
        quadkeys = [r[0] for r in cur.fetchall()]
        for qk in quadkeys:
            cur.execute(f'SELECT bdate, adate, chdesc FROM "{table}" WHERE quadkey=?', (qk,))
            rows = cur.fetchall()
            bdates = sorted({r[0] for r in rows})
            adates = sorted({r[1] for r in rows})
            chdescs = [r[2] for r in rows]
            total = sum(1 for c in chdescs if c != "nodata")
            damage = sum(1 for c in chdescs if c in ("rem", "och"))
            if total == 0:
                continue
            out[(zone, qk)] = {
                "bdates": bdates, "adates": adates,
                "total": total, "damage": damage, "ratio": damage / total,
            }
    return out


def find_pre_post_dates(zone, qk):
    """Real acquisition dates available for this quadkey's ARD tile, from S3
    (not the gpkg, which only records the pair Maxar happened to diff)."""
    prefix = f"events/{EVENT}/ard/{zone}/{qk}/"
    prefixes = list_prefixes(prefix)
    dates = sorted({p.rstrip("/").split("/")[-1] for p in prefixes
                     if re.match(r"^\d{4}-\d{2}-\d{2}$", p.rstrip("/").split("/")[-1])})
    pre = [d for d in dates if d < EARTHQUAKE_DATE]
    post = [d for d in dates if d >= EARTHQUAKE_DATE]
    if not pre or not post:
        return None, None
    return pre[-1], post[0]  # most recent pre-event date, earliest post-event date


def find_visual_tif(zone, qk, date):
    prefix = f"events/{EVENT}/ard/{zone}/{qk}/{date}/"
    keys = list_keys(prefix)
    for k in keys:
        if k.endswith("-visual.tif"):
            return k
    return None


def read_cog_as_rgb(key, size=1024):
    url = f"/vsicurl/{BUCKET}/{key}"
    with rasterio.open(url) as src:
        arr = src.read(out_shape=(src.count, size, size), resampling=Resampling.bilinear)
    arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] >= 3:
        arr = arr[:, :, :3]
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def severity_from_ratio(ratio, thresholds):
    lo, hi = thresholds
    if ratio <= lo:
        return "Low"
    if ratio <= hi:
        return "Moderate"
    return "Severe"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-tier", type=int, default=20,
                         help="Max quadkeys to fetch per severity tier (Low/Moderate/Severe).")
    parser.add_argument("--image-size", type=int, default=1024)
    args = parser.parse_args()

    gpkg_paths = download_gpkgs()
    damage = compute_damage_ratios(gpkg_paths)
    print(f"{len(damage)} quadkeys have a real Maxar-derived damage ratio.")

    # Keep only quadkeys where the ARD tile actually has a real pre-event and
    # real post-event acquisition (checked against S3, not assumed).
    print("Checking which quadkeys have real pre+post imagery on S3 (this queries S3 for each)...")
    candidates = {}
    with ThreadPoolExecutor(max_workers=20) as ex:
        keys = list(damage.keys())
        results = ex.map(lambda k: (k, find_pre_post_dates(*k)), keys)
        for (zone, qk), (pre, post) in results:
            if pre and post:
                candidates[(zone, qk)] = {**damage[(zone, qk)], "pre_date": pre, "post_date": post}
    print(f"{len(candidates)} quadkeys have both a real pre-event and real post-event image.")

    ratios = sorted(v["ratio"] for v in candidates.values())
    t1 = statistics.quantiles(ratios, n=3)[0]
    t2 = statistics.quantiles(ratios, n=3)[1]
    print(f"Tertile thresholds on real damage_ratio: Low<={t1:.4f} Moderate<={t2:.4f} Severe>{t2:.4f}")

    for (zone, qk), v in candidates.items():
        v["severity"] = severity_from_ratio(v["ratio"], (t1, t2))

    by_tier = {"Low": [], "Moderate": [], "Severe": []}
    for k, v in candidates.items():
        by_tier[v["severity"]].append((k, v))
    selected = []
    for tier, items in by_tier.items():
        items.sort(key=lambda kv: kv[1]["ratio"])
        selected.extend(items[: args.per_tier])
    print(f"Selected {len(selected)} quadkeys to download "
          f"({ {t: min(len(v), args.per_tier) for t, v in by_tier.items()} }).")

    os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
    rows = []

    def fetch_one(item):
        (zone, qk), v = item
        pre_key = find_visual_tif(zone, qk, v["pre_date"])
        post_key = find_visual_tif(zone, qk, v["post_date"])
        if not pre_key or not post_key:
            return None
        sample_id = f"kahramanmaras-earthquake_{zone}_{qk}"
        pre_png = os.path.join("images", f"{sample_id}_pre_disaster.png")
        post_png = os.path.join("images", f"{sample_id}_post_disaster.png")
        try:
            pre_img = read_cog_as_rgb(pre_key, args.image_size)
            post_img = read_cog_as_rgb(post_key, args.image_size)
        except Exception as e:
            print(f"  FAILED {sample_id}: {e}")
            return None
        cv2.imwrite(os.path.join(OUT_DIR, pre_png), pre_img)
        cv2.imwrite(os.path.join(OUT_DIR, post_png), post_img)
        print(f"  OK {sample_id} severity={v['severity']} ratio={v['ratio']:.4f}")
        return {
            "sample_id": sample_id,
            "pre_image_path": pre_png,
            "post_image_path": post_png,
            "severity": v["severity"],
            "source": "Maxar Open Data (CC BY-NC-4.0) - Kahramanmaras Turkiye Earthquake 2023",
            "sensor": "Maxar WorldView/GeoEye ARD visual",
            "gsd": None,
            "capture_date": f"{v['pre_date']} -> {v['post_date']}",
            "damage_ratio": round(v["ratio"], 6),
        }

    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(fetch_one, selected):
            if r is not None:
                rows.append(r)

    import pandas as pd
    labels_path = os.path.join(OUT_DIR, "labels.csv")
    pd.DataFrame(rows).to_csv(labels_path, index=False)
    print(f"\nWrote {len(rows)} real earthquake pre/post samples to {labels_path}")
    for tier in ["Low", "Moderate", "Severe"]:
        n = sum(1 for r in rows if r["severity"] == tier)
        print(f"  Earthquake_{tier}: {n}")


if __name__ == "__main__":
    main()
