"""Shared constants: class names, severity thresholds, paths and image sizes."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("DISASTER_DATA_ROOT", PROJECT_ROOT.parent / "xbd-data"))
OUTPUT_ROOT = Path(os.environ.get("DISASTER_OUTPUT_ROOT", PROJECT_ROOT / "outputs"))
CACHE_DIR = OUTPUT_ROOT / "cache"
METADATA_CSV = PROJECT_ROOT / "data" / "metadata.csv"
GEOTRANSFORMS_ARCHIVE = DATA_ROOT / "xview_geotransforms.json.tgz"

DISASTER_TYPES = ["Flood", "Earthquake", "Tsunami", "Landslide"]
SEVERITIES = ["Low", "Moderate", "Severe"]
COMBINED_CLASSES = [f"{d}_{s}" for d in DISASTER_TYPES for s in SEVERITIES]

# xBD event -> project disaster type.
XBD_EVENTS = {
    "midwest-flooding": "Flood",
    "mexico-earthquake": "Earthquake",
    "palu-tsunami": "Tsunami",
}
# BRIGHT (Chen et al., 2025) events containing this string are used as Earthquake samples:
# xBD's only earthquake (mexico-earthquake) has no moderately/severely damaged tiles.
BRIGHT_EVENT_FILTER = "earthquake"

# Severity from the fraction of damaged buildings among classified buildings in an
# xBD tile (minor/major/destroyed) or of damaged building pixels in a BRIGHT tile
# (damaged/destroyed): Low <= 10% < Moderate <= 30% < Severe.
XBD_SEVERITY_EDGES = (0.10, 0.30)
# Severity from the fraction of landslide pixels in a Landslide4Sense patch.
LANDSLIDE_SEVERITY_EDGES = (0.05, 0.20)

MIN_SAMPLES_FLOOR = 10
MIN_SAMPLES_TARGET = 15

# xBD target PNG pixel codes.
XBD_BACKGROUND, XBD_NO_DAMAGE, XBD_MINOR, XBD_MAJOR, XBD_DESTROYED, XBD_UNCLASSIFIED = range(6)

# Segmentation classes predicted by the U-Net.
SEG_BACKGROUND, SEG_INTACT, SEG_AFFECTED = 0, 1, 2
SEG_CLASS_NAMES = ["background", "intact structure", "affected (damaged / landslide)"]

IMAGE_SIZE = 256  # tiles are resized to this for both models
SEED = 42

# Landslide4Sense 14-band layout: B1..B12 Sentinel-2, B13 slope, B14 DEM (0-based RGB = B4, B3, B2).
L4S_RGB_BANDS = (3, 2, 1)


def severity_from_fraction(fraction, edges):
    low, high = edges
    if fraction <= low:
        return "Low"
    if fraction <= high:
        return "Moderate"
    return "Severe"
