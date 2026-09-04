"""Shared constants for the disaster-detection pipeline."""
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_XBD_DIR = os.path.join(ROOT_DIR, "data", "raw", "xbd")
RAW_XBD_IMAGES_DIR = os.path.join(RAW_XBD_DIR, "images")
RAW_XBD_LABELS_DIR = os.path.join(RAW_XBD_DIR, "labels")
RAW_LANDSLIDE_DIR = os.path.join(ROOT_DIR, "data", "raw", "landslide")
RAW_MAXAR_EARTHQUAKE_DIR = os.path.join(ROOT_DIR, "data", "raw", "maxar_earthquake")
METADATA_CSV = os.path.join(ROOT_DIR, "data", "metadata.csv")
SPLITS_DIR = os.path.join(ROOT_DIR, "data", "splits")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
METRICS_DIR = os.path.join(OUTPUTS_DIR, "metrics")
FIGURES_DIR = os.path.join(OUTPUTS_DIR, "figures")
SAMPLES_DIR = os.path.join(OUTPUTS_DIR, "samples")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "disaster_model.pt")

# The 4 disaster types this project detects.
DISASTER_TYPES = ["Flood", "Earthquake", "Wildfire", "Landslide"]

# The 3 severity tiers, derived from xBD building-damage subtypes.
SEVERITY_LEVELS = ["Low", "Moderate", "Severe"]

# All 12 disaster+severity joint classes, in the order required for reporting
# (Flood -> Earthquake -> Wildfire -> Landslide, each Low -> Moderate -> Severe).
JOINT_CLASSES = [
    f"{d}_{s}" for d in DISASTER_TYPES for s in SEVERITY_LEVELS
]

# Minimum real samples required per joint class before training is allowed.
MIN_SAMPLES_TARGET = 15   # target (15-20 per the project spec)
MIN_SAMPLES_FLOOR = 10    # absolute floor

# Maps the official xBD `metadata.disaster_type` field to our 4 disaster types.
# xBD covers wind/flooding/fire/earthquake/tsunami/volcano; tsunami damage in
# xBD (Palu, Indonesia 2018) was earthquake-triggered, so it is folded into
# Earthquake. "wind" (pure hurricane wind damage, no flooding) has no home in
# our 4 classes and is left unmapped (excluded) rather than force-fit.
XBD_DISASTER_TYPE_MAP = {
    "flooding": "Flood",
    "earthquake": "Earthquake",
    "tsunami": "Earthquake",
    "fire": "Wildfire",
    "volcano": None,
    "wind": None,
}

# xBD per-building damage subtype -> severity tier used for majority-vote
# tile-level severity. 'un-classified' buildings are excluded from the vote.
XBD_SUBTYPE_TO_SEVERITY = {
    "no-damage": "Low",
    "minor-damage": "Moderate",
    "major-damage": "Severe",
    "destroyed": "Severe",
}

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

RANDOM_SEED = 42
