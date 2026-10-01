"""Stage 6: automated per-image summary report (console + .txt + .json)."""
import json

from . import config as C

# Rule-based recommended emergency action, one entry per combined class.
RECOMMENDED_ACTIONS = {
    "Flood_Low": "Monitor water levels and drainage; issue a flood advisory and pre-position sandbags and pumps.",
    "Flood_Moderate": "Evacuate low-lying homes, deploy boat rescue teams, open shelters and protect drinking-water supplies.",
    "Flood_Severe": "Order mass evacuation, launch air and boat search-and-rescue, set up emergency shelters, clean water and disease control.",
    "Earthquake_Low": "Inspect critical buildings, bridges and utilities for cracks; keep the public alert for aftershocks.",
    "Earthquake_Moderate": "Deploy structural assessment and urban search-and-rescue teams, cordon off unsafe buildings, set up field medical posts.",
    "Earthquake_Severe": "Activate full urban search-and-rescue, field hospitals, mass-casualty triage and temporary shelters; cut gas and power to collapsed areas.",
    "Tsunami_Low": "Keep people away from beaches and harbours, inspect coastal infrastructure, and stay on alert for further waves.",
    "Tsunami_Moderate": "Evacuate the coastal inundation zone to high ground, search damaged coastal buildings and clear debris from access routes.",
    "Tsunami_Severe": "Launch coastal search-and-rescue by air and sea, set up shelters inland, and provide clean water, food and medical care.",
    "Landslide_Low": "Close affected roads and slopes, monitor for ground movement and warn residents downslope.",
    "Landslide_Moderate": "Evacuate homes in the slide path, clear blocked roads with heavy machinery and restrict access to unstable slopes.",
    "Landslide_Severe": "Launch search-and-rescue for buried people, evacuate the surrounding slopes, and bring in heavy equipment and geotechnical teams.",
}
assert set(RECOMMENDED_ACTIONS) == set(C.COMBINED_CLASSES)


def build_report(pred):
    type_w = max(map(len, C.DISASTER_TYPES))
    sev_w = max(map(len, C.SEVERITIES))
    lines = [
        f"Predicted Disaster : {pred['disaster_type']} ({100 * pred['disaster_confidence']:.2f}% confidence)",
        f"Predicted Severity  : {pred['severity']} ({100 * pred['severity_confidence']:.2f}% confidence)",
        "",
        "Disaster Probabilities:",
        *[f"  - {name:<{type_w}} : {100 * p:.2f}%" for name, p in pred["disaster_probabilities"].items()],
        "",
        "Severity Probabilities:",
        *[f"  - {name:<{sev_w}} : {100 * p:.2f}%" for name, p in pred["severity_probabilities"].items()],
        "",
        f"Affected Area: {pred['affected_area_pct']:.2f}%",
        "",
        "Recommended Emergency Action:",
        f"  {RECOMMENDED_ACTIONS[pred['combined_class']]}",
    ]
    return "\n".join(lines)


def save_report(pred, text, stem, extra=None):
    """Write <stem>.txt and <stem>.json."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    stem.with_suffix(".txt").write_text(text + "\n")
    record = {k: v for k, v in pred.items() if k != "segmentation"}
    record["recommended_action"] = RECOMMENDED_ACTIONS[pred["combined_class"]]
    record.update(extra or {})
    stem.with_suffix(".json").write_text(json.dumps(record, indent=2, default=float))
