from typing import Dict
import numpy as np


def enrich_imn_with_poi(imn: Dict, poi_info: Dict) -> Dict:
    # POI classes now use activity labels directly (transit, health, admin, etc.)
    # No conversion needed - they match activity_pools.py labels
    poi_classes = poi_info["poi_classes"]
    enriched = {}
    for loc_id, loc in imn["locations"].items():
        vec = poi_info["poi_freq"].get(loc_id, [0.0] * len(poi_classes))
        top_idx = int(np.argmax(vec))
        label = poi_classes[top_idx] if top_idx < len(poi_classes) else "unknown"
        # Map 'other' to 'unknown' for consistency
        if label == "other":
            label = "unknown"
        if loc_id == imn.get("home"):
            label = "home"
        if loc_id == imn.get("work"):
            label = "work"
        enriched[loc_id] = {**loc, "activity_label": label}
    imn["locations"] = enriched
    return imn


