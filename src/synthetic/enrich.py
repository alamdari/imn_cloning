from typing import Dict
import numpy as np


def enrich_imn_with_poi(imn: Dict, poi_info: Dict) -> Dict:
    POI_TO_ACTIVITY = {
        "education": "school", "food_and_drink": "eat", "shopping": "shop",
        "entertainment_and_recreation": "leisure", "transportation": "transit",
        "healthcare": "health", "public_services": "admin", "finance": "finance",
        "utilities": "utility", "other": "unknown"
    }
    poi_classes = poi_info["poi_classes"]
    enriched = {}
    for loc_id, loc in imn["locations"].items():
        vec = poi_info["poi_freq"].get(loc_id, [0.0] * len(poi_classes))
        top_idx = int(np.argmax(vec))
        label = POI_TO_ACTIVITY.get(poi_classes[top_idx], "unknown")
        if loc_id == imn.get("home"):
            label = "home"
        if loc_id == imn.get("work"):
            label = "work"
        enriched[loc_id] = {**loc, "activity_label": label}
    imn["locations"] = enriched
    return imn


