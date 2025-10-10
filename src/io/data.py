import os
import gzip
import json
import pickle
from typing import Dict, Tuple
import imn_loading


def read_poi_data(filepath: str) -> Dict[int, Dict]:
    poi_data = {}
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            poi_data[row['uid']] = row
    return poi_data


def load_datasets(paths, results_dir_cache: str = None) -> Tuple[Dict, Dict]:
    cache_path = os.path.join(paths.results_dir if results_dir_cache is None else results_dir_cache, "datasets_cache.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            imns = cached["imns"]
            poi_data = cached["poi"]
            print(f"✓ Loaded datasets from cache ({cache_path})")
            return imns, poi_data
        except Exception as e:
            print(f"⚠ Failed to load cache, re-reading datasets: {e}")

    full_imns = imn_loading.read_imn(paths.full_imn_path)
    filtered_user_ids = list(full_imns.keys())
    imns = imn_loading.read_imn(paths.test_imn_path)
    imns = {k: imns[k] for k in filtered_user_ids if k in imns}
    poi_data = read_poi_data(paths.poi_path)
    print(f"✓ Loaded {len(imns)} IMNs and POI data for {len(poi_data)} users")

    try:
        os.makedirs(paths.results_dir, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({"imns": imns, "poi": poi_data, "full_user_ids": filtered_user_ids}, f)
    except Exception as e:
        print(f"⚠ Could not cache datasets: {e}")
    return imns, poi_data


