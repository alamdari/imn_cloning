import os
import gzip
import json
import pickle
import random
from typing import Dict, Tuple
from src.io import imn_loading


def read_poi_data(filepath: str) -> Dict[int, Dict]:
    poi_data = {}
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            poi_data[row['uid']] = row
    return poi_data


def load_datasets(paths, results_dir_cache: str = None) -> Tuple[Dict, Dict]:
    cache_path = os.path.join(paths.results_dir if results_dir_cache is None else results_dir_cache, "datasets_cache.pkl")
    
    # Load full IMNs
    print(f"Loading IMNs from {paths.imn_path}...")
    full_imns = imn_loading.read_imn(paths.imn_path)
    print(f"  ✓ Loaded {len(full_imns)} IMNs")
    
    # Load POI data
    print(f"Loading POI data from {paths.poi_path}...")
    poi_data = read_poi_data(paths.poi_path)
    print(f"  ✓ Loaded POI data for {len(poi_data)} users")
    
    # Find users that have both IMN and POI data
    common_users = set(full_imns.keys()) & set(poi_data.keys())
    print(f"  ✓ Found {len(common_users)} users with both IMN and POI data")
    
    # Randomly sample num_users or use all users
    if paths.num_users is None:
        # Process all users
        sampled_user_ids = sorted(common_users)
        print(f"  → Processing ALL {len(sampled_user_ids)} users (no sampling)")
    elif len(common_users) > paths.num_users:
        # Sample requested number
        sampled_user_ids = random.sample(sorted(common_users), paths.num_users)
        print(f"  → Randomly sampled {paths.num_users} users for processing")
    else:
        # Use all available users (fewer than requested)
        sampled_user_ids = sorted(common_users)
        print(f"  → Using all {len(sampled_user_ids)} users (fewer than requested {paths.num_users})")
    
    # Filter to selected users
    imns = {uid: full_imns[uid] for uid in sampled_user_ids}
    poi_data_filtered = {uid: poi_data[uid] for uid in sampled_user_ids if uid in poi_data}
    
    print(f"✓ Ready to process {len(imns)} users\n")
    
    return imns, poi_data_filtered


