#!/usr/bin/env python3
"""
Convert Milano2007_ORIGINAL_combined.csv to trajectory format based on IMN trips.

This script:
1. Loads IMN data (which contains trip segments per user)
2. Loads the original Milano GPS CSV
3. Segments GPS points into trajectories based on trip start/end times
4. Outputs trajectory CSV files compatible with quality evaluation
"""

import pandas as pd
import imn_loading
import os
from collections import defaultdict

# Configuration
MILANO_CSV = "data/Milano2007_ORIGINAL_combined.csv"
IMN_FILE = "data/milano_2007_imns.json.gz"
OUTPUT_DIR = "data/original_trajectories"

def main():
    print("="*60)
    print("Milano GPS to Trajectory Converter")
    print("="*60)
    
    # Load IMNs
    print(f"\n1. Loading IMNs from {IMN_FILE}...")
    imns = imn_loading.read_imn(IMN_FILE)
    print(f"   ✓ Loaded {len(imns)} IMNs")
    
    # Load Milano GPS CSV
    print(f"\n2. Loading GPS data from {MILANO_CSV}...")
    df = pd.read_csv(MILANO_CSV)
    print(f"   ✓ Loaded {len(df)} GPS points")
    print(f"   Columns: {list(df.columns)}")
    
    # Rename columns to standard format
    df = df.rename(columns={
        'id': 'user_id',
        'longitude': 'lon',
        'latitude': 'lat',
        'timestamp': 'time'
    })
    
    # Sort by user and time
    df = df.sort_values(['user_id', 'time']).reset_index(drop=True)
    print(f"   ✓ Sorted by user and time")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Process each user
    print(f"\n3. Segmenting GPS points into trajectories based on IMN trips...")
    users_with_imn = set(imns.keys())
    users_in_gps = set(df['user_id'].unique())
    common_users = users_with_imn & users_in_gps
    
    print(f"   Users with IMN: {len(users_with_imn)}")
    print(f"   Users in GPS data: {len(users_in_gps)}")
    print(f"   Users in both: {len(common_users)}")
    
    processed_users = 0
    total_trajectories = 0
    total_points = 0
    
    for user_id in sorted(common_users):
        imn = imns[user_id]
        trips = imn.get('trips', [])
        
        if not trips:
            continue
        
        # Get GPS points for this user
        user_gps = df[df['user_id'] == user_id].copy()
        
        if len(user_gps) == 0:
            continue
        
        # Assign trajectory_id based on trips
        trajectory_records = []
        trajectory_id = 0
        
        for from_loc, to_loc, start_time, end_time in trips:
            # Find GPS points within this trip's time window
            trip_points = user_gps[
                (user_gps['time'] >= start_time) & 
                (user_gps['time'] <= end_time)
            ].copy()
            
            if len(trip_points) > 0:
                trip_points['trajectory_id'] = trajectory_id
                trajectory_records.append(trip_points)
                trajectory_id += 1
        
        if trajectory_records:
            # Combine all trajectories for this user
            user_trajectories = pd.concat(trajectory_records, ignore_index=True)
            
            # Select and reorder columns
            user_trajectories = user_trajectories[['trajectory_id', 'lat', 'lon', 'time']]
            
            # Save to CSV
            output_file = os.path.join(OUTPUT_DIR, f"user_{user_id}_trajectory.csv")
            user_trajectories.to_csv(output_file, index=False)
            
            processed_users += 1
            total_trajectories += trajectory_id
            total_points += len(user_trajectories)
            
            if processed_users <= 5 or processed_users % 10 == 0:
                print(f"   [{processed_users}] User {user_id}: {trajectory_id} trajectories, {len(user_trajectories)} points")
    
    print(f"\n" + "="*60)
    print("Conversion Complete!")
    print("="*60)
    print(f"\nSummary:")
    print(f"  ✓ Processed {processed_users} users")
    print(f"  ✓ Created {total_trajectories} trajectories")
    print(f"  ✓ Converted {total_points} GPS points")
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Files: user_*_trajectory.csv")
    print(f"\nYou can now run:")
    print(f"  python evaluate_quality.py --original {OUTPUT_DIR}")
    print()


if __name__ == "__main__":
    main()


