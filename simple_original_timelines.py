#!/usr/bin/env python3
"""
Simple visualization of original user timelines.
Copies relevant parts from existing code.
"""

import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from collections import defaultdict
import imn_loading

# Timezone config
tz = pytz.timezone("Europe/Rome")

# Activity color mapping
ACTIVITY_COLORS = {
    "home": "orange",
    "work": "purple",
    "shop": "skyblue",
    "leisure": "red",
    "admin": "green",
    "school": "brown",
    "eat": "pink",
    "transit": "gray",
    "health": "cyan",
    "finance": "magenta",
    "utility": "yellow",
    "unknown": "black"
}

def read_poi_data(filepath):
    poi_data = {}
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            poi_data[row['uid']] = row
    return poi_data

poi_to_activity = {
    "education": "school", "food_and_drink": "eat", "shopping": "shop",
    "entertainment_and_recreation": "leisure", "transportation": "transit",
    "healthcare": "health", "public_services": "admin", "finance": "finance",
    "utilities": "utility", "other": "unknown"
}

def enrich_imn_with_poi(imn, poi_info):
    poi_classes = poi_info["poi_classes"]
    enriched = {}
    for loc_id, loc in imn["locations"].items():
        vec = poi_info["poi_freq"].get(loc_id, [0.0]*len(poi_classes))
        top_idx = int(np.argmax(vec))
        label = poi_to_activity.get(poi_classes[top_idx], "unknown")
        if loc_id == imn.get("home"): label = "home"
        if loc_id == imn.get("work"): label = "work"
        enriched[loc_id] = {**loc, "activity_label": label}
    imn["locations"] = enriched
    return imn

class Stay:
    def __init__(self, location_id, activity_label, start_time, end_time):
        self.location_id = location_id
        self.activity_label = activity_label
        self.start_time = start_time
        self.end_time = end_time
        self.duration = self.end_time - self.start_time if self.end_time is not None and self.start_time is not None else None

def extract_stays_from_trips(trips, locations):
    """Extract stays from trips."""
    stays = []
    for from_id, to_id, st, et in trips:
        from_label = locations[from_id].get('activity_label', 'unknown')
        to_label = locations[to_id].get('activity_label', 'unknown')
        
        # Create stay for origin location
        stay = Stay(from_id, from_label, st, et)
        stays.append(stay)
    
    return stays

def extract_stays_by_day(stays):
    """Group stays by day."""
    stays_by_day = defaultdict(list)
    for stay in stays:
        if stay.start_time is not None:
            dt_local = datetime.fromtimestamp(stay.start_time, pytz.utc).astimezone(tz)
            day_key = dt_local.date()
            stays_by_day[day_key].append(stay)
    
    # Sort stays within each day by start time
    for day in stays_by_day:
        stays_by_day[day].sort(key=lambda s: s.start_time)
    
    return stays_by_day

def plot_stays(stays, y_offset=0, highlight_stay=None, ax=None):
    """Plot stays as horizontal bars."""
    for stay in stays:
        st_local = datetime.fromtimestamp(stay.start_time, pytz.utc).astimezone(tz)
        et_local = datetime.fromtimestamp(stay.end_time, pytz.utc).astimezone(tz)
        color = ACTIVITY_COLORS.get(stay.activity_label, 'gray')
        
        if ax is None:
            plt.hlines(y=y_offset, xmin=st_local, xmax=et_local, color=color, linewidth=10)
        else:
            ax.hlines(y=y_offset, xmin=st_local, xmax=et_local, color=color, linewidth=10)

def get_original_user_ids():
    """Get user IDs from the original Milano IMNs file."""
    original_user_ids = set()
    try:
        with gzip.open('data/milano_2007_imns.json.gz', 'rt', encoding='utf-8') as f:
            for line in f:
                imn = json.loads(line.strip())
                if 'uid' in imn:
                    original_user_ids.add(imn['uid'])
        print(f"Found {len(original_user_ids)} users in original Milano IMNs file")
        return original_user_ids
    except Exception as e:
        print(f"Error reading original file: {e}")
        return set()

def main():
    print("Loading data...")
    
    # Get original user IDs
    original_user_ids = get_original_user_ids()
    
    # Load IMNs and POI data from the test file
    imns = imn_loading.read_imn('data/test_milano_imns.json.gz')
    poi_data = read_poi_data('data/test_milano_imns_pois.json.gz')
    print(f"Loaded {len(imns)} IMNs and POI data for {len(poi_data)} users")
    
    # Filter to only include users that exist in both the test file and the original file
    available_users = [uid for uid in list(imns.keys()) 
                      if int(uid) in poi_data and int(uid) in original_user_ids]
    
    # Select first 20 users from the filtered list
    selected_users = available_users[:20]
    
    print(f"Selected {len(selected_users)} users for visualization (from {len(available_users)} available)")
    
    if not selected_users:
        print("No users found that exist in both files!")
        return
    
    # Create figure
    fig, axes = plt.subplots(len(selected_users), 1, figsize=(16, 2 * len(selected_users)), sharex=True)
    if len(selected_users) == 1:
        axes = [axes]
    
    # Process each user
    for user_idx, uid in enumerate(selected_users):
        print(f"Processing user {user_idx + 1}/{len(selected_users)}: {uid}")
        
        imn = imns[uid]
        enriched = enrich_imn_with_poi(imn, poi_data[int(uid)])
        
        # Extract stays from trips
        stays = extract_stays_from_trips(enriched['trips'], enriched['locations'])
        
        # Group stays by day
        stays_by_day = extract_stays_by_day(stays)
        
        ax = axes[user_idx]
        
        # Plot each day for this user
        y_offset = 0.5
        for day, day_stays in stays_by_day.items():
            plot_stays(day_stays, y_offset=y_offset, ax=ax)
            y_offset += 0.3  # Space between days
        
        # Customize subplot
        ax.set_yticks([0.5 + i * 0.3 for i in range(len(stays_by_day))])
        ax.set_yticklabels([f"User {uid} - {day}" for day in sorted(stays_by_day.keys())], fontsize=8)
        ax.set_ylim(0, 0.5 + len(stays_by_day) * 0.3)
        ax.grid(True, axis='x', linestyle=':', alpha=0.3)
        
        # Remove y-axis spines
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    
    # Set x-axis formatting for the last subplot
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].set_xlabel("Time", fontsize=12)
    
    # Add legend
    legend_handles = [mpatches.Patch(color=c, label=l) for l, c in ACTIVITY_COLORS.items()]
    fig.legend(handles=legend_handles, title="Activity Types", 
              loc='upper right', fontsize=10, title_fontsize=11)
    
    # Add title
    fig.suptitle("Original User Timelines - First 20 Users from Original Dataset", fontsize=14, y=0.95)
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    
    # Save figure to results folder
    output_filename = "results/original_milano_original_timelines2/simple_original_timelines_20_users.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved {output_filename}")
    
    plt.show()

if __name__ == "__main__":
    main() 