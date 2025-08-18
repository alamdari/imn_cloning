#!/usr/bin/env python3
"""
Simple visualization of original user timelines.
Shows multiple users' timelines in a compact format.
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

# Timezone configuration
tz = pytz.timezone("Europe/Rome")

# Color mapping for trip types
color_map = {
    "home→work": "orange",
    "work→home": "purple", 
    "home→home": "skyblue",
    "work→work": "red",
    "home→admin": "green",
    "home→school": "brown",
    "home→leisure": "pink",
    "work→leisure": "pink",
    "leisure→home": "pink",
    "admin→home": "green",
    "school→home": "brown",
}

def read_poi_data(filepath):
    """Read POI data from gzipped JSON file."""
    poi_data = {}
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            poi_data[row['uid']] = row
    return poi_data

def enrich_imn_with_poi(imn, poi_info):
    """Enrich IMN with POI activity labels."""
    poi_to_activity = {
        "education": "school",
        "food_and_drink": "eat", 
        "shopping": "shop",
        "entertainment_and_recreation": "leisure",
        "transportation": "transit",
        "healthcare": "health",
        "public_services": "admin",
        "finance": "finance",
        "utilities": "utility",
        "other": "unknown"
    }
    
    enriched_locations = {}
    poi_classes = poi_info["poi_classes"]
    
    for loc_id, loc in imn["locations"].items():
        poi_vector = poi_info["poi_freq"].get(loc_id, [0.0]*len(poi_classes))
        max_idx = int(np.argmax(poi_vector))
        top_class = poi_classes[max_idx]
        activity = poi_to_activity.get(top_class, "unknown")
        
        if loc_id == imn.get("home"):
            activity = "home"
        elif loc_id == imn.get("work"):
            activity = "work"
            
        enriched_locations[loc_id] = {
            **loc,
            "poi_vector": poi_vector,
            "activity_label": activity
        }
    
    imn["locations"] = enriched_locations
    return imn

def plot_user_timeline(trips, enriched_locs, y_offset, ax, user_id, day_date):
    """Plot timeline for a single user on a specific day."""
    for from_id, to_id, st, et in trips:
        st_local = datetime.fromtimestamp(st, pytz.utc).astimezone(tz)
        et_local = datetime.fromtimestamp(et, pytz.utc).astimezone(tz)
        
        # Only plot trips for the specified day
        if st_local.date() != day_date:
            continue
            
        from_label = enriched_locs[from_id].get('activity_label', 'unknown')
        to_label = enriched_locs[to_id].get('activity_label', 'unknown')
        trip_type = f"{from_label}→{to_label}"
        color = color_map.get(trip_type, 'gray')
        
        ax.hlines(y=y_offset, xmin=st_local, xmax=et_local, 
                 color=color, linewidth=8, alpha=0.8)

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

def visualize_original_timelines(n_users=None, users_per_page=20, figsize=(16, 12)):
    """
    Visualize original timelines for multiple users in a compact format.
    
    Args:
        n_users: Number of users to visualize (None for all original users)
        users_per_page: Number of users per page
        figsize: Figure size (width, height)
    """
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
    
    if n_users is None:
        n_users = len(available_users)
    
    selected_users = available_users[:n_users]
    
    print(f"Selected {len(selected_users)} users for visualization (from {len(available_users)} available)")
    
    if not selected_users:
        print("No users found that exist in both files!")
        return
    
    # Group users by pages
    pages = [selected_users[i:i + users_per_page] for i in range(0, len(selected_users), users_per_page)]
    
    for page_idx, page_users in enumerate(pages):
        print(f"Creating visualization for page {page_idx + 1}/{len(pages)}")
        
        # Create figure
        fig, axes = plt.subplots(len(page_users), 1, figsize=figsize, sharex=True)
        if len(page_users) == 1:
            axes = [axes]
        
        # Collect all days for this page to set consistent x-axis
        all_days = set()
        
        # First pass: collect all days
        for uid in page_users:
            imn = imns[uid]
            enriched = enrich_imn_with_poi(imn, poi_data[int(uid)])
            
            for from_id, to_id, st, et in enriched['trips']:
                st_local = datetime.fromtimestamp(st, pytz.utc).astimezone(tz)
                all_days.add(st_local.date())
        
        # Sort days
        sorted_days = sorted(all_days)
        if not sorted_days:
            print(f"No valid days found for page {page_idx + 1}")
            continue
            
        # Use the first day as reference for this page
        reference_day = sorted_days[0]
        
        # Plot each user
        for user_idx, uid in enumerate(page_users):
            ax = axes[user_idx]
            
            imn = imns[uid]
            enriched = enrich_imn_with_poi(imn, poi_data[int(uid)])
            
            # Plot timeline for this user
            plot_user_timeline(enriched['trips'], enriched['locations'], 
                             y_offset=0.5, ax=ax, user_id=uid, day_date=reference_day)
            
            # Customize subplot
            ax.set_yticks([0.5])
            ax.set_yticklabels([f"User {uid}"], fontsize=10)
            ax.set_ylim(0, 1)
            ax.grid(True, axis='x', linestyle=':', alpha=0.3)
            
            # Remove y-axis spines
            ax.spines['left'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            
        # Set x-axis formatting for the last subplot
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axes[-1].set_xlabel("Time", fontsize=12)
        
        # Add legend
        legend_handles = [mpatches.Patch(color=c, label=tt) for tt, c in color_map.items()]
        fig.legend(handles=legend_handles, title="Trip Types", 
                  loc='upper right', fontsize=10, title_fontsize=11)
        
        # Add title
        fig.suptitle(f"Original User Timelines - Page {page_idx + 1} ({reference_day})", 
                    fontsize=14, y=0.95)
        
        plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        
        # Save figure to results folder
        output_filename = f"results/original_milano_original_timelines/original_timelines_page_{page_idx + 1}.png"
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        print(f"Saved {output_filename}")
        
        plt.show()

if __name__ == "__main__":
    # Visualize all original users with 20 users per page
    visualize_original_timelines(n_users=None, users_per_page=20) 