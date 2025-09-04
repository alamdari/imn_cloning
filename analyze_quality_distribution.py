#!/usr/bin/env python3
"""
Analyze quality distribution of user data for blending statistics.
"""

import gzip
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from typing import Dict, List, Tuple, Any

import imn_loading

# Configuration
TIMEZONE = "Europe/Rome"
BLEND_MIN_SAMPLES = 5          # Very low quality threshold
BLEND_LOW_QUALITY_SAMPLES = 20 # Low quality threshold

# Activity color mapping
ACTIVITY_COLORS = {
    "home": "skyblue",
    "work": "orange", 
    "eat": "green",
    "utility": "purple",
    "transit": "red",
    "unknown": "gray",
    "school": "yellow",
    "shop": "pink",
    "leisure": "lightgreen",
    "health": "lightcoral",
    "admin": "lightblue",
    "finance": "gold",
    "trip": "lightgray"
}

# POI to activity mapping
POI_TO_ACTIVITY = {
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


class Stay:
    """Represents a stay at a location with activity and timing information."""
    
    def __init__(self, location_id: int, activity_label: str, start_time: int, end_time: int):
        self.location_id = location_id
        self.activity_label = activity_label
        self.start_time = start_time
        self.end_time = end_time
        self.duration = self.end_time - self.start_time if self.end_time is not None and self.start_time is not None else None

    def set_end_time(self, end_time: int):
        """Set the end time and recalculate duration."""
        self.end_time = end_time
        if self.start_time is not None:
            self.duration = self.end_time - self.start_time


def read_poi_data(filepath: str) -> Dict[int, Dict]:
    """Read POI data from gzipped JSON file."""
    poi_data = {}
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            poi_data[row['uid']] = row
    return poi_data


def enrich_imn_with_poi(imn: Dict, poi_info: Dict) -> Dict:
    """Enrich IMN with POI activity labels."""
    poi_classes = poi_info["poi_classes"]
    enriched = {}
    
    for loc_id, loc in imn["locations"].items():
        vec = poi_info["poi_freq"].get(loc_id, [0.0] * len(poi_classes))
        top_idx = int(np.argmax(vec))
        label = POI_TO_ACTIVITY.get(poi_classes[top_idx], "unknown")
        
        # Override with home/work if applicable
        if loc_id == imn.get("home"): 
            label = "home"
        if loc_id == imn.get("work"): 
            label = "work"
            
        enriched[loc_id] = {**loc, "activity_label": label}
    
    imn["locations"] = enriched
    return imn


def extract_stays_from_trips(trips: List[Tuple], locations: Dict) -> List[Stay]:
    """Convert trips into stays by considering the destination of each trip as a stay.
    Also adds trip activities in gaps between stays."""
    stays = []
    
    # First pass: create stays with start times
    for from_id, to_id, st, et in trips:
        activity_label = locations[to_id].get('activity_label', 'unknown')
        stays.append(Stay(to_id, activity_label, et, None))
    
    # Second pass: set end times and add trip activities in gaps
    all_activities = []
    for i in range(len(stays)):
        current_stay = stays[i]
        
        # Set end time based on next trip's start time
        if i < len(stays) - 1:
            next_trip_start = trips[i + 1][2]  # start time of the next trip
            current_stay.set_end_time(next_trip_start)
        else:
            # Handle the last stay
            if current_stay.start_time is not None:
                current_stay.set_end_time(current_stay.start_time + 3600)  # Default 1 hour duration
        
        # Add the stay activity
        all_activities.append(current_stay)
        
        # Add trip activity if there's a gap before the next stay
        if i < len(stays) - 1 and current_stay.end_time is not None:
            next_stay = stays[i + 1]
            if next_stay.start_time is not None and next_stay.start_time > current_stay.end_time:
                # Create trip activity to fill the gap
                trip_duration = next_stay.start_time - current_stay.end_time
                if trip_duration > 0:  # Only add if there's actually a gap
                    trip_stay = Stay(-1, "trip", current_stay.end_time, next_stay.start_time)
                    all_activities.append(trip_stay)
    
    return all_activities


def extract_stays_by_day(stays: List[Stay], tz) -> Dict:
    """Group stays by day, handling cross-day stays."""
    from datetime import datetime, timedelta
    import pytz
    
    timezone = pytz.timezone(tz)
    stays_by_day = defaultdict(list)
    
    for stay in stays:
        if stay.start_time is None or stay.end_time is None:
            continue
            
        start_dt = datetime.fromtimestamp(stay.start_time, timezone)
        end_dt = datetime.fromtimestamp(stay.end_time, timezone)
        
        # If stay spans multiple days, split it
        current_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = current_dt + timedelta(days=1)
        
        while current_dt < end_dt:
            day_start = max(start_dt, current_dt)
            day_end = min(end_dt, end_of_day)
            
            # Create stay for this day
            day_stay = Stay(
                stay.location_id,
                stay.activity_label,
                int(day_start.timestamp()),
                int(day_end.timestamp())
            )
            
            stays_by_day[current_dt.date()].append(day_stay)
            
            # Move to next day
            current_dt = end_of_day
            end_of_day = current_dt + timedelta(days=1)
    
    return stays_by_day


def build_stay_distributions(stays_by_day: Dict) -> Tuple[Dict, Dict, Any]:
    """Build distributions for stay durations and activity types across all days."""
    duration_dist = defaultdict(list)
    activity_transitions = defaultdict(list)
    trip_durations = []
    
    # Collect data from all days
    for day, day_stays in stays_by_day.items():
        for i in range(len(day_stays) - 1):
            current_stay = day_stays[i]
            next_stay = day_stays[i + 1]
            
            # Record duration for this activity type
            if current_stay.duration is not None:
                duration_dist[current_stay.activity_label].append(current_stay.duration)
            
            # Record activity transition
            activity_transitions[current_stay.activity_label].append(next_stay.activity_label)
            
            # Record trip duration (gap between stays)
            if current_stay.end_time is not None and next_stay.start_time is not None:
                trip_duration = next_stay.start_time - current_stay.end_time
                if trip_duration > 0:
                    trip_durations.append(trip_duration)
    
    # Convert lists to probability distributions
    duration_probs = {}
    for activity, durations in duration_dist.items():
        if len(durations) > 0:
            hist, bins = np.histogram(durations, bins=20, density=False)
            duration_probs[activity] = (hist, bins)
    
    transition_probs = {}
    for from_activity, to_activities in activity_transitions.items():
        if len(to_activities) > 0:
            unique_activities, counts = np.unique(to_activities, return_counts=True)
            probs = counts / counts.sum()
            transition_probs[from_activity] = dict(zip(unique_activities, probs))
    
    # Trip duration distribution
    trip_duration_probs = None
    if len(trip_durations) > 0:
        hist, bins = np.histogram(trip_durations, bins=20, density=False)
        trip_duration_probs = (hist, bins)
    
    return duration_probs, transition_probs, trip_duration_probs


def analyze_quality_distribution():
    """Analyze the quality distribution of user data."""
    print("Analyzing quality distribution of user data...")
    print("=" * 60)
    
    # Load data
    print("Loading data...")
    try:
        # Load full dataset first to get user IDs
        full_imns = imn_loading.read_imn('data/milano_2007_imns.json.gz')
        filtered_user_ids = list(full_imns.keys())
        
        # Load test dataset (subset)
        imns = imn_loading.read_imn('data/test_milano_imns.json.gz')
        # Keep only users that exist in both datasets
        imns = {k: imns[k] for k in filtered_user_ids if k in imns}
        
        poi_data = read_poi_data('data/test_milano_imns_pois.json.gz')
        
        print(f"✓ Loaded {len(imns)} IMNs and POI data for {len(poi_data)} users")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Track quality statistics
    quality_stats = {
        'very_low': 0,      # < 5 samples
        'low': 0,           # 5-19 samples  
        'good': 0,          # >= 20 samples
        'total_activities': 0,
        'total_users': 0
    }
    
    # Track sample counts for detailed analysis
    all_sample_counts = []
    activity_sample_counts = defaultdict(list)
    user_activity_counts = []
    
    print(f"\nAnalyzing {len(imns)} users...")
    print("-" * 40)
    
    for i, (user_id, imn) in enumerate(imns.items(), 1):
        if i % 50 == 0:
            print(f"Processed {i}/{len(imns)} users...")
            
        if user_id not in poi_data:
            continue
            
        try:
            # Enrich IMN with POI data
            enriched = enrich_imn_with_poi(imn, poi_data[user_id])
            
            # Extract stays from trips
            stays = extract_stays_from_trips(enriched['trips'], enriched['locations'])
            
            # Group stays by day
            stays_by_day = extract_stays_by_day(stays, TIMEZONE)
            
            if not stays_by_day:
                continue
            
            # Build user-specific distributions
            user_duration_probs, user_transition_probs, user_trip_duration_probs = build_stay_distributions(stays_by_day)
            
            # Analyze each activity's sample count
            user_activity_count = 0
            for activity, (hist, bins) in user_duration_probs.items():
                sample_count = int(hist.sum())
                all_sample_counts.append(sample_count)
                activity_sample_counts[activity].append(sample_count)
                user_activity_count += sample_count
                
                # Categorize quality
                if sample_count < BLEND_MIN_SAMPLES:
                    quality_stats['very_low'] += 1
                elif sample_count < BLEND_LOW_QUALITY_SAMPLES:
                    quality_stats['low'] += 1
                else:
                    quality_stats['good'] += 1
                
                quality_stats['total_activities'] += 1
            
            user_activity_counts.append(user_activity_count)
            quality_stats['total_users'] += 1
            
        except Exception as e:
            print(f"❌ Error processing user {user_id}: {e}")
            continue
    
    # Print results
    print(f"\nQuality Distribution Analysis Results:")
    print("=" * 60)
    print(f"Total users analyzed: {quality_stats['total_users']}")
    print(f"Total activities analyzed: {quality_stats['total_activities']}")
    print()
    print(f"Quality Categories:")
    print(f"  Very Low Quality (< {BLEND_MIN_SAMPLES} samples): {quality_stats['very_low']:,} activities ({quality_stats['very_low']/quality_stats['total_activities']*100:.1f}%)")
    print(f"  Low Quality ({BLEND_MIN_SAMPLES}-{BLEND_LOW_QUALITY_SAMPLES-1} samples): {quality_stats['low']:,} activities ({quality_stats['low']/quality_stats['total_activities']*100:.1f}%)")
    print(f"  Good Quality (≥ {BLEND_LOW_QUALITY_SAMPLES} samples): {quality_stats['good']:,} activities ({quality_stats['good']/quality_stats['total_activities']*100:.1f}%)")
    
    # Activity-specific statistics
    print(f"\nActivity-specific sample count statistics:")
    print("-" * 40)
    for activity, counts in activity_sample_counts.items():
        if counts:
            avg_samples = np.mean(counts)
            median_samples = np.median(counts)
            max_samples = np.max(counts)
            very_low_count = sum(1 for c in counts if c < BLEND_MIN_SAMPLES)
            low_count = sum(1 for c in counts if BLEND_MIN_SAMPLES <= c < BLEND_LOW_QUALITY_SAMPLES)
            good_count = sum(1 for c in counts if c >= BLEND_LOW_QUALITY_SAMPLES)
            
            print(f"  {activity:12}: avg={avg_samples:5.1f}, median={median_samples:5.1f}, max={max_samples:4.0f} | "
                  f"very_low={very_low_count:3d}, low={low_count:3d}, good={good_count:3d}")
    
    # Create visualization
    create_quality_visualization(all_sample_counts, activity_sample_counts, quality_stats)
    
    return quality_stats, all_sample_counts, activity_sample_counts


def create_quality_visualization(all_sample_counts, activity_sample_counts, quality_stats):
    """Create visualization of quality distribution."""
    print(f"\nCreating quality distribution visualization...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Overall sample count distribution
    ax1 = axes[0, 0]
    ax1.hist(all_sample_counts, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(BLEND_MIN_SAMPLES, color='red', linestyle='--', label=f'Very Low Threshold ({BLEND_MIN_SAMPLES})')
    ax1.axvline(BLEND_LOW_QUALITY_SAMPLES, color='orange', linestyle='--', label=f'Low Threshold ({BLEND_LOW_QUALITY_SAMPLES})')
    ax1.set_xlabel('Sample Count')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Sample Counts Across All Activities')
    ax1.legend()
    ax1.set_yscale('log')
    
    # 2. Quality category pie chart
    ax2 = axes[0, 1]
    labels = ['Very Low', 'Low', 'Good']
    sizes = [quality_stats['very_low'], quality_stats['low'], quality_stats['good']]
    colors = ['red', 'orange', 'green']
    ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('Quality Distribution of Activities')
    
    # 3. Activity-specific sample counts (box plot)
    ax3 = axes[1, 0]
    activity_data = []
    activity_labels = []
    for activity, counts in activity_sample_counts.items():
        if len(counts) > 0:
            activity_data.append(counts)
            activity_labels.append(activity)
    
    if activity_data:
        ax3.boxplot(activity_data, labels=activity_labels)
        ax3.axhline(BLEND_MIN_SAMPLES, color='red', linestyle='--', alpha=0.7)
        ax3.axhline(BLEND_LOW_QUALITY_SAMPLES, color='orange', linestyle='--', alpha=0.7)
        ax3.set_ylabel('Sample Count')
        ax3.set_title('Sample Count Distribution by Activity Type')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_yscale('log')
    
    # 4. Cumulative distribution
    ax4 = axes[1, 1]
    sorted_counts = np.sort(all_sample_counts)
    cumulative = np.arange(1, len(sorted_counts) + 1) / len(sorted_counts)
    ax4.plot(sorted_counts, cumulative, linewidth=2)
    ax4.axvline(BLEND_MIN_SAMPLES, color='red', linestyle='--', alpha=0.7, label=f'Very Low ({BLEND_MIN_SAMPLES})')
    ax4.axvline(BLEND_LOW_QUALITY_SAMPLES, color='orange', linestyle='--', alpha=0.7, label=f'Low ({BLEND_LOW_QUALITY_SAMPLES})')
    ax4.set_xlabel('Sample Count')
    ax4.set_ylabel('Cumulative Probability')
    ax4.set_title('Cumulative Distribution of Sample Counts')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = 'quality_distribution_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Quality distribution visualization saved: {output_path}")


if __name__ == "__main__":
    analyze_quality_distribution()
