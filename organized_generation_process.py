#!/usr/bin/env python3
"""
Organized Generation Process for Individual Mobility Networks
Processes all users and generates synthetic timelines with organized output structure.
"""

import gzip
import json
import os
import random
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pytz
import seaborn as sns
import pandas as pd

import imn_loading

# Configuration
RANDOMNESS_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
TIMEZONE = pytz.timezone("Europe/Rome")

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

    def to_dict(self) -> Dict[str, Any]:
        """Convert stay to dictionary representation."""
        return {
            'location_id': self.location_id,
            'activity_label': self.activity_label,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration
        }


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


def extract_stays_by_day(stays: List[Stay], tz) -> Dict[datetime.date, List[Stay]]:
    """Group stays by day, handling cross-day stays."""
    stays_by_day = defaultdict(list)
    
    for stay in stays:
        if stay.start_time is None or stay.end_time is None:
            continue
            
        start_dt = datetime.fromtimestamp(stay.start_time, tz)
        end_dt = datetime.fromtimestamp(stay.end_time, tz)
        
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


def build_stay_distributions(stays_by_day: Dict[datetime.date, List[Stay]]) -> Tuple[Dict, Dict, Any]:
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


def user_probs_report(duration_probs: Dict, transition_probs: Dict, trip_duration_probs: Any, 
                     user_id: int, out_folder: str) -> None:
    """Generate and save user probability report with visualizations."""
    os.makedirs(out_folder, exist_ok=True)

    # Expand durations into samples for plotting
    expanded = {}
    for act, (hist, bins) in duration_probs.items():
        mids = (bins[:-1] + bins[1:]) / 2
        samples = np.repeat(mids, hist)
        expanded[act] = samples

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Durations boxplot
    if expanded:
        sns.boxplot(data=pd.DataFrame(dict([(k, pd.Series(v)) for k, v in expanded.items()])),
                    ax=axes[0])
        axes[0].set_title("Stay Durations per Activity")
        axes[0].set_ylabel("Duration (s)")
        axes[0].tick_params(axis='x', rotation=45)

    # Transition heatmap
    df = pd.DataFrame(transition_probs).fillna(0)
    sns.heatmap(df, annot=True, cmap="Blues", cbar_kws={'label': 'Probability'}, ax=axes[1])
    axes[1].set_title("Transition Probability Matrix")
    axes[1].set_xlabel("From Activity")
    axes[1].set_ylabel("To Activity")

    # Trip duration density
    if trip_duration_probs is not None:
        hist, bins = trip_duration_probs
        mids = (bins[:-1] + bins[1:]) / 2
        samples = np.repeat(mids, hist)
        if len(samples) > 1:
            sns.kdeplot(samples, fill=True, ax=axes[2])
        else:
            axes[2].hist(samples, bins=10, edgecolor="k")
        axes[2].set_title("Trip Duration Distribution")
        axes[2].set_xlabel("Duration (s)")
        axes[2].set_ylabel("Density")

    plt.tight_layout()
    fig_path = os.path.join(out_folder, f"user_probs_report_{user_id}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Save JSON report
    json_report = {
        "duration_probs": {k: {"hist": hist.tolist(), "bins": bins.tolist()} 
                          for k, (hist, bins) in duration_probs.items()},
        "transition_probs": {k: {kk: float(vv) for kk, vv in d.items()} 
                           for k, d in transition_probs.items()},
        "trip_duration_probs": {
            "hist": trip_duration_probs[0].tolist(),
            "bins": trip_duration_probs[1].tolist()
        } if trip_duration_probs is not None else None
    }

    json_path = os.path.join(out_folder, f"user_probs_report_{user_id}.json")
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)

    print(f"  ✓ Probability report saved: {os.path.basename(fig_path)}, {os.path.basename(json_path)}")


def sample_from_hist(hist: np.ndarray, bins: np.ndarray) -> float:
    """Sample a value from histogram (hist, bins)."""
    if hist.sum() == 0:
        return np.mean(bins)
    probs = hist / hist.sum()
    bin_idx = np.random.choice(len(hist), p=probs)
    return np.random.uniform(bins[bin_idx], bins[bin_idx + 1])


def find_anchor_stay_for_day(stays: List[Stay]) -> Stay:
    """Find the longest non-home stay for a given day."""
    non_home_stays = [s for s in stays if s.activity_label != 'home']
    if not non_home_stays:
        return None
    return max(non_home_stays, key=lambda s: s.duration)


def generate_synthetic_day(original_stays: List[Stay], duration_probs: Dict, 
                          transition_probs: Dict, randomness: float = 0.5, 
                          day_length: int = 24 * 3600, tz=None) -> List[Tuple[str, int, int]]:
    """Generate synthetic day timeline based on original stays and learned distributions."""
    if not original_stays:
        return []
    
    # Convert to relative seconds
    first_dt = datetime.fromtimestamp(original_stays[0].start_time, tz)
    day_start_dt = first_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    day_start = int(day_start_dt.timestamp())

    rel_stays = []
    for s in original_stays:
        rel_start = int(s.start_time - day_start)
        rel_end = int(s.end_time - day_start)
        rel_stays.append((s, rel_start, rel_end))

    # Find anchor (in relative time)
    anchor_stay = find_anchor_stay_for_day(original_stays)
    if anchor_stay is None:
        return []  # fallback if no non-home stay

    anchor_rel_start = int(anchor_stay.start_time - day_start)
    anchor_rel_end = int(anchor_stay.end_time - day_start)

    synthetic_stays = []
    current_time = 0
    prev_activity = "home"  # always force first stay to begin at home

    # Generate before anchor
    for (s, rel_start, rel_end) in rel_stays:
        if s is anchor_stay:
            break

        # Choose activity
        if current_time == 0:
            act = "home"  # enforce home at midnight
        elif random.random() < (1 - randomness):
            act = s.activity_label
        else:
            if prev_activity in transition_probs:
                to_probs = transition_probs[prev_activity]
                act = random.choices(list(to_probs.keys()), weights=to_probs.values())[0]
            else:
                act = s.activity_label

        # Choose duration
        orig_dur = rel_end - rel_start
        if act in duration_probs:
            hist, bins = duration_probs[act]
            sampled_dur = sample_from_hist(hist, bins)
        else:
            # Fallback for activities not in distributions (like "trip")
            sampled_dur = orig_dur
        dur = int((1 - randomness) * orig_dur + randomness * sampled_dur)

        end_time = min(current_time + dur, anchor_rel_start)
        if end_time > current_time:  # avoid zero/negative durations
            synthetic_stays.append((act, current_time, end_time))
            prev_activity = act
        current_time = end_time
        if current_time >= anchor_rel_start:
            break

    # Insert anchor unchanged
    synthetic_stays.append((anchor_stay.activity_label, anchor_rel_start, anchor_rel_end))
    current_time = anchor_rel_end
    prev_activity = anchor_stay.activity_label

    # Generate after anchor
    passed_anchor = False
    for (s, rel_start, rel_end) in rel_stays:
        if s is anchor_stay:
            passed_anchor = True
            continue
        if not passed_anchor:
            continue

        # Choose activity
        if random.random() < (1 - randomness):
            act = s.activity_label
        else:
            if prev_activity in transition_probs:
                to_probs = transition_probs[prev_activity]
                act = random.choices(list(to_probs.keys()), weights=to_probs.values())[0]
            else:
                act = s.activity_label

        # Choose duration
        orig_dur = rel_end - rel_start
        if act in duration_probs:
            hist, bins = duration_probs[act]
            sampled_dur = sample_from_hist(hist, bins)
        else:
            # Fallback for activities not in distributions (like "trip")
            sampled_dur = orig_dur
        dur = int((1 - randomness) * orig_dur + randomness * sampled_dur)

        end_time = min(current_time + dur, day_length)
        if end_time > current_time:
            synthetic_stays.append((act, current_time, end_time))
            prev_activity = act
        current_time = end_time
        if current_time >= day_length:
            break

    # Fix last stay to midnight
    if synthetic_stays:
        act, start, _ = synthetic_stays[-1]
        synthetic_stays[-1] = (act, start, day_length)

    return synthetic_stays


def prepare_day_data(stays_by_day: Dict[datetime.date, List[Stay]], 
                    user_duration_probs: Dict, user_transition_probs: Dict, 
                    randomness_levels: List[float], tz) -> Dict:
    """Prepare day data structure for visualization."""
    day_data = {}
    sorted_days = sorted(stays_by_day.keys())

    for day_date in sorted_days:
        day = stays_by_day[day_date]

        # Midnight timestamp
        first_dt = datetime.fromtimestamp(day[0].start_time, tz)
        midnight_dt = first_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        day_start = int(midnight_dt.timestamp())

        # Original stays → relative seconds
        original_stays = [
            (s.activity_label, s.start_time - day_start, s.end_time - day_start)
            for s in day
        ]

        # Synthetic stays for each randomness
        synthetic_dict = {}
        for r in randomness_levels:
            synthetic = generate_synthetic_day(
                day, user_duration_probs, user_transition_probs, randomness=r, tz=tz
            )
            synthetic_dict[r] = synthetic

        # Anchor stay in relative seconds
        anchor = find_anchor_stay_for_day(day)
        anchor_tuple = None
        if anchor:
            anchor_tuple = (
                anchor.activity_label,
                anchor.start_time - day_start,
                anchor.end_time - day_start,
            )

        day_data[day_date] = {
            "original": original_stays,
            "synthetic": synthetic_dict,
            "anchor": anchor_tuple
        }

    return day_data


def plot_stays(stays: List[Tuple[str, int, int]], y_offset: float, ax, anchor: Tuple = None):
    """Plot stays as bars at a given y_offset. Highlight anchor with bold stroke."""
    for act, st, et in stays:
        is_anchor = anchor and (act, st, et) == anchor
        ax.barh(
            y_offset,
            et - st,
            left=st,
            height=0.25,  # same height for all
            color=ACTIVITY_COLORS.get(act, "black"),
            edgecolor="black" if is_anchor else None,
            linewidth=2 if is_anchor else 0.5,
            alpha=0.9
        )


def visualize_day_data(day_data: Dict, user_id: int = 0) -> plt.Figure:
    """Plot all days in one figure with original and synthetic timelines."""
    fig, axes = plt.subplots(len(day_data), 1, figsize=(16, 3 * len(day_data)), sharex=True)
    if len(day_data) == 1:
        axes = [axes]

    for ax, (day_date, content) in zip(axes, sorted(day_data.items())):
        original = content["original"]
        synthetics = content["synthetic"]
        anchor = content.get("anchor")

        # Plot original
        y_offset = 0.5
        plot_stays(original, y_offset, ax, anchor=anchor)
        labels = ["Original"]

        # Plot synthetics
        for r, stays in sorted(synthetics.items()):
            y_offset += 0.4
            plot_stays(stays, y_offset, ax)
            labels.append(f"Rand={r}")

        # Y axis
        ax.set_yticks([0.5 + i * 0.4 for i in range(len(labels))])
        ax.set_yticklabels(labels, fontsize=8)

        # X axis: 24h time-of-day
        ax.set_xlim(0, 24 * 3600)
        ax.set_xticks([i * 3600 for i in range(0, 25, 2)])
        ax.set_xticklabels([f"{i:02d}:00" for i in range(0, 25, 2)], rotation=0)
        ax.grid(True, axis="x", linestyle=":", alpha=0.3)

        ax.set_title(f"User {user_id} - {day_date}", fontsize=12)

    # Legend
    legend_handles = [mpatches.Patch(color=c, label=l) for l, c in ACTIVITY_COLORS.items()]
    fig.legend(handles=legend_handles, title="Activity Types",
               loc="upper right", fontsize=9, title_fontsize=10)

    fig.suptitle(f"User {user_id} - Original and Synthetic Timelines", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.85, 0.98])
    return fig


def process_single_user(user_id: int, imn: Dict, poi_info: Dict, 
                       randomness_levels: List[float], results_dir: str, tz) -> None:
    """Process a single user and generate all outputs."""
    print(f"Processing user {user_id}...")
    
    # Enrich IMN with POI data
    enriched = enrich_imn_with_poi(imn, poi_info)
    
    # Extract stays from trips
    stays = extract_stays_from_trips(enriched['trips'], enriched['locations'])
    
    # Group stays by day
    stays_by_day = extract_stays_by_day(stays, tz)
    
    if not stays_by_day:
        print(f"  ⚠ No valid stays found for user {user_id}")
        return
    
    # Build user-specific distributions
    user_duration_probs, user_transition_probs, user_trip_duration_probs = build_stay_distributions(stays_by_day)
    
    # Generate probability report
    user_probs_report(user_duration_probs, user_transition_probs, user_trip_duration_probs, 
                     user_id, os.path.join(results_dir, "user_probability_reports"))
    
    # Prepare day data for visualization
    day_data = prepare_day_data(stays_by_day, user_duration_probs, user_transition_probs, 
                               randomness_levels, tz)
    
    # Generate timeline visualization
    fig = visualize_day_data(day_data, user_id=user_id)
    timeline_path = os.path.join(results_dir, "user_timeline_visualizations", f"user_{user_id}_timelines.png")
    os.makedirs(os.path.dirname(timeline_path), exist_ok=True)
    fig.savefig(timeline_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Timeline visualization saved: {os.path.basename(timeline_path)}")


def main():
    """Main function to process all users."""
    print("Starting Individual Mobility Network Generation Process")
    print("=" * 60)
    
    # Create results directory structure
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "user_probability_reports"), exist_ok=True)
    os.makedirs(os.path.join(results_dir, "user_timeline_visualizations"), exist_ok=True)
    
    print(f"Results will be saved to: {results_dir}")
    print(f"  - User probability reports: {results_dir}/user_probability_reports/")
    print(f"  - Timeline visualizations: {results_dir}/user_timeline_visualizations/")
    print()
    
    # Load IMNs and POI data
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
    
    # Process each user
    print(f"\nProcessing {len(imns)} users...")
    print("-" * 40)
    
    for i, (user_id, imn) in enumerate(imns.items(), 1):
        print(f"[{i}/{len(imns)}] ", end="")
        
        if user_id not in poi_data:
            print(f"⚠ No POI data found for user {user_id}, skipping...")
            continue
            
        try:
            process_single_user(user_id, imn, poi_data[user_id], RANDOMNESS_LEVELS, results_dir, TIMEZONE)
        except Exception as e:
            print(f"❌ Error processing user {user_id}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()
