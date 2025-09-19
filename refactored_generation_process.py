#!/usr/bin/env python3
"""
Organized Generation Process for Individual Mobility Networks
Processes all users and generates synthetic timelines with organized output structure.
"""

import gzip
import json
import os
import random
import argparse
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

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
    "finance": "gold"
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


# ------------------------------
# IO / CONFIG
# ------------------------------

@dataclass
class PathsConfig:
    full_imn_path: str = 'data/milano_2007_imns.json.gz'
    test_imn_path: str = 'data/test_milano_imns.json.gz'
    poi_path: str = 'data/test_milano_imns_pois.json.gz'
    results_dir: str = './results'
    prob_subdir: str = 'user_probability_reports3'
    vis_subdir: str = 'user_timeline_visualizations3'

    def prob_dir(self) -> str:
        return os.path.join(self.results_dir, self.prob_subdir)

    def vis_dir(self) -> str:
        return os.path.join(self.results_dir, self.vis_subdir)


def ensure_output_structure(paths: PathsConfig) -> None:
    os.makedirs(paths.results_dir, exist_ok=True)
    os.makedirs(paths.prob_dir(), exist_ok=True)
    os.makedirs(paths.vis_dir(), exist_ok=True)


def parse_args(argv: Optional[List[str]] = None) -> PathsConfig:
    parser = argparse.ArgumentParser(description='Generate synthetic timelines from IMNs.')
    parser.add_argument('--full-imn', default='data/milano_2007_imns.json.gz', help='Path to full IMNs (for user IDs).')
    parser.add_argument('--test-imn', default='data/test_milano_imns.json.gz', help='Path to subset/test IMNs.')
    parser.add_argument('--poi', default='data/test_milano_imns_pois.json.gz', help='Path to POI enrichment data.')
    parser.add_argument('--results-dir', default='./results', help='Base directory for results.')
    parser.add_argument('--prob-subdir', default='user_probability_reports3', help='Probability reports subdirectory.')
    parser.add_argument('--vis-subdir', default='user_timeline_visualizations3', help='Timeline visualizations subdirectory.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (optional).')
    args = parser.parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    return PathsConfig(
        full_imn_path=args.full_imn,
        test_imn_path=args.test_imn,
        poi_path=args.poi,
        results_dir=args.results_dir,
        prob_subdir=args.prob_subdir,
        vis_subdir=args.vis_subdir,
    )


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
    Gaps are later handled by day-stretching; we do not create a 'trip' activity."""
    stays = []
    
    # First pass: create stays with start times
    for from_id, to_id, st, et in trips:
        activity_label = locations[to_id].get('activity_label', 'unknown')
        stays.append(Stay(to_id, activity_label, et, None))
    
    # Second pass: set end times only; gaps will be handled by stretching
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


def _stretch_day_stays_to_full_coverage(day_stays: List[Stay], tz) -> List[Stay]:
    """Stretch a single day's stays so they exactly cover the whole day without gaps.

    Rules:
    - Snap first stay start to midnight
    - Between consecutive stays, set previous end to next start (remove gaps/overlaps)
    - Snap last stay end to next midnight
    """
    if not day_stays:
        return []

    # Determine day bounds from first stay
    first_dt = datetime.fromtimestamp(day_stays[0].start_time, tz)
    midnight_dt = first_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    day_start = int(midnight_dt.timestamp())
    day_end = int((midnight_dt + timedelta(days=1)).timestamp())

    # Sort by start time
    sorted_stays = sorted(day_stays, key=lambda s: (s.start_time or 0, s.end_time or 0))

    # Clone to avoid mutating original objects unintentionally
    stretched: List[Stay] = []
    for s in sorted_stays:
        stretched.append(Stay(s.location_id, s.activity_label, s.start_time, s.end_time))

    # Snap first start to midnight
    if stretched[0].start_time is None:
        stretched[0].start_time = day_start
    else:
        stretched[0].start_time = min(max(stretched[0].start_time, day_start), day_end)

    # Ensure continuity between stays (fill gaps, trim overlaps)
    for i in range(len(stretched) - 1):
        current = stretched[i]
        nxt = stretched[i + 1]

        # Normalize None times
        if current.end_time is None:
            current.end_time = current.start_time
        if nxt.start_time is None:
            nxt.start_time = current.end_time

        # Force continuity: current ends exactly at next start
        nxt.start_time = max(min(nxt.start_time, day_end), day_start)
        current.end_time = max(min(nxt.start_time, day_end), day_start)

        # Update duration
        current.duration = max(0, current.end_time - current.start_time)

    # Snap last end to day_end
    last = stretched[-1]
    if last.end_time is None:
        last.end_time = day_end
    else:
        last.end_time = min(max(last.end_time, day_start), day_end)
    last.end_time = day_end
    last.duration = max(0, last.end_time - last.start_time)

    # Ensure first start is day_start after adjustments
    stretched[0].start_time = day_start
    stretched[0].duration = max(0, (stretched[0].end_time or day_start) - stretched[0].start_time)

    return stretched


def stretch_all_days(stays_by_day: Dict[datetime.date, List[Stay]], tz) -> Dict[datetime.date, List[Stay]]:
    """Apply stretching to every day's stays to eliminate gaps and cover full day."""
    stretched_by_day: Dict[datetime.date, List[Stay]] = {}
    for day, day_stays in stays_by_day.items():
        stretched_by_day[day] = _stretch_day_stays_to_full_coverage(day_stays, tz)
    return stretched_by_day


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
    anchor_orig_dur = max(1, anchor_rel_end - anchor_rel_start)

    # Perturb anchor start and duration within +/- 0.25 * randomness * original_duration
    max_shift = int(0.25 * randomness * anchor_orig_dur)
    if max_shift > 0:
        delta_start = int(np.random.uniform(-max_shift, max_shift))
        delta_dur = int(np.random.uniform(-max_shift, max_shift))
    else:
        delta_start = 0
        delta_dur = 0

    pert_start = anchor_rel_start + delta_start
    pert_start = max(0, min(pert_start, day_length))
    pert_dur = max(1, anchor_orig_dur + delta_dur)
    # If anchor is the first activity of the day, do not allow any leading gap
    is_anchor_first = False
    if rel_stays and rel_stays[0][0] is anchor_stay:
        is_anchor_first = True
        pert_start = 0

    pert_end = pert_start + pert_dur
    if pert_end > day_length:
        pert_end = day_length
        pert_dur = max(1, pert_end - pert_start)

    synthetic_stays = []
    current_time = 0
    prev_activity = "home"  # always force first stay to begin at home

    # Generate before anchor (respect perturbed anchor start)
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
            # Fallback for activities not in distributions
            sampled_dur = orig_dur
        dur = int((1 - randomness) * orig_dur + randomness * sampled_dur)

        end_time = min(current_time + dur, pert_start)
        if end_time > current_time:  # avoid zero/negative durations
            synthetic_stays.append((act, current_time, end_time))
            prev_activity = act
        current_time = end_time
        if current_time >= pert_start:
            break

    # Insert anchor with perturbation applied
    synthetic_stays.append((anchor_stay.activity_label, pert_start, pert_end))
    current_time = pert_end
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
            # Fallback for activities not in distributions
            sampled_dur = orig_dur
        dur = int((1 - randomness) * orig_dur + randomness * sampled_dur)

        end_time = min(current_time + dur, day_length)
        if end_time > current_time:
            synthetic_stays.append((act, current_time, end_time))
            prev_activity = act
        current_time = end_time
        if current_time >= day_length:
            break

    # Anchor-aware stretching: do not change anchor; extend pre-anchor last stay to
    # anchor start if needed; extend final non-anchor stay to day end.
    def _stretch_anchor_aware(
        stays_rel: List[Tuple[str, int, int]],
        anchor_tuple: Tuple[str, int, int],
        total_len: int
    ) -> List[Tuple[str, int, int]]:
        if not stays_rel:
            return []
        # Sort by start time
        ordered = sorted(stays_rel, key=lambda x: x[1])

        # Identify anchor index if present exactly
        anchor_idx = None
        for i, t in enumerate(ordered):
            if t == anchor_tuple:
                anchor_idx = i
                break

        # Ensure the first stay starts at 0 if it's not the anchor
        out: List[Tuple[str, int, int]] = []
        act0, s0, e0 = ordered[0]
        if anchor_idx == 0:
            # Keep anchor unchanged; do not force start to 0
            out.append((act0, s0, e0))
        else:
            s0 = 0
            e0 = max(0, min(e0, total_len))
            if e0 < s0:
                e0 = s0
            out.append((act0, s0, e0))

        # Iterate and enforce continuity, preserving anchor start
        for i in range(1, len(ordered)):
            act, st, et = ordered[i]
            st = max(0, min(st, total_len))
            et = max(0, min(et, total_len))
            prev_act, prev_st, prev_et = out[-1]

            if anchor_idx is not None and i == anchor_idx:
                # Do not move anchor start; stretch previous end to anchor start if gap
                st_fixed = st
                prev_et = min(max(st_fixed, 0), total_len)
                out[-1] = (prev_act, prev_st, prev_et)
                if et < st_fixed:
                    et = st_fixed
                out.append((act, st_fixed, et))
            else:
                # Force current start to previous end
                st = prev_et
                if et < st:
                    et = st
                out.append((act, st, et))

        # Stretch last non-anchor stay to day end
        last_idx = len(out) - 1
        if anchor_idx is not None and last_idx == anchor_idx:
            # Do not stretch if anchor is the last; leave as is
            pass
        else:
            last_act, last_st, _ = out[-1]
            out[-1] = (last_act, last_st, total_len)

        return out

    anchor_tuple = (anchor_stay.activity_label, pert_start, pert_end)
    synthetic_stays = _stretch_anchor_aware(synthetic_stays, anchor_tuple, day_length)
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
                       randomness_levels: List[float], paths: PathsConfig, tz) -> None:
    """Process a single user and generate all outputs."""
    print(f"Processing user {user_id}...")
    
    # Enrich IMN with POI data
    enriched = enrich_imn_with_poi(imn, poi_info)
    
    # Extract stays from trips
    stays = extract_stays_from_trips(enriched['trips'], enriched['locations'])
    
    # Group stays by day (keep original, non-stretched for distributions/visualization)
    stays_by_day = extract_stays_by_day(stays, tz)
    
    if not stays_by_day:
        print(f"  ⚠ No valid stays found for user {user_id}")
        return
    
    # Build user-specific distributions
    user_duration_probs, user_transition_probs, user_trip_duration_probs = build_stay_distributions(stays_by_day)
    
    # Generate probability report
    user_probs_report(user_duration_probs, user_transition_probs, user_trip_duration_probs, 
                     user_id, paths.prob_dir())
    
    # Prepare day data for visualization
    day_data = prepare_day_data(stays_by_day, user_duration_probs, user_transition_probs, 
                               randomness_levels, tz)
    
    # Generate timeline visualization
    fig = visualize_day_data(day_data, user_id=user_id)
    timeline_path = os.path.join(paths.vis_dir(), f"user_{user_id}_timelines.png")
    os.makedirs(os.path.dirname(timeline_path), exist_ok=True)
    fig.savefig(timeline_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Timeline visualization saved: {os.path.basename(timeline_path)}")


def load_datasets(paths: PathsConfig) -> Tuple[Dict, Dict]:
    print("Loading data...")
    # Load full dataset first to get user IDs
    full_imns = imn_loading.read_imn(paths.full_imn_path)
    filtered_user_ids = list(full_imns.keys())

    # Load test dataset (subset)
    imns = imn_loading.read_imn(paths.test_imn_path)
    # Keep only users that exist in both datasets
    imns = {k: imns[k] for k in filtered_user_ids if k in imns}

    poi_data = read_poi_data(paths.poi_path)
    print(f"✓ Loaded {len(imns)} IMNs and POI data for {len(poi_data)} users")
    return imns, poi_data


def run_pipeline(paths: PathsConfig, randomness_levels: List[float], tz) -> None:
    print("Starting Individual Mobility Network Generation Process")
    print("=" * 60)

    ensure_output_structure(paths)

    print(f"Results will be saved to: {paths.results_dir}")
    print(f"  - User probability reports: {paths.results_dir}/{paths.prob_subdir}/")
    print(f"  - Timeline visualizations: {paths.results_dir}/{paths.vis_subdir}/")
    print()

    try:
        imns, poi_data = load_datasets(paths)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    print(f"\nProcessing {len(imns)} users...")
    print("-" * 40)
    for i, (user_id, imn) in enumerate(imns.items(), 1):
        print(f"[{i}/{len(imns)}] ", end="")
        if user_id not in poi_data:
            print(f"⚠ No POI data found for user {user_id}, skipping...")
            continue
        try:
            process_single_user(user_id, imn, poi_data[user_id], randomness_levels, paths, tz)
        except Exception as e:
            print(f"❌ Error processing user {user_id}: {e}")
            continue

    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Results saved to: {paths.results_dir}")


def main(argv: Optional[List[str]] = None):
    paths = parse_args(argv)
    run_pipeline(paths, RANDOMNESS_LEVELS, TIMEZONE)


if __name__ == "__main__":
    main()
