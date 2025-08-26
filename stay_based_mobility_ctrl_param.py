import gzip
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import pytz
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import os
import random

import imn_loading

# Create output directory for visualizations
VISUALIZATION_DIR = "stay_visualizations6"
PROBABILITY_VIS_DIR = "probability_visualizations6"
if not os.path.exists(VISUALIZATION_DIR):
    os.makedirs(VISUALIZATION_DIR)
if not os.path.exists(PROBABILITY_VIS_DIR):
    os.makedirs(PROBABILITY_VIS_DIR)

# Configuration parameters
USE_GLOBAL_DISTRIBUTIONS = False  # Set to True for population-level distributions
ANCHOR_JITTER_START_TIME = True  # Jitter anchor stay start time
ANCHOR_JITTER_AMOUNT = 1800  # ±30 minutes in seconds
SAMPLE_ANCHOR_DURATION = True  # Sample anchor duration from distribution
ADD_TRIP_GAPS = True  # Add gaps between stays to represent trips
TRIP_GAP_JITTER = True  # Jitter trip gap durations

# Control parameter for behavior replication
CONTROL_LEVEL = 0.5  # For distribution interpolation: 0=global (random), 1=user-specific (clone)
# Synthetic generation randomness (per colleague suggestions) is derived at call site as:
# copy_probability = 1 - control_level

def read_poi_data(filepath):
    poi_data = {}
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            # Keep UID as int to match IMN data format
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


# --- Timezone config ---
tz = pytz.timezone("Europe/Rome")

# --- Activity color mapping ---
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

class Stay:
    def __init__(self, location_id, activity_label, start_time, end_time):
        self.location_id = location_id
        self.activity_label = activity_label
        self.start_time = start_time
        self.end_time = end_time
        self.duration = self.end_time - self.start_time if self.end_time is not None and self.start_time is not None else None

    def set_end_time(self, end_time):
        self.end_time = end_time
        if self.start_time is not None:
            self.duration = self.end_time - self.start_time

    def to_dict(self):
        return {
            'location_id': self.location_id,
            'activity_label': self.activity_label,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration
        }

def extract_stays_from_trips(trips, locations):
    """Convert trips into stays by considering the destination of each trip as a stay."""
    stays = []
    
    # First pass: create stays with start times
    for from_id, to_id, st, et in trips:
        activity_label = locations[to_id].get('activity_label', 'unknown')
        # Create stay with start time as trip end time
        stays.append(Stay(to_id, activity_label, et, None))
    
    # Second pass: set end times based on next stay's start time
    for i in range(len(stays)-1):
        stays[i].set_end_time(stays[i+1].start_time)
    
    # Handle the last stay
    if stays:
        # For the last stay, if there's a next day's first trip, use that as end time
        # Otherwise, use a default duration of 1 hour
        last_stay = stays[-1]
        if last_stay.start_time is not None:
            last_stay.set_end_time(last_stay.start_time + 3600)  # Default 1 hour duration
    
    return stays

def extract_stays_by_day(stays):
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

def build_stay_distributions(stays_by_day):
    """Build distributions for stay durations and activity types across all days."""
    duration_dist = defaultdict(list)
    activity_transitions = defaultdict(list)
    trip_durations = []
    
    # Collect data from all days
    for day, day_stays in stays_by_day.items():
        for i in range(len(day_stays)-1):
            current_stay = day_stays[i]
            next_stay = day_stays[i+1]
            
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
            hist, bins = np.histogram(durations, bins=20, density=False)  # Use counts instead of density
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
        hist, bins = np.histogram(trip_durations, bins=20, density=False)  # Use counts instead of density
        trip_duration_probs = (hist, bins)
    
    return duration_probs, transition_probs, trip_duration_probs

def visualize_probabilities(uid, duration_probs, transition_probs, trip_duration_probs):
    """Create visualizations of the probability distributions for a user."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Probability Distributions for User {uid}', fontsize=16)
    
    # Debug: print what we have
    print(f"User {uid} - Duration probs: {len(duration_probs)} activities")
    for activity, (hist, bins) in duration_probs.items():
        if len(hist) > 0:
            print(f"  {activity}: {len(hist)} bins, sum={hist.sum():.3f}")
        else:
            print(f"  {activity}: empty")
    
    # Plot 1: Duration distributions (histogram)
    ax1 = axes[0, 0]
    has_data = False
    for activity, (hist, bins) in duration_probs.items():
        if len(hist) > 0 and hist.sum() > 0:
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax1.bar(bin_centers, hist, width=bins[1]-bins[0], alpha=0.7, 
                   label=activity, color=ACTIVITY_COLORS.get(activity, 'gray'))
            has_data = True
    
    if not has_data:
        ax1.text(0.5, 0.5, 'No duration data available', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        print(f"Warning: No duration data for user {uid}")
    
    ax1.set_xlabel('Duration (seconds)')
    ax1.set_ylabel('Count')
    ax1.set_title('Stay Duration Distributions')
    if has_data:
        ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Transition probabilities heatmap
    ax2 = axes[0, 1]
    if transition_probs:
        activities = list(set([k for k in transition_probs.keys()] + 
                            [v for d in transition_probs.values() for v in d.keys()]))
        activities.sort()
        
        # Create transition matrix
        transition_matrix = np.zeros((len(activities), len(activities)))
        for i, from_act in enumerate(activities):
            for j, to_act in enumerate(activities):
                if from_act in transition_probs and to_act in transition_probs[from_act]:
                    transition_matrix[i, j] = transition_probs[from_act][to_act]
        
        im = ax2.imshow(transition_matrix, cmap='Blues', aspect='auto')
        ax2.set_xticks(range(len(activities)))
        ax2.set_yticks(range(len(activities)))
        ax2.set_xticklabels(activities, rotation=45, ha='right')
        ax2.set_yticklabels(activities)
        ax2.set_title('Activity Transition Probabilities')
        plt.colorbar(im, ax=ax2)
    
    # Plot 3: Trip duration distribution
    ax3 = axes[1, 0]
    if trip_duration_probs is not None:
        hist, bins = trip_duration_probs
        if len(hist) > 0 and hist.sum() > 0:
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax3.plot(bin_centers, hist, color='red', linewidth=2)
            ax3.set_xlabel('Trip Duration (seconds)')
            ax3.set_ylabel('Probability Density')
            ax3.set_title('Trip Duration Distribution')
            ax3.grid(True, alpha=0.3)
    
    # Plot 4: Activity frequency
    ax4 = axes[1, 1]
    activity_counts = defaultdict(int)
    for activity, durations in duration_probs.items():
        activity_counts[activity] = len(durations)
    
    if activity_counts:
        activities = list(activity_counts.keys())
        counts = list(activity_counts.values())
        colors = [ACTIVITY_COLORS.get(act, 'gray') for act in activities]
        ax4.bar(activities, counts, color=colors)
        ax4.set_xlabel('Activity Type')
        ax4.set_ylabel('Number of Stays')
        ax4.set_title('Activity Frequency')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_file = os.path.join(PROBABILITY_VIS_DIR, f"user_{uid}_probabilities.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_global_probabilities(global_duration_dist, global_transition_dist, global_trip_dist):
    """Create visualizations of the global probability distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Global Probability Distributions (All Users)', fontsize=16)
    
    # Plot 1: Global duration distributions (histogram)
    ax1 = axes[0, 0]
    has_data = False
    for activity, (hist, bins) in global_duration_dist.items():
        if len(hist) > 0 and hist.sum() > 0:
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax1.bar(bin_centers, hist, width=bins[1]-bins[0], alpha=0.7, 
                   label=activity, color=ACTIVITY_COLORS.get(activity, 'gray'))
            has_data = True
    
    if not has_data:
        ax1.text(0.5, 0.5, 'No global duration data available', 
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        print("Warning: No global duration data")
    
    ax1.set_xlabel('Duration (seconds)')
    ax1.set_ylabel('Count')
    ax1.set_title('Global Stay Duration Distributions')
    if has_data:
        ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Global transition probabilities heatmap
    ax2 = axes[0, 1]
    if global_transition_dist:
        activities = list(set([k for k in global_transition_dist.keys()] + 
                            [v for d in global_transition_dist.values() for v in d.keys()]))
        activities.sort()
        
        # Create transition matrix
        transition_matrix = np.zeros((len(activities), len(activities)))
        for i, from_act in enumerate(activities):
            for j, to_act in enumerate(activities):
                if from_act in global_transition_dist and to_act in global_transition_dist[from_act]:
                    transition_matrix[i, j] = global_transition_dist[from_act][to_act]
        
        im = ax2.imshow(transition_matrix, cmap='Blues', aspect='auto')
        ax2.set_xticks(range(len(activities)))
        ax2.set_yticks(range(len(activities)))
        ax2.set_xticklabels(activities, rotation=45, ha='right')
        ax2.set_yticklabels(activities)
        ax2.set_title('Global Activity Transition Probabilities')
        plt.colorbar(im, ax=ax2)
    
    # Plot 3: Global trip duration distribution
    ax3 = axes[1, 0]
    if global_trip_dist is not None:
        hist, bins = global_trip_dist
        if len(hist) > 0 and hist.sum() > 0:
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax3.bar(bin_centers, hist, width=bins[1]-bins[0], alpha=0.7, color='red')
            ax3.set_xlabel('Trip Duration (seconds)')
            ax3.set_ylabel('Probability Density')
            ax3.set_title('Global Trip Duration Distribution')
            ax3.grid(True, alpha=0.3)
    
    # Plot 4: Global activity frequency
    ax4 = axes[1, 1]
    activity_counts = defaultdict(int)
    for activity, (hist, bins) in global_duration_dist.items():
        if len(hist) > 0:
            # Estimate total count from histogram
            bin_width = bins[1] - bins[0]
            total_count = int(hist.sum() * bin_width * len(bins))
            activity_counts[activity] = total_count
    
    if activity_counts:
        activities = list(activity_counts.keys())
        counts = list(activity_counts.values())
        colors = [ACTIVITY_COLORS.get(act, 'gray') for act in activities]
        ax4.bar(activities, counts, color=colors)
        ax4.set_xlabel('Activity Type')
        ax4.set_ylabel('Estimated Total Stays')
        ax4.set_title('Global Activity Frequency')
        ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    output_file = os.path.join(PROBABILITY_VIS_DIR, "global_probabilities.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def sample_stay_duration(activity, duration_probs):
    """Sample a duration for a given activity type."""
    if activity not in duration_probs:
        return 3600  # Default 1 hour
    
    hist, bins = duration_probs[activity]
    if hist.sum() == 0:
        return 3600
    
    bin_idx = np.random.choice(len(hist), p=hist/hist.sum())
    return np.random.uniform(bins[bin_idx], bins[bin_idx+1])

def sample_trip_duration(trip_duration_probs):
    """Sample a trip duration."""
    if trip_duration_probs is None:
        return 900  # Default 15 minutes
    
    hist, bins = trip_duration_probs
    if hist.sum() == 0:
        return 900
    
    bin_idx = np.random.choice(len(hist), p=hist/hist.sum())
    return np.random.uniform(bins[bin_idx], bins[bin_idx+1])

def sample_next_activity(current_activity, transition_probs):
    """Sample the next activity based on transition probabilities."""
    if current_activity not in transition_probs:
        return 'unknown'
    
    activities = list(transition_probs[current_activity].keys())
    probs = list(transition_probs[current_activity].values())
    return np.random.choice(activities, p=probs)

def interpolate_distributions(user_dist, global_dist, control_level):
    """
    Interpolate between user-specific and global distributions based on control_level.
    
    Args:
        user_dist: User-specific distribution
        global_dist: Global distribution
        control_level: Float between 0 and 1
                      0 = use only global_dist
                      1 = use only user_dist
                      0.5 = mix both equally
    
    Returns:
        Interpolated distribution
    """
    if control_level == 0:
        return global_dist
    elif control_level == 1:
        return user_dist
    else:
        # For duration distributions (hist, bins format) - check this first
        if isinstance(user_dist, tuple) and isinstance(global_dist, tuple):
            user_hist, user_bins = user_dist
            global_hist, global_bins = global_dist
            
            # Ensure same bin structure
            if len(user_bins) != len(global_bins):
                # Use global bins if different
                return global_dist
            
            # Interpolate histograms
            interpolated_hist = (control_level * user_hist + 
                               (1 - control_level) * global_hist)
            return (interpolated_hist, user_bins)
        
        # For transition probabilities (dict format)
        elif isinstance(user_dist, dict) and isinstance(global_dist, dict):
            interpolated_dict = {}
            all_activities = set(user_dist.keys()) | set(global_dist.keys())
            
            for activity in all_activities:
                user_probs = user_dist.get(activity, {})
                global_probs = global_dist.get(activity, {})
                all_targets = set(user_probs.keys()) | set(global_probs.keys())
                
                interpolated_probs = {}
                for target in all_targets:
                    user_prob = user_probs.get(target, 0.0)
                    global_prob = global_probs.get(target, 0.0)
                    interpolated_prob = (control_level * user_prob + 
                                       (1 - control_level) * global_prob)
                    if interpolated_prob > 0:
                        interpolated_probs[target] = interpolated_prob
                
                if interpolated_probs:
                    # Normalize probabilities
                    total = sum(interpolated_probs.values())
                    if total > 0:
                        interpolated_dict[activity] = {k: v/total for k, v in interpolated_probs.items()}
            
            return interpolated_dict
        
        # For single values (like trip duration distributions)
        elif isinstance(user_dist, tuple) and isinstance(global_dist, tuple):
            user_hist, user_bins = user_dist
            global_hist, global_bins = global_dist
            
            if len(user_bins) != len(global_bins):
                return global_dist
            
            interpolated_hist = (control_level * user_hist + 
                               (1 - control_level) * global_hist)
            return (interpolated_hist, user_bins)
        
        else:
            # Fallback: use control_level to choose between distributions
            if np.random.random() < control_level:
                return user_dist
            else:
                return global_dist

def safe_jitter_time(base_time, jitter_amount, min_time=None, max_time=None):
    """Add jitter to a timestamp with safety checks."""
    if not jitter_amount:
        return base_time
    
    # Calculate jitter with bounds
    jitter = np.random.uniform(-jitter_amount, jitter_amount)
    jittered_time = base_time + jitter
    
    # Ensure we don't go below minimum time
    if min_time is not None:
        jittered_time = max(jittered_time, min_time)
    
    # Ensure we don't go above maximum time
    if max_time is not None:
        jittered_time = min(jittered_time, max_time)
    
    return jittered_time

def _coin_flip_activity(prev_activity, original_activity, transition_probs, copy_probability):
    """Choose next activity using coin flip method.
    With probability copy_probability, copy the original activity; otherwise sample from transition_probs.
    """
    if np.random.random() < copy_probability and original_activity is not None:
        return original_activity
    return sample_next_activity(prev_activity if prev_activity is not None else original_activity, transition_probs)

def _blend_duration(original_duration, sampled_duration, copy_probability):
    """Convex combination of durations per colleague suggestions: d = p*orig + (1-p)*sampled."""
    if original_duration is None:
        return sampled_duration
    if sampled_duration is None:
        return original_duration
    return copy_probability * original_duration + (1.0 - copy_probability) * sampled_duration

def _build_segment_from_original(original_segment, seg_start_ts, seg_end_ts,
                                 duration_probs, transition_probs, copy_probability):
    """Build a synthetic segment (before or after anchor) following colleague suggestions.
    - At each step, flip coin to decide: copy original activity OR sample from distributions
    - Duration = p*orig + (1-p)*sampled (convex combination)
    - Enforce alignment: if last overlaps end -> cut; if ends early -> rescale durations to fit exactly
    """
    synthetic = []
    t_cursor = seg_start_ts
    prev_activity = None
    
    # Process each original stay in sequence
    for stay in original_segment:
        # Coin flip for activity: copy original OR sample from transition probs
        if np.random.random() < copy_probability:
            # Copy original activity exactly
            act = stay.activity_label
            orig_dur = max(0, (stay.end_time or seg_end_ts) - (stay.start_time or seg_start_ts))
        else:
            # Sample from transition probabilities
            act = sample_next_activity(prev_activity if prev_activity is not None else stay.activity_label, transition_probs)
            orig_dur = max(0, (stay.end_time or seg_end_ts) - (stay.start_time or seg_start_ts))
        
        # Duration: convex combination of original and sampled
        sampled_dur = sample_stay_duration(act, duration_probs)
        dur = _blend_duration(orig_dur, sampled_dur, copy_probability)
        dur = max(60, float(dur))  # keep a minimum 1 minute
        
        # Place this stay
        if t_cursor >= seg_end_ts:
            break
        end_ts = min(seg_end_ts, int(t_cursor + dur))
        synthetic.append(Stay(None, act, int(t_cursor), int(end_ts)))
        t_cursor = end_ts
        prev_activity = act

    # If we ended before seg_end_ts, rescale durations to stretch to seg_end_ts
    if synthetic and t_cursor < seg_end_ts:
        total = sum(s.end_time - s.start_time for s in synthetic)
        if total > 0:
            scale = (seg_end_ts - synthetic[0].start_time) / total
            t = synthetic[0].start_time
            for s in synthetic:
                new_len = int((s.end_time - s.start_time) * scale)
                s.start_time = t
                s.end_time = min(seg_end_ts, t + max(60, new_len))
                s.duration = s.end_time - s.start_time
                t = s.end_time
            # ensure exact end
            if synthetic[-1].end_time != seg_end_ts:
                synthetic[-1].end_time = seg_end_ts
                synthetic[-1].duration = synthetic[-1].end_time - synthetic[-1].start_time
    return synthetic

def save_timeline_to_text(original_stays, synthetic_stays, user_id, day, control_level, output_dir):
    """Save timeline comparison to text file for debugging."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    filename = os.path.join(output_dir, f"user_{user_id}_day_{day}_control_{control_level}.txt")
    
    with open(filename, 'w') as f:
        f.write(f"Timeline Comparison for User {user_id}, Day {day}, Control Level {control_level}\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("ORIGINAL TIMELINE:\n")
        f.write("-" * 40 + "\n")
        for i, stay in enumerate(original_stays):
            f.write(f"{i+1:2d}. {stay.activity_label:12s} | {datetime.fromtimestamp(stay.start_time, tz).strftime('%H:%M')} - {datetime.fromtimestamp(stay.end_time, tz).strftime('%H:%M')} | Duration: {stay.duration//60:3d}min\n")
        
        f.write("\nSYNTHETIC TIMELINE:\n")
        f.write("-" * 40 + "\n")
        for i, stay in enumerate(synthetic_stays):
            f.write(f"{i+1:2d}. {stay.activity_label:12s} | {datetime.fromtimestamp(stay.start_time, tz).strftime('%H:%M')} - {datetime.fromtimestamp(stay.end_time, tz).strftime('%H:%M')} | Duration: {stay.duration//60:3d}min\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Control Level: {control_level} (Copy Probability: {1-control_level:.2f})\n")
        f.write(f"Original stays: {len(original_stays)}, Synthetic stays: {len(synthetic_stays)}\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

def generate_synthetic_day(anchor_stay, duration_probs, transition_probs, trip_duration_probs,
                           original_day_stays,
                           control_level=CONTROL_LEVEL):
    """Generate a synthetic day around anchor using colleague suggestions.
    - copy_probability = 1 - control_level
    - Before anchor: follow original sequence up to anchor, align exactly at anchor start
    - After anchor: follow original sequence after anchor, align exactly at midnight
    """
    copy_probability = 1.0 - control_level
    # Anchor times (may be jittered and/or re-sampled for duration)
    anchor_start_dt = datetime.fromtimestamp(anchor_stay.start_time, tz)
    anchor_end_dt = datetime.fromtimestamp(anchor_stay.end_time, tz)
    day_start_dt = anchor_start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    day_end_dt = day_start_dt + timedelta(days=1)

    # Optionally adjust anchor duration and start-time jitter
    anchor_duration = anchor_stay.duration
    if SAMPLE_ANCHOR_DURATION:
        anchor_duration = sample_stay_duration(anchor_stay.activity_label, duration_probs)
        anchor_end_dt = anchor_start_dt + timedelta(seconds=anchor_duration)
    if ANCHOR_JITTER_START_TIME:
        jitter_amount = ANCHOR_JITTER_AMOUNT * (1 - control_level)
        jittered_start = safe_jitter_time(anchor_start_dt.timestamp(), jitter_amount, min_time=day_start_dt.timestamp())
        anchor_start_dt = datetime.fromtimestamp(jittered_start, tz)
        anchor_end_dt = anchor_start_dt + timedelta(seconds=anchor_duration)

    # Split original stays into before/after anchor
    before_segment = [s for s in original_day_stays if s.end_time is not None and s.end_time <= int(anchor_start_dt.timestamp())]
    after_segment = [s for s in original_day_stays if s.start_time is not None and s.start_time >= int(anchor_end_dt.timestamp())]

    # Build segments with alignment constraints
    before = _build_segment_from_original(
        before_segment,
        int(day_start_dt.timestamp()),
        int(anchor_start_dt.timestamp()),
        duration_probs,
        transition_probs,
        copy_probability,
    )

    # Anchor (kept as-is at computed times)
    anchor_synth = Stay(None, anchor_stay.activity_label, int(anchor_start_dt.timestamp()), int(anchor_end_dt.timestamp()))

    after = _build_segment_from_original(
        after_segment,
        int(anchor_end_dt.timestamp()),
        int(day_end_dt.timestamp()),
        duration_probs,
        transition_probs,
        copy_probability,
    )

    return before + [anchor_synth] + after

def find_anchor_stay_for_day(stays):
    """Find the longest non-home stay for a given day."""
    return max([s for s in stays if s.activity_label != 'home'], 
              key=lambda s: s.duration, default=None)

def get_day_name(date):
    """Get day name from date."""
    return date.strftime('%A')

def plot_stays(stays, y_offset=0, highlight_stay=None, ax=None):
    """Plot stays as horizontal bars."""
    for stay in stays:
        st_local = datetime.fromtimestamp(stay.start_time, pytz.utc).astimezone(tz)
        et_local = datetime.fromtimestamp(stay.end_time, pytz.utc).astimezone(tz)
        color = ACTIVITY_COLORS.get(stay.activity_label, 'gray')
        
        # If this is the anchor stay, make it more prominent (bigger size and text, but same color)
        if stay == highlight_stay:
            if ax is None:
                plt.hlines(y=y_offset, xmin=st_local, xmax=et_local, color=color, 
                          linewidth=15, label='Anchor Stay')
                # Add a text label
                plt.text(st_local, y_offset + 0.02, stay.activity_label, 
                        fontsize=8, ha='center')
            else:
                ax.hlines(y=y_offset, xmin=st_local, xmax=et_local, color=color, 
                         linewidth=15, label='Anchor Stay')
                # Add a text label
                ax.text(st_local, y_offset + 0.02, stay.activity_label, 
                       fontsize=8, ha='center')
        else:
            if ax is None:
                plt.hlines(y=y_offset, xmin=st_local, xmax=et_local, color=color, 
                          linewidth=10)
            else:
                ax.hlines(y=y_offset, xmin=st_local, xmax=et_local, color=color, 
                         linewidth=10)

def collect_global_statistics(imns, poi_data):
    """First pass: collect global statistics from all users."""
    print("Collecting global statistics from all users...")
    
    global_duration_probs = defaultdict(list)
    global_transition_probs = defaultdict(list)
    global_trip_durations = []
    
    total_users = len(imns)
    processed_users = 0
    
    for uid_str, imn in imns.items():
        processed_users += 1
        if processed_users % 50 == 0:
            print(f"Collecting stats: {processed_users}/{total_users}")
            
        uid = uid_str  # Keep as string since that's how it's stored
        if uid not in poi_data:
            continue
            
        # Enrich IMN with POI data
        enriched = enrich_imn_with_poi(imn, poi_data[uid])
        
        # Extract stays from trips
        stays = extract_stays_from_trips(enriched['trips'], enriched['locations'])
        
        # Group stays by day
        stays_by_day = extract_stays_by_day(stays)
        
        # Collect raw data directly from stays (not from build_stay_distributions)
        for day, day_stays in stays_by_day.items():
            for i in range(len(day_stays)-1):
                current_stay = day_stays[i]
                next_stay = day_stays[i+1]
                
                # Record duration for this activity type
                if current_stay.duration is not None:
                    global_duration_probs[current_stay.activity_label].append(current_stay.duration)
                
                # Record activity transition
                global_transition_probs[current_stay.activity_label].append(next_stay.activity_label)
                
                # Record trip duration (gap between stays)
                if current_stay.end_time is not None and next_stay.start_time is not None:
                    trip_duration = next_stay.start_time - current_stay.end_time
                    if trip_duration > 0:
                        global_trip_durations.append(trip_duration)
    
    # Build final global distributions
    global_duration_dist = {}
    for activity, durations in global_duration_probs.items():
        if len(durations) > 0:
            print(f"Global {activity}: {len(durations)} durations, range: {min(durations):.1f}-{max(durations):.1f}")
            hist, bins = np.histogram(durations, bins=20, density=False)  # Use counts instead of density
            global_duration_dist[activity] = (hist, bins)
    
    global_transition_dist = {}
    for from_activity, to_activities in global_transition_probs.items():
        if len(to_activities) > 0:
            unique_activities, counts = np.unique(to_activities, return_counts=True)
            probs = counts / counts.sum()
            global_transition_dist[from_activity] = dict(zip(unique_activities, probs))
    
    global_trip_dist = None
    if len(global_trip_durations) > 0:
        print(f"Global trip durations: {len(global_trip_durations)} trips, range: {min(global_trip_durations):.1f}-{max(global_trip_durations):.1f}")
        hist, bins = np.histogram(global_trip_durations, bins=20, density=False)  # Use counts instead of density
        global_trip_dist = (hist, bins)
    
    print(f"Global statistics collected: {len(global_duration_dist)} activity types, "
          f"{len(global_transition_dist)} transition types, "
          f"{len(global_trip_durations)} trip durations")
    
    return global_duration_dist, global_transition_dist, global_trip_dist

def get_original_user_ids():
    """Extract user IDs from the original Milano IMNs file."""
    original_user_ids = set()
    try:
        with gzip.open('data/milano_2007_imns.json.gz', 'rt', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        imn = json.loads(line)
                        if 'uid' in imn:
                            # Convert to int to match test data format
                            original_user_ids.add(int(imn['uid']))
                    except json.JSONDecodeError:
                        continue
    except Exception as e:
        print(f"Error reading original file: {e}")
        return set()
    
    print(f"Found {len(original_user_ids)} users in original Milano file")
    return original_user_ids

def visualize_original_timelines_compact(users_per_page=10):
    """
    Visualize original timelines for users from the original Milano dataset in a compact format.
    Shows each user's different days stacked vertically.
    Creates multiple pages with users_per_page users per page.
    """
    print("Loading data for original timeline visualization...")
    
    # Get original user IDs
    original_user_ids = get_original_user_ids()
    
    # Load IMNs and POI data from the new test files
    imns = imn_loading.read_imn('data/test_milano_imns.json.gz')
    poi_data = read_poi_data('data/test_milano_imns_pois.json.gz')
    print(f"Loaded {len(imns)} IMNs and POI data for {len(poi_data)} users")
    
    # Select only users that exist in both the original file and the new test files
    available_users = [uid for uid in list(imns.keys()) if uid in original_user_ids and uid in poi_data]
    
    print(f"Found {len(available_users)} users from original dataset with POI data")
    
    # Group users into pages
    pages = [available_users[i:i + users_per_page] for i in range(0, len(available_users), users_per_page)]
    print(f"Creating {len(pages)} pages with {users_per_page} users per page")
    
    # Import matplotlib.backends for PDF support
    from matplotlib.backends.backend_pdf import PdfPages
    
    # Create output directory
    output_dir = "results/original_milano_original_timelines"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create PDF file
    pdf_filename = os.path.join(output_dir, "original_timelines_all_users.pdf")
    with PdfPages(pdf_filename) as pdf:
        
        # Process each page
        for page_idx, page_users in enumerate(pages):
            print(f"Creating page {page_idx + 1}/{len(pages)} with {len(page_users)} users")
            
            # Create figure for this page
            fig, axes = plt.subplots(len(page_users), 1, figsize=(16, 2 * len(page_users)), sharex=True)
            if len(page_users) == 1:
                axes = [axes]
            
            # Process each user on this page
            for user_idx, uid in enumerate(page_users):
                print(f"  Processing user {user_idx + 1}/{len(page_users)}: {uid}")
                
                imn = imns[uid]
                enriched = enrich_imn_with_poi(imn, poi_data[uid])
                
                # Extract stays from trips
                stays = extract_stays_from_trips(enriched['trips'], enriched['locations'])
                
                # Group stays by day
                stays_by_day = extract_stays_by_day(stays)
                
                ax = axes[user_idx]
                
                # Plot each day for this user (earliest days at top, latest days at bottom)
                # Sort days chronologically (they are already date objects from extract_stays_by_day)
                sorted_days = sorted(stays_by_day.keys())
                
                # Calculate total height needed for all days
                total_height = len(sorted_days) * 0.3
                y_offset = 0.5 + total_height - 0.3  # Start from top (highest y value)
                
                for day in sorted_days:  # Plot earliest days at top, latest days at bottom
                    day_stays = stays_by_day[day]
                    plot_stays(day_stays, y_offset=y_offset, ax=ax)
                    y_offset -= 0.3  # Move down (decrease y value)
                
                # Customize subplot
                # Calculate y-positions for labels (top to bottom)
                y_positions = [0.5 + total_height - 0.3 - i * 0.3 for i in range(len(sorted_days))]
                ax.set_yticks(y_positions)
                ax.set_yticklabels([f"User {uid} - {day}" for day in sorted_days], fontsize=8)
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
            fig.suptitle(f"Original User Timelines - Page {page_idx + 1}/{len(pages)} (Users {page_idx * users_per_page + 1}-{page_idx * users_per_page + len(page_users)})", 
                        fontsize=14, y=0.95)
            
            plt.tight_layout(rect=[0, 0, 0.85, 0.95])
            
            # Save page to PDF
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    
    print(f"Saved all pages to {pdf_filename}")

def test_control_levels(user_id=650, test_levels=[0.0, 0.25, 0.5, 0.75, 1.0]):
    """
    Test different control levels for a specific user and create comparison visualization.
    """
    print(f"Testing control levels for user {user_id}")
    
    # Get original user IDs
    original_user_ids = get_original_user_ids()
    
    # Load IMNs and POI data from the new test files
    imns = imn_loading.read_imn('data/test_milano_imns.json.gz')
    poi_data = read_poi_data('data/test_milano_imns_pois.json.gz')
    
    # Use user_id directly (it's already an int)
    if user_id not in imns or str(user_id) not in poi_data:
        print(f"User {user_id} not found in data")
        return
    
    # Check if user is from original dataset
    if user_id not in original_user_ids:
        print(f"User {user_id} is not from the original Milano dataset")
        return
    
    # Get user data
    imn = imns[user_id]
    enriched = enrich_imn_with_poi(imn, poi_data[str(user_id)])
    stays = extract_stays_from_trips(enriched['trips'], enriched['locations'])
    stays_by_day = extract_stays_by_day(stays)
    
    # Filter to only include original users for global statistics
    filtered_imns = {uid: imn for uid, imn in imns.items() if uid in original_user_ids}
    filtered_poi_data = {uid: data for uid, data in poi_data.items() if uid in original_user_ids}
    
    # Collect global statistics
    global_duration_dist, global_transition_dist, global_trip_dist = collect_global_statistics(filtered_imns, filtered_poi_data)
    
    # Debug: print what we got
    print(f"Global duration dist: {type(global_duration_dist)}, keys: {list(global_duration_dist.keys()) if isinstance(global_duration_dist, dict) else 'not a dict'}")
    print(f"Global transition dist: {type(global_transition_dist)}, keys: {list(global_transition_dist.keys()) if isinstance(global_transition_dist, dict) else 'not a dict'}")
    
    # Build user-specific distributions
    user_duration_probs, user_transition_probs, user_trip_duration_probs = build_stay_distributions(stays_by_day)
    
    # Debug: print user distributions
    print(f"User duration probs: {type(user_duration_probs)}, keys: {list(user_duration_probs.keys()) if isinstance(user_duration_probs, dict) else 'not a dict'}")
    print(f"User transition probs: {type(user_transition_probs)}, keys: {list(user_transition_probs.keys()) if isinstance(user_transition_probs, dict) else 'not a dict'}")
    
    # Create comparison plot
    fig, axes = plt.subplots(len(test_levels) + 1, 1, figsize=(16, 3 * (len(test_levels) + 1)), sharex=True)
    if len(test_levels) + 1 == 1:
        axes = [axes]
    
    # Plot original timeline
    ax_original = axes[0]
    # Find a day with anchor stay
    test_day = None
    test_anchor = None
    for day, day_stays in stays_by_day.items():
        anchor = find_anchor_stay_for_day(day_stays)
        if anchor:
            test_day = day
            test_anchor = anchor
            break
    
    if test_day and test_anchor:
        day_stays = stays_by_day[test_day]
        plot_stays(day_stays, y_offset=0.5, ax=ax_original)
        ax_original.set_yticks([0.5])
        ax_original.set_yticklabels(["Original"], fontsize=10)
        ax_original.set_title(f"User {user_id} - Original Timeline ({test_day})", fontsize=12)
        ax_original.grid(True, axis='x', linestyle=':', alpha=0.3)
    
    # Plot synthetic timelines for different control levels
    for i, control_level in enumerate(test_levels):
        ax = axes[i + 1]
        
        # Interpolate distributions (handle empty global distributions)
        if not global_duration_dist or not global_transition_dist:
            print("Warning: Global distributions are empty, using user-specific distributions only")
            duration_probs = user_duration_probs
            transition_probs = user_transition_probs
            trip_duration_probs = user_trip_duration_probs
        else:
            duration_probs = interpolate_distributions(user_duration_probs, global_duration_dist, control_level)
            transition_probs = interpolate_distributions(user_transition_probs, global_transition_dist, control_level)
            trip_duration_probs = interpolate_distributions(user_trip_duration_probs, global_trip_dist, control_level)
        
        # Generate synthetic day
        synthetic_stays = generate_synthetic_day(
            test_anchor,
            duration_probs,
            transition_probs,
            trip_duration_probs,
            stays_by_day[test_day],
            control_level=control_level,
        )
        
        # Save timeline comparison to text file
        save_timeline_to_text(stays_by_day[test_day], synthetic_stays, user_id, test_day, control_level, "timeline_text_outputs")
        
        # Plot synthetic stays
        plot_stays(synthetic_stays, y_offset=0.5, ax=ax)
        ax.set_yticks([0.5])
        ax.set_yticklabels([f"Control Level: {control_level}"], fontsize=10)
        ax.set_title(f"User {user_id} - Synthetic Timeline (Control Level: {control_level})", fontsize=12)
        ax.grid(True, axis='x', linestyle=':', alpha=0.3)
    
    # Set x-axis formatting for the last subplot
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[-1].set_xlabel("Time", fontsize=12)
    
    # Add legend
    legend_handles = [mpatches.Patch(color=c, label=l) for l, c in ACTIVITY_COLORS.items()]
    fig.legend(handles=legend_handles, title="Activity Types", 
              loc='upper right', fontsize=10, title_fontsize=11)
    
    # Add title
    fig.suptitle(f"Control Level Comparison for User {user_id}", fontsize=14, y=0.95)
    
    plt.tight_layout(rect=[0, 0, 0.85, 0.95])
    
    # Save figure to results folder
    output_dir = "results/original_milano_original_timelines"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_file = os.path.join(output_dir, f"control_level_comparison_user_{user_id}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved control level comparison to {output_file}")
    
    plt.close(fig)

def main():
    print("Starting improved stay-based mobility analysis...")
    
    # Get original user IDs
    original_user_ids = get_original_user_ids()
    
    # Load IMNs and POI data from the new test files
    imns = imn_loading.read_imn('data/test_milano_imns.json.gz')
    poi_data = read_poi_data('data/test_milano_imns_pois.json.gz')
    print(f"Loaded {len(imns)} IMNs and POI data for {len(poi_data)} users")
    
    # Filter to only include original users
    filtered_imns = {uid: imn for uid, imn in imns.items() if uid in original_user_ids}
    filtered_poi_data = {uid: data for uid, data in poi_data.items() if uid in original_user_ids}
    
    print(f"Filtered to {len(filtered_imns)} users from original dataset")
    
    # First pass: collect global statistics for interpolation
    global_duration_dist = None
    global_transition_dist = None
    global_trip_dist = None
    
    # Always collect global statistics for control_level interpolation
    global_duration_dist, global_transition_dist, global_trip_dist = collect_global_statistics(filtered_imns, filtered_poi_data)
    
    # Create global probability visualization if using global distributions
    if USE_GLOBAL_DISTRIBUTIONS:
        visualize_global_probabilities(global_duration_dist, global_transition_dist, global_trip_dist)
    
    # Second pass: process first 10 users with different control levels
    control_levels = [0.0, 0.25, 0.5, 0.75, 1.0]
    users_processed = 0
    
    for uid_str, imn in filtered_imns.items():
        if users_processed >= 10:  # Only process first 10 users
            break
            
        uid = uid_str  # Keep as string since that's how it's stored
        if uid not in filtered_poi_data:
            continue
            
        users_processed += 1
        print(f"Processing user {users_processed}/10: {uid}")
        
        # Enrich IMN with POI data
        enriched = enrich_imn_with_poi(imn, filtered_poi_data[uid])
        
        # Extract stays from trips
        stays = extract_stays_from_trips(enriched['trips'], enriched['locations'])
        
        # Group stays by day
        stays_by_day = extract_stays_by_day(stays)
        
        # Build user-specific distributions
        user_duration_probs, user_transition_probs, user_trip_duration_probs = build_stay_distributions(stays_by_day)
        
        # Create a single figure for this user showing all days and control levels
        print(f"  Creating comprehensive visualization for user {uid}")
        
        # Get sorted days for this user
        sorted_days = sorted(stays_by_day.keys())
        
        # Create figure with one subplot per day
        fig, axes = plt.subplots(len(sorted_days), 1, figsize=(16, 3 * len(sorted_days)), sharex=True)
        if len(sorted_days) == 1:
            axes = [axes]
        
        # Process each day
        for day_idx, day in enumerate(sorted_days):
            day_stays = stays_by_day[day]
            anchor_stay = find_anchor_stay_for_day(day_stays)
            
            if anchor_stay:
                ax = axes[day_idx]
                y_offset = 0.5
                
                # Plot original timeline for this day
                plot_stays(day_stays, y_offset=y_offset, ax=ax)
                y_offset += 0.3  # Space between timelines
                
                # Test different control levels for this day
                for control_level in control_levels:
                    print(f"    Testing control level {control_level} for user {uid}, day {day}")
                    
                    # Apply control_level to interpolate between user-specific and global distributions
                    if control_level == 0.0:
                        # Use only global distributions for fully randomized behavior
                        duration_probs = global_duration_dist
                        transition_probs = global_transition_dist
                        trip_duration_probs = global_trip_dist
                    elif global_duration_dist is not None and global_transition_dist is not None and global_trip_dist is not None:
                        # Interpolate distributions based on control_level
                        duration_probs = interpolate_distributions(user_duration_probs, global_duration_dist, control_level)
                        transition_probs = interpolate_distributions(user_transition_probs, global_transition_dist, control_level)
                        trip_duration_probs = interpolate_distributions(user_trip_duration_probs, global_trip_dist, control_level)
                    else:
                        # Use user-specific distributions if no global data available
                        duration_probs = user_duration_probs
                        transition_probs = user_transition_probs
                        trip_duration_probs = user_trip_duration_probs
                    
                    # Generate synthetic day with current control level
                    synthetic_stays = generate_synthetic_day(
                        anchor_stay,
                        duration_probs,
                        transition_probs,
                        trip_duration_probs,
                        day_stays,
                        control_level=control_level,
                    )
                    
                    # Save timeline comparison to text file
                    save_timeline_to_text(day_stays, synthetic_stays, uid, day, control_level, "timeline_text_outputs")
                    
                    # Plot synthetic timeline for this control level
                    plot_stays(synthetic_stays, y_offset=y_offset, ax=ax)
                    y_offset += 0.3  # Space between timelines
                
                # Customize subplot
                # Calculate y-positions for labels
                y_positions = [0.5 + i * 0.3 for i in range(6)]  # 6 = original + 5 control levels
                ax.set_yticks(y_positions)
                ax.set_yticklabels([f"Original - {day}"] + [f"Control {cl} - {day}" for cl in control_levels], fontsize=8)
                ax.set_ylim(0, 0.5 + 6 * 0.3)
                ax.set_title(f"User {uid} - {day}", fontsize=12)
                ax.grid(True, axis='x', linestyle=':', alpha=0.3)
        
        # Set x-axis formatting for the last subplot
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        axes[-1].set_xlabel("Time", fontsize=12)
        
        # Add legend
        legend_handles = [mpatches.Patch(color=c, label=l) for l, c in ACTIVITY_COLORS.items()]
        fig.legend(handles=legend_handles, title="Activity Types", 
                  loc='upper right', fontsize=10, title_fontsize=11)
        
        # Add title
        fig.suptitle(f"User {uid} - All Days and Control Levels", fontsize=14, y=0.95)
        
        plt.tight_layout(rect=[0, 0, 0.85, 0.95])
        
        # Save single figure for this user
        output_file = os.path.join(VISUALIZATION_DIR, f"user_{uid}_all_days_control_levels.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved comprehensive visualization for user {uid}")
    
    print(f"\nCompleted processing first 10 users with all control levels.")
    print(f"Visualizations saved in {VISUALIZATION_DIR}/")
    print(f"Probability visualizations saved in {PROBABILITY_VIS_DIR}/")
    print(f"Control levels tested: {control_levels}")
    

def main_original_timelines():
    """Create compact original timeline visualization for original Milano users."""
    print("\nCreating compact original timeline visualization...")
    visualize_original_timelines_compact(users_per_page=10)


if __name__ == "__main__":
    main() 