#!/usr/bin/env python3
"""
Diagnostic script to investigate user 7444's first day issue.

The problem: Original trajectories show a trip from home to work on 2007-04-02,
but the extracted stays only show work, skipping the home part.
"""

import os
import sys
import json
import gzip
from datetime import datetime
from typing import Dict, List, Tuple
import pytz

# Import required modules
from src.io import imn_loading
from src.io.data import read_poi_data
from src.synthetic.enrich import enrich_imn_with_poi
from src.synthetic.stays import read_stays_from_trips, extract_stays_by_day, Stay
from src.visualization.maps import load_original_trajectories, organize_trajectories_by_day


def format_timestamp(ts: int, tz) -> str:
    """Format Unix timestamp to readable string."""
    dt = datetime.fromtimestamp(ts, tz)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def print_trip(trip: Tuple, locations: Dict, tz, index: int = None):
    """Print a trip in readable format."""
    from_id, to_id, st, et = trip
    from_label = locations.get(from_id, {}).get('activity_label', 'unknown')
    to_label = locations.get(to_id, {}).get('activity_label', 'unknown')
    prefix = f"Trip {index}: " if index is not None else "Trip: "
    print(f"  {prefix}{from_label} ({from_id}) → {to_label} ({to_id})")
    print(f"    Time: {format_timestamp(st, tz)} → {format_timestamp(et, tz)}")
    print(f"    Duration: {(et - st) / 60:.1f} minutes")


def print_stay(stay: Stay, locations: Dict, tz, index: int = None):
    """Print a stay in readable format."""
    loc_label = locations.get(stay.location_id, {}).get('activity_label', 'unknown')
    prefix = f"Stay {index}: " if index is not None else "Stay: "
    print(f"  {prefix}{loc_label} (location_id: {stay.location_id})")
    if stay.start_time and stay.end_time:
        print(f"    Time: {format_timestamp(stay.start_time, tz)} → {format_timestamp(stay.end_time, tz)}")
        print(f"    Duration: {(stay.end_time - stay.start_time) / 60:.1f} minutes")
    else:
        print(f"    Time: {format_timestamp(stay.start_time, tz) if stay.start_time else 'None'} → {format_timestamp(stay.end_time, tz) if stay.end_time else 'None'}")


def analyze_original_trajectories(user_id: int, trajectories_dir: str, tz):
    """Analyze original GPS trajectories."""
    print("\n" + "="*60)
    print("ORIGINAL GPS TRAJECTORIES ANALYSIS")
    print("="*60)
    
    trajectories = load_original_trajectories(user_id, trajectories_dir)
    if not trajectories:
        print(f"  ⚠ No trajectories found for user {user_id}")
        return None
    
    print(f"\nFound {len(trajectories)} trajectories in JSON file")
    
    # Organize by day
    trajectories_by_day = organize_trajectories_by_day(trajectories, tz)
    print(f"\nTrajectories organized into {len(trajectories_by_day)} days:")
    
    for day_date in sorted(trajectories_by_day.keys()):
        day_trajs = trajectories_by_day[day_date]
        print(f"\n  Day {day_date}: {len(day_trajs)} trajectories")
        for traj_id, traj_data in day_trajs[:5]:  # Show first 5
            start_time = traj_data.get('_start_time', 0)
            points = traj_data.get('object', [])
            if points:
                first_point = points[0]
                last_point = points[-1]
                print(f"    Trajectory {traj_id}: {len(points)} points")
                print(f"      Start: {format_timestamp(start_time, tz)} at ({first_point[1]:.6f}, {first_point[0]:.6f})")
                print(f"      End: {format_timestamp(last_point[2], tz)} at ({last_point[1]:.6f}, {last_point[0]:.6f})")
    
    return trajectories_by_day


def analyze_imn_trips(user_id: int, imn: Dict, tz):
    """Analyze IMN trips."""
    print("\n" + "="*60)
    print("IMN TRIPS ANALYSIS")
    print("="*60)
    
    trips = imn.get('trips', [])
    locations = imn.get('locations', {})
    home_id = imn.get('home')
    work_id = imn.get('work')
    
    print(f"\nTotal trips: {len(trips)}")
    print(f"Total locations: {len(locations)}")
    print(f"Home location ID: {home_id}")
    print(f"Work location ID: {work_id}")
    
    if trips:
        print(f"\nFirst 10 trips:")
        for i, trip in enumerate(trips[:10], 1):
            print_trip(trip, locations, tz, index=i)
    
    # Group trips by day
    trips_by_day = {}
    for trip in trips:
        from_id, to_id, st, et = trip
        start_dt = datetime.fromtimestamp(st, tz)
        day_date = start_dt.date()
        if day_date not in trips_by_day:
            trips_by_day[day_date] = []
        trips_by_day[day_date].append(trip)
    
    print(f"\nTrips organized into {len(trips_by_day)} days:")
    for day_date in sorted(trips_by_day.keys()):
        day_trips = trips_by_day[day_date]
        print(f"  {day_date}: {len(day_trips)} trips")
        if day_date == datetime(2007, 4, 2).date():
            print(f"    ⚠ FIRST DAY - Showing all trips:")
            for i, trip in enumerate(day_trips, 1):
                print_trip(trip, locations, tz, index=i)
    
    return trips_by_day


def analyze_extracted_stays(user_id: int, enriched_imn: Dict, tz):
    """Analyze extracted stays from trips."""
    print("\n" + "="*60)
    print("EXTRACTED STAYS ANALYSIS")
    print("="*60)
    
    from src.synthetic.stays import read_stays_from_trips, extract_stays_by_day
    
    trips = enriched_imn.get('trips', [])
    locations = enriched_imn.get('locations', {})
    
    print(f"\nExtracting stays from {len(trips)} trips...")
    stays = read_stays_from_trips(trips, locations)
    print(f"Extracted {len(stays)} stays")
    
    if stays:
        print(f"\nFirst 10 stays:")
        for i, stay in enumerate(stays[:10], 1):
            print_stay(stay, locations, tz, index=i)
    
    # Extract by day
    stays_by_day = extract_stays_by_day(stays, tz)
    print(f"\nStays organized into {len(stays_by_day)} days:")
    
    for day_date in sorted(stays_by_day.keys()):
        day_stays = stays_by_day[day_date]
        print(f"\n  {day_date}: {len(day_stays)} stays")
        if day_date == datetime(2007, 4, 2).date():
            print(f"    ⚠ FIRST DAY - Showing all stays:")
            for i, stay in enumerate(day_stays, 1):
                print_stay(stay, locations, tz, index=i)
    
    return stays_by_day


def analyze_synthetic_timeline(stays_by_day: Dict, tz):
    """Analyze synthetic timeline for first day."""
    print("\n" + "="*60)
    print("SYNTHETIC TIMELINE ANALYSIS (First Day)")
    print("="*60)
    
    from src.synthetic.timelines import build_stay_distributions, prepare_day_data
    
    first_day = datetime(2007, 4, 2).date()
    
    if first_day not in stays_by_day:
        print(f"  ⚠ No stays found for {first_day}")
        return
    
    original_stays = stays_by_day[first_day]
    print(f"\nOriginal stays for {first_day}: {len(original_stays)}")
    for i, stay in enumerate(original_stays, 1):
        print_stay(stay, {}, tz, index=i)
    
    # Build distributions
    user_duration_probs, user_transition_probs, user_trip_duration_probs = build_stay_distributions(stays_by_day)
    
    # Generate synthetic day
    from src.synthetic.timelines import generate_synthetic_day
    RANDOMNESS_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
    chosen_r = 0.5
    
    print(f"\nGenerating synthetic timeline with randomness={chosen_r}...")
    synthetic_stays = generate_synthetic_day(
        original_stays, 
        user_duration_probs, 
        user_transition_probs, 
        randomness=chosen_r,
        day_length=24 * 3600,
        tz=tz
    )
    
    print(f"\nSynthetic stays for {first_day}: {len(synthetic_stays)}")
    if synthetic_stays:
        for i, (act, st, et) in enumerate(synthetic_stays, 1):
            print(f"  Stay {i}: {act}")
            print(f"    Time: {st//3600:02d}:{(st%3600)//60:02d} → {et//3600:02d}:{(et%3600)//60:02d} (relative seconds)")
            print(f"    Duration: {(et - st) / 60:.1f} minutes")
    else:
        print("  ⚠ No synthetic stays generated!")


def main():
    user_id = 7444
    imn_path = "data/milano_2007_full_imns.json.gz"
    poi_path = "data/milano_2007_full_imns_pois.json.gz"
    trajectories_dir = "data/trajectories"
    tz = pytz.timezone("Europe/Rome")
    
    print("="*60)
    print(f"DIAGNOSTIC ANALYSIS FOR USER {user_id}")
    print("="*60)
    print(f"Investigating first day (2007-04-02) issue")
    print(f"Problem: Original trajectories show home→work trip,")
    print(f"         but extracted stays only show work (skipping home)")
    print("="*60)
    
    # Load IMN
    print("\nLoading IMN data...")
    try:
        imns = imn_loading.read_imn(imn_path)
        if user_id not in imns:
            print(f"  ⚠ User {user_id} not found in IMN file")
            return
        imn = imns[user_id]
        print(f"  ✓ Loaded IMN for user {user_id}")
    except Exception as e:
        print(f"  ⚠ Error loading IMN: {e}")
        return
    
    # Load POI
    print("\nLoading POI data...")
    try:
        poi_data_all = read_poi_data(poi_path)
        poi_info = poi_data_all.get(user_id, {})
        print(f"  ✓ Loaded POI for user {user_id}")
    except Exception as e:
        print(f"  ⚠ Error loading POI: {e}")
        return
    
    # Enrich IMN
    print("\nEnriching IMN with POI...")
    enriched_imn = enrich_imn_with_poi(imn, poi_info)
    print(f"  ✓ Enriched IMN")
    
    # Analyze original trajectories
    trajectories_by_day = analyze_original_trajectories(user_id, trajectories_dir, tz)
    
    # Analyze IMN trips
    trips_by_day = analyze_imn_trips(user_id, imn, tz)
    
    # Analyze extracted stays
    stays_by_day = analyze_extracted_stays(user_id, enriched_imn, tz)
    
    # Analyze synthetic timeline
    analyze_synthetic_timeline(stays_by_day, tz)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    first_day = datetime(2007, 4, 2).date()
    
    print(f"\nFirst Day ({first_day}):")
    
    if trajectories_by_day and first_day in trajectories_by_day:
        print(f"  Original GPS trajectories: {len(trajectories_by_day[first_day])} trajectories")
    else:
        print(f"  Original GPS trajectories: Not found")
    
    if trips_by_day and first_day in trips_by_day:
        first_trip = trips_by_day[first_day][0] if trips_by_day[first_day] else None
        if first_trip:
            from_id, to_id, st, et = first_trip
            from_label = enriched_imn['locations'].get(from_id, {}).get('activity_label', 'unknown')
            to_label = enriched_imn['locations'].get(to_id, {}).get('activity_label', 'unknown')
            print(f"  First IMN trip: {from_label} → {to_label}")
        print(f"  Total IMN trips: {len(trips_by_day[first_day])}")
    else:
        print(f"  IMN trips: Not found")
    
    if stays_by_day and first_day in stays_by_day:
        print(f"  Extracted stays: {len(stays_by_day[first_day])}")
        if stays_by_day[first_day]:
            first_stay = stays_by_day[first_day][0]
            first_stay_label = enriched_imn['locations'].get(first_stay.location_id, {}).get('activity_label', 'unknown')
            print(f"  First stay location: {first_stay_label} (location_id: {first_stay.location_id})")
    else:
        print(f"  Extracted stays: Not found")
    
    print("\n" + "="*60)
    print("DIAGNOSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()


