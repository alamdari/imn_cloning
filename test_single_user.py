#!/usr/bin/env python3
"""
Test script for processing a single user with detailed visualization.

This script processes one user at a time and generates comprehensive visualizations
including original trajectories in the source city and synthetic trajectories in Porto.

Usage:
    python test_single_user.py <user_id>
    python test_single_user.py 151443
"""

import os
import sys
import argparse
import pickle
from typing import Dict, List, Tuple, Any
import pytz

# Import required modules
from src.io.paths import PathsConfig, ensure_output_structure
from src.io.data import read_poi_data
from src.io import imn_loading
from src.population.utils import generate_cumulative_map
from src.spatial.resources import ensure_spatial_resources
from src.synthetic.enrich import enrich_imn_with_poi
from src.synthetic.stays import read_stays_from_trips, extract_stays_by_day
from src.synthetic.timelines import build_stay_distributions, prepare_day_data
from src.synthetic.spatial_sim import simulate_synthetic_trips
from src.spatial.mapping import map_imn_to_osm
from src.visualization.reports import user_probs_report
from src.visualization.timelines import save_user_timelines
from src.visualization.maps import generate_interactive_porto_map_multi, generate_interactive_original_city_map, create_split_map_html


# Configuration
RANDOMNESS_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
TIMEZONE = pytz.timezone("Europe/Rome")
CACHE_DIR = "results/cache"
CACHE_FILE = os.path.join(CACHE_DIR, "test_single_user_cache.pkl")


def process_user(user_id: int, imn: Dict, poi_info: Dict, randomness_levels: List[float], 
                 results_dir: str, tz, G, gdf_cumulative_p, activity_pools: Dict[str, List[int]]) -> None:
    """Process a single user with full pipeline."""
    print(f"\n{'=' * 60}")
    print(f"Processing user {user_id}")
    print(f"{'=' * 60}\n")
    
    # Create output directories
    os.makedirs(results_dir, exist_ok=True)
    prob_dir = os.path.join(results_dir, "probability_reports")
    vis_dir = os.path.join(results_dir, "timeline_visualizations")
    traj_dir = os.path.join(results_dir, "trajectories")
    os.makedirs(prob_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(traj_dir, exist_ok=True)
    
    # Debug: Inspect raw IMN structure
    print("0. Inspecting raw IMN structure...")
    print(f"   IMN keys: {list(imn.keys())}")
    print(f"   Home location ID: {imn.get('home')}")
    print(f"   Work location ID: {imn.get('work')}")
    print(f"   Number of locations: {len(imn.get('locations', {}))}")
    print(f"   Location IDs: {list(imn.get('locations', {}).keys())}")
    print(f"   Number of trips: {len(imn.get('trips', []))}")
    
    # Check if home/work exist in locations
    home_id = imn.get('home')
    work_id = imn.get('work')
    locations = imn.get('locations', {})
    
    if home_id not in locations:
        print(f"   ⚠ WARNING: Home location '{home_id}' not found in locations!")
    else:
        home_loc = locations[home_id]
        print(f"   ✓ Home location: {home_loc.get('coordinates')} (freq: {home_loc.get('frequency')})")
    
    if work_id not in locations:
        print(f"   ⚠ WARNING: Work location '{work_id}' not found in locations!")
    else:
        work_loc = locations[work_id]
        print(f"   ✓ Work location: {work_loc.get('coordinates')} (freq: {work_loc.get('frequency')})")
    
    # Enrich IMN with POI activity labels
    print("\n1. Enriching IMN with POI activity labels...")
    print(f"   POI info keys: {list(poi_info.keys())}")
    print(f"   POI classes: {poi_info.get('poi_classes', [])}")
    enriched = enrich_imn_with_poi(imn, poi_info)
    print(f"   ✓ Enriched IMN with {len(enriched['locations'])} locations")
    
    # Debug: Check enriched locations
    print(f"   Enriched location IDs: {list(enriched['locations'].keys())}")
    for loc_id, loc_data in list(enriched['locations'].items())[:5]:  # Show first 5
        print(f"     - Location {loc_id}: activity={loc_data.get('activity_label')}, coords={loc_data.get('coordinates')}")
    if len(enriched['locations']) > 5:
        print(f"     ... and {len(enriched['locations']) - 5} more locations")
    
    # Extract stays from trips
    print("\n2. Extracting stays from trips...")
    stays = read_stays_from_trips(enriched['trips'], enriched['locations'])
    stays_by_day = extract_stays_by_day(stays, tz)
    print(f"   ✓ Extracted {len(stays)} stays across {len(stays_by_day)} days")
    
    if not stays_by_day:
        print(f"   ⚠ No valid stays found for user {user_id}")
        return
    
    # Build stay distributions
    print("\n3. Building stay distributions...")
    user_duration_probs, user_transition_probs, user_trip_duration_probs = build_stay_distributions(stays_by_day)
    
    # Save probability reports
    user_probs_report(user_duration_probs, user_transition_probs, user_trip_duration_probs, user_id, prob_dir)
    print(f"   ✓ Saved probability reports to {prob_dir}")
    
    # Prepare day data with synthetic timelines
    print("\n4. Generating synthetic timelines...")
    day_data = prepare_day_data(stays_by_day, user_duration_probs, user_transition_probs, randomness_levels, tz)
    print(f"   ✓ Generated synthetic timelines for {len(day_data)} days")
    
    # Save timeline visualizations
    try:
        save_user_timelines(day_data, user_id, PathsConfig(results_dir=results_dir, vis_subdir="timeline_visualizations"))
        print(f"   ✓ Saved timeline visualizations to {vis_dir}")
    except Exception as e:
        print(f"   ⚠ Failed to save timeline visualization: {e}")
    
    # Map IMN to OSM
    print("\n5. Mapping IMN locations to Porto OSM nodes...")
    print(f"   Enriched home ID: {enriched.get('home')}")
    print(f"   Enriched work ID: {enriched.get('work')}")
    print(f"   Enriched locations count: {len(enriched.get('locations', {}))}")
    
    # Check if home/work are in enriched locations
    if enriched.get('home') not in enriched.get('locations', {}):
        print(f"   ⚠ ERROR: Home '{enriched.get('home')}' not in enriched locations!")
        print(f"   Available location IDs: {list(enriched.get('locations', {}).keys())}")
    
    if enriched.get('work') not in enriched.get('locations', {}):
        print(f"   ⚠ ERROR: Work '{enriched.get('work')}' not in enriched locations!")
    
    chosen_r = randomness_levels[2] if len(randomness_levels) > 2 else randomness_levels[0]
    
    try:
        map_loc_imn_user, rmse_user = map_imn_to_osm(
            enriched, 
            G, 
            gdf_cumulative_p=gdf_cumulative_p,
            activity_pools=activity_pools,
            random_seed=user_id
        )
        
        print(f"   ✓ Mapped {len(map_loc_imn_user)} locations with RMSE: {rmse_user:.2f} km")
        print(f"   Mapping result: {map_loc_imn_user}")
        
        fixed_home = map_loc_imn_user.get(enriched['home'])
        fixed_work = map_loc_imn_user.get(enriched['work'])
        
        print(f"   ✓ Home: {enriched['home']} → OSM node {fixed_home}")
        print(f"   ✓ Work: {enriched['work']} → OSM node {fixed_work}")
        
        if fixed_home is None:
            print(f"   ⚠ WARNING: Home was not mapped to any OSM node!")
        if fixed_work is None:
            print(f"   ⚠ WARNING: Work was not mapped to any OSM node!")
            
    except Exception as e:
        print(f"   ✗ ERROR in map_imn_to_osm: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Run spatial simulation for all days
    print("\n6. Running spatial simulation in Porto...")
    per_day_outputs: Dict[Any, Dict[str, Any]] = {}
    combined_traj: List[Tuple[int, float, float, int]] = []
    trajectory_id = 0
    
    for day_idx, (some_day, ddata) in enumerate(day_data.items(), 1):
        synthetic_for_r = ddata["synthetic"].get(chosen_r, [])
        
        traj, osm_usage, pseudo_map_loc, rmse, legs_coords = simulate_synthetic_trips(
            enriched,
            synthetic_for_r,
            G,
            gdf_cumulative_p,
            randomness=chosen_r,
            fixed_home_node=fixed_home,
            fixed_work_node=fixed_work,
            precomputed_map_loc_rmse=(map_loc_imn_user, rmse_user),
            activity_pools=activity_pools,
        )
        
        if traj is None:
            print(f"   ⚠ Spatial simulation failed for day {some_day}")
            continue
        
        trajectory_id += 1
        traj_with_id = [(trajectory_id, lat, lon, time) for lat, lon, time in traj]
        combined_traj.extend(traj_with_id)
        
        per_day_outputs[some_day] = {
            'trajectory': traj,
            'pseudo_map_loc': pseudo_map_loc,
            'synthetic_stays': synthetic_for_r,
            'legs_coords': legs_coords,
        }
        
        print(f"   ✓ Day {day_idx}/{len(day_data)}: {some_day} - {len(traj)} trajectory points")
    
    if not per_day_outputs:
        print("   ⚠ Spatial simulation failed for all days")
        return
    
    # Save trajectory CSV
    import pandas as pd
    traj_path = os.path.join(traj_dir, f"user_{user_id}_porto_trajectory.csv")
    pd.DataFrame(combined_traj, columns=["trajectory_id", "lat", "lon", "time"]).to_csv(traj_path, index=False)
    print(f"\n7. Saved Porto trajectory: {traj_path}")
    
    # Generate Porto map
    print("\n8. Generating interactive maps...")
    porto_map_path = os.path.join(traj_dir, f"user_{user_id}_porto_map.html")
    try:
        generate_interactive_porto_map_multi(user_id, per_day_outputs, G, porto_map_path)
        print(f"   ✓ Porto map: {porto_map_path}")
    except Exception as e:
        print(f"   ⚠ Failed to create Porto map: {e}")
    
    # Generate original city map with trajectories
    try:
        orig_map_path = os.path.join(traj_dir, f"user_{user_id}_original_city_map.html")
        generate_interactive_original_city_map(user_id, enriched, stays_by_day, orig_map_path, tz=tz)
        print(f"   ✓ Original city map: {orig_map_path}")
        
        # Create split view
        split_path = os.path.join(traj_dir, f"user_{user_id}_split_map.html")
        left_rel = os.path.basename(orig_map_path)
        right_rel = os.path.basename(porto_map_path)
        create_split_map_html("Original IMN (Milano)", left_rel, "Simulated Trajectory (Porto)", right_rel, split_path)
        print(f"   ✓ Split map: {split_path}")
    except Exception as e:
        print(f"   ⚠ Failed to create original city map: {e}")
    
    print(f"\n{'=' * 60}")
    print(f"✓ Processing complete for user {user_id}")
    print(f"{'=' * 60}")
    print(f"Results saved to: {results_dir}")


def load_cached_data(force_reload: bool = False):
    """Load and cache IMN data, POI data, and spatial resources."""
    import pickle
    
    # Check if cache exists and is valid
    if os.path.exists(CACHE_FILE) and not force_reload:
        print("\n1. Loading cached data...")
        try:
            with open(CACHE_FILE, 'rb') as f:
                cached_data = pickle.load(f)
            print(f"   ✓ Loaded from cache: {CACHE_FILE}")
            print(f"   ✓ IMNs: {len(cached_data['full_imns'])} users")
            print(f"   ✓ POI data: {len(cached_data['poi_data_all'])} users")
            print(f"   ✓ OSM graph: {len(cached_data['G'].nodes())} nodes, {len(cached_data['G'].edges())} edges")
            print(f"   ✓ Activity pools: {len(cached_data['activity_pools'])} activities")
            return cached_data
        except Exception as e:
            print(f"   ⚠ Failed to load cache: {e}")
            print(f"   → Loading fresh data...")
    
    # Load fresh data
    print("\n1. Loading IMN and POI data...")
    full_imns = imn_loading.read_imn("data/milano_2007_full_imns.json.gz")
    print(f"   ✓ Loaded {len(full_imns)} IMNs total")
    
    poi_data_all = read_poi_data("data/milano_2007_full_imns_pois.json.gz")
    print(f"   ✓ Loaded POI data for {len(poi_data_all)} users")
    
    # Load spatial resources
    print("\n2. Loading spatial resources (Porto OSM + population + activity pools)...")
    G, gdf_cumulative_p, activity_pools = ensure_spatial_resources("data", generate_cumulative_map)
    print(f"   ✓ Spatial resources ready")
    print(f"   ✓ OSM graph has {len(G.nodes())} nodes and {len(G.edges())} edges")
    print(f"   ✓ Activity pools available for: {len(activity_pools)} activities")
    
    # Cache the data
    cached_data = {
        'full_imns': full_imns,
        'poi_data_all': poi_data_all,
        'G': G,
        'gdf_cumulative_p': gdf_cumulative_p,
        'activity_pools': activity_pools
    }
    
    print(f"\n   Saving to cache: {CACHE_FILE}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    try:
        with open(CACHE_FILE, 'wb') as f:
            pickle.dump(cached_data, f)
        print(f"   ✓ Cache saved successfully")
    except Exception as e:
        print(f"   ⚠ Failed to save cache: {e}")
    
    return cached_data


def main():
    """Main execution."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Process a single user with detailed debugging',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python test_single_user.py 151443
  python test_single_user.py 302397
  python test_single_user.py 151443 --force-reload  # Reload data from scratch
        """
    )
    parser.add_argument('user_id', type=int, help='User ID to process')
    parser.add_argument('--results-dir', type=str, default=None, 
                       help='Results directory (default: results/single_user_<USER_ID>)')
    parser.add_argument('--force-reload', action='store_true',
                       help='Force reload data from source files (ignore cache)')
    
    args = parser.parse_args()
    user_id = args.user_id
    results_dir = args.results_dir if args.results_dir else f"results/single_user_{user_id}"
    
    print("=" * 60)
    print(f"Single User Processing: User {user_id}")
    print("=" * 60)
    
    # Load data (cached or fresh)
    cached_data = load_cached_data(force_reload=args.force_reload)
    
    full_imns = cached_data['full_imns']
    poi_data_all = cached_data['poi_data_all']
    G = cached_data['G']
    gdf_cumulative_p = cached_data['gdf_cumulative_p']
    activity_pools = cached_data['activity_pools']
    
    # Verify user exists
    print(f"\n3. Verifying user {user_id}...")
    if user_id not in full_imns:
        print(f"   ✗ User {user_id} not found in IMN data")
        print(f"   Available user IDs (first 10): {sorted(full_imns.keys())[:10]}")
        sys.exit(1)
    
    if user_id not in poi_data_all:
        print(f"   ✗ User {user_id} not found in POI data")
        print(f"   Available user IDs (first 10): {sorted(poi_data_all.keys())[:10]}")
        sys.exit(1)
    
    print(f"   ✓ Found user {user_id} in both datasets")
    
    # Process the user
    process_user(
        user_id,
        full_imns[user_id],
        poi_data_all[user_id],
        RANDOMNESS_LEVELS,
        results_dir,
        TIMEZONE,
        G,
        gdf_cumulative_p,
        activity_pools
    )
    
    print(f"\n{'=' * 60}")
    print("All done!")
    print(f"{'=' * 60}")
    print(f"Results saved to: {results_dir}")


if __name__ == "__main__":
    main()

