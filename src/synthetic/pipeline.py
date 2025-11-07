import os
from typing import Dict, List, Tuple, Any
import pandas as pd

from src.io.paths import PathsConfig, ensure_output_structure
from src.io.data import load_datasets
from src.population.utils import generate_cumulative_map
from src.spatial.resources import ensure_spatial_resources
from src.synthetic.timelines import build_stay_distributions, prepare_day_data
from src.synthetic.spatial_sim import simulate_synthetic_trips
from src.synthetic.stays import read_stays_from_trips, extract_stays_by_day
from src.synthetic.enrich import enrich_imn_with_poi
from src.visualization.reports import user_probs_report
from src.visualization.timelines import save_user_timelines
from src.visualization.maps import generate_interactive_porto_map_multi, generate_interactive_original_city_map, create_split_map_html


def process_single_user(user_id: int, imn: Dict, poi_info: Dict, randomness_levels: List[float], paths: PathsConfig, tz, G, gdf_cumulative_p, activity_pools: Dict[str, List[int]]) -> None:
    print(f"Processing user {user_id}...")
    enriched = enrich_imn_with_poi(imn, poi_info)
    stays = read_stays_from_trips(enriched['trips'], enriched['locations'])
    stays_by_day = extract_stays_by_day(stays, tz)
    if not stays_by_day:
        print(f"  ⚠ No valid stays found for user {user_id}")
        return
    user_duration_probs, user_transition_probs, user_trip_duration_probs = build_stay_distributions(stays_by_day)
    user_probs_report(user_duration_probs, user_transition_probs, user_trip_duration_probs, user_id, paths.prob_dir())
    day_data = prepare_day_data(stays_by_day, user_duration_probs, user_transition_probs, randomness_levels, tz)

    # Save timeline visualization PNG per user via visualization module
    try:
        save_user_timelines(day_data, user_id, paths)
    except Exception as e:
        print(f"  ⚠ Failed to save timeline visualization for user {user_id}: {e}")

    print("  ↳ Running spatial simulation in Porto (synthetic stays, all days)...")
    chosen_r = randomness_levels[2] if len(randomness_levels) > 2 else randomness_levels[0]
    from src.spatial.mapping import map_imn_to_osm
    map_loc_imn_user, rmse_user = map_imn_to_osm(
        enriched, 
        G, 
        gdf_cumulative_p=gdf_cumulative_p,
        activity_pools=activity_pools,
        random_seed=user_id  # Use user_id as seed for deterministic but unique mapping
    )
    fixed_home = map_loc_imn_user.get(enriched['home'])
    fixed_work = map_loc_imn_user.get(enriched['work'])
    print(f"  ↳ RMSE for user {user_id}: {rmse_user:.2f} km")

    per_day_outputs: Dict[Any, Dict[str, Any]] = {}
    combined_traj: List[Tuple[int, float, float, int]] = []
    any_success = False
    trajectory_id = 0
    for some_day, ddata in day_data.items():
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
            print(f"  ⚠ Spatial simulation failed for user {user_id} on day {some_day}")
            continue
        any_success = True
        trajectory_id += 1
        # Add trajectory_id to each point (trajectory_id, lat, lon, time)
        traj_with_id = [(trajectory_id, lat, lon, time) for lat, lon, time in traj]
        combined_traj.extend(traj_with_id)
        per_day_outputs[some_day] = {
            'trajectory': traj,
            'pseudo_map_loc': pseudo_map_loc,
            'synthetic_stays': synthetic_for_r,
            'legs_coords': legs_coords,
        }

    if not any_success:
        print("  ⚠ Spatial simulation failed for all days for this user")
        return

    traj_path = os.path.join(paths.results_dir, "synthetic_trajectories", f"user_{user_id}_porto_trajectory.csv")
    os.makedirs(os.path.dirname(traj_path), exist_ok=True)
    pd.DataFrame(combined_traj, columns=["trajectory_id", "lat", "lon", "time"]).to_csv(traj_path, index=False)
    print(f"  ✓ Spatial trajectory saved (all days combined): {os.path.basename(traj_path)}")

    map_path = os.path.join(paths.results_dir, "synthetic_trajectories", f"user_{user_id}_porto_map.html")
    try:
        generate_interactive_porto_map_multi(user_id, per_day_outputs, G, map_path)
        print(f"  ✓ Interactive multi-day map saved: {os.path.basename(map_path)}")
    except Exception as e:
        print(f"  ⚠ Failed to create interactive multi-day map: {e}")

    # Also create original-city map based on IMN coordinates
    try:
        orig_map_path = os.path.join(paths.results_dir, "synthetic_trajectories", f"user_{user_id}_original_city_map.html")
        generate_interactive_original_city_map(user_id, enriched, stays_by_day, orig_map_path, tz=tz)
        print(f"  ✓ Original-city map saved: {os.path.basename(orig_map_path)}")
        # Create split view combining original-city and Porto maps
        try:
            split_path = os.path.join(paths.results_dir, "synthetic_trajectories", f"user_{user_id}_split_map.html")
            left_rel = os.path.basename(orig_map_path)
            right_rel = os.path.basename(map_path)
            create_split_map_html("Original IMN (source city)", left_rel, "Simulated Trajectory (Porto)", right_rel, split_path)
            print(f"  ✓ Split map saved: {os.path.basename(split_path)}")
        except Exception as e:
            print(f"  ⚠ Failed to create split map: {e}")
    except Exception as e:
        print(f"  ⚠ Failed to create original-city map: {e}")


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
        print(f"!! Error loading data: {e}")
        return

    try:
        print("Preparing spatial resources (Porto OSM + population TIFF + activity pools)...")
        G, gdf_cumulative_p, activity_pools = ensure_spatial_resources("data", generate_cumulative_map)
        print("✓ Spatial resources ready")
    except Exception as e:
        print(f"⚠ Spatial resources setup failed: {e}")
        G, gdf_cumulative_p, activity_pools = None, None, None

    print(f"\nProcessing {len(imns)} users...")
    print("-" * 40)
    for idx, uid in enumerate(imns.keys(), 1):
        print(f"[{idx}/{len(imns)}] ")
        try:
            process_single_user(uid, imns[uid], poi_data[uid], randomness_levels, paths, tz, G, gdf_cumulative_p, activity_pools)
        except Exception as e:
            print(f"!! Error processing user {uid}: {e}")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Results saved to: {paths.results_dir}")
    