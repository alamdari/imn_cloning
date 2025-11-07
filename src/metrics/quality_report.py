"""
Main orchestrator for generating comprehensive quality evaluation reports.
"""

import os
import json
import gzip
import pandas as pd
from typing import Dict, List
from pathlib import Path

from .trajectory_stats import compute_trajectory_statistics
from .spatial_stats import (
    compute_od_matrix, 
    compute_origin_density, 
    compute_destination_density,
    compute_spatial_coverage
)
from .visualizations import (
    plot_trip_duration_distribution,
    plot_trip_length_distribution,
    plot_trip_length_boxplot,
    plot_temporal_distribution,
    plot_day_of_week_distribution,
    plot_path_vs_od_distance,
    plot_origin_density_interactive_map,
    plot_od_pairs_interactive_map,
    plot_distribution_comparison,
)


def load_trajectory_from_json(json_path: Path) -> pd.DataFrame:
    """
    Load trajectory from JSON file and convert to DataFrame (optimized).
    
    JSON format:
    {
      "uid": 123,
      "num_trajectories": 45,
      "trajectories": {
        "0": {"object": [[lon, lat, time], ...], ...},
        "1": {"object": [[lon, lat, time], ...], ...},
        ...
      }
    }
    
    Returns:
        DataFrame with columns: trajectory_id, lat, lon, time
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Pre-allocate lists for better performance
    traj_ids = []
    lats = []
    lons = []
    times = []
    
    for traj_id_str, traj_obj in data['trajectories'].items():
        # traj_obj['object'] is list of [lon, lat, timestamp]
        if 'object' in traj_obj:
            points = traj_obj['object']
            num_points = len(points)
            traj_id = int(traj_id_str)
            
            # Batch append for efficiency
            traj_ids.extend([traj_id] * num_points)
            for point in points:
                lons.append(point[0])
                lats.append(point[1])
                times.append(point[2])
    
    # Create DataFrame in one go (much faster)
    return pd.DataFrame({
        'trajectory_id': traj_ids,
        'lat': lats,
        'lon': lons,
        'time': times
    })


def load_all_user_trajectories(trajectories_dir: str, limit_users: int = None) -> Dict[int, pd.DataFrame]:
    """
    Load all user trajectory files from a directory.
    Supports both CSV and JSON formats.
    
    Args:
        trajectories_dir: Path to directory containing trajectory files
        limit_users: Optional limit on number of users to load (for large datasets)
        
    Returns:
        Dictionary mapping user_id to trajectory DataFrame
    """
    user_trajectories = {}
    
    # Try CSV files first (synthetic trajectories)
    csv_files = list(Path(trajectories_dir).glob("user_*_porto_trajectory.csv"))
    if not csv_files:
        csv_files = list(Path(trajectories_dir).glob("user_*_trajectory.csv"))
    
    # Try JSON files (original segmented trajectories)
    json_files = list(Path(trajectories_dir).glob("user_*_trajectories.json"))
    
    if not csv_files and not json_files:
        print(f"⚠ No trajectory files (CSV or JSON) found in {trajectories_dir}")
        return user_trajectories
    
    # Apply limit if specified
    if limit_users is not None:
        if csv_files and len(csv_files) > limit_users:
            csv_files = csv_files[:limit_users]
            print(f"  → Limiting to first {limit_users} CSV files")
        if json_files and len(json_files) > limit_users:
            json_files = json_files[:limit_users]
            print(f"  → Limiting to first {limit_users} JSON files")
    
    # Load CSV files
    if csv_files:
        print(f"Loading {len(csv_files)} CSV trajectory files...")
        for csv_path in csv_files:
            filename = csv_path.stem
            try:
                user_id = int(filename.split('_')[1])
            except (IndexError, ValueError):
                print(f"  ⚠ Could not extract user_id from {filename}, skipping")
                continue
            
            try:
                df = pd.read_csv(csv_path)
                required_cols = ['trajectory_id', 'lat', 'lon', 'time']
                if not all(col in df.columns for col in required_cols):
                    print(f"  ⚠ Missing required columns in {filename}, skipping")
                    continue
                user_trajectories[user_id] = df
            except Exception as e:
                print(f"  ⚠ Error loading {filename}: {e}")
                continue
        print(f"✓ Loaded {len(user_trajectories)} CSV trajectories")
    
    # Load JSON files
    if json_files:
        print(f"Loading {len(json_files)} JSON trajectory files...")
        loaded_count = 0
        for idx, json_path in enumerate(json_files, 1):
            if idx % 50 == 0 or idx == len(json_files):
                print(f"  Progress: {idx}/{len(json_files)} files...")
            
            filename = json_path.stem
            try:
                user_id = int(filename.split('_')[1])
            except (IndexError, ValueError):
                continue
            
            try:
                df = load_trajectory_from_json(json_path)
                if len(df) > 0:
                    user_trajectories[user_id] = df
                    loaded_count += 1
            except Exception as e:
                if idx <= 5:  # Only show first few errors
                    print(f"  ⚠ Error loading {filename}: {e}")
                continue
        print(f"✓ Loaded {loaded_count} user trajectories from JSON files")
    
    return user_trajectories


def generate_quality_report(trajectories_dir: str, output_dir: str, grid_size_m: float = 500.0, 
                           original_trajectories_dir: str = None):
    """
    Generate comprehensive quality evaluation report for synthetic trajectories.
    
    Args:
        trajectories_dir: Path to directory with trajectory CSV files
        output_dir: Path to save metrics and plots
        grid_size_m: Grid cell size for spatial aggregation (meters)
        original_trajectories_dir: Optional path to original trajectories for comparison
    """
    print("\n" + "="*60)
    print("SYNTHETIC TRAJECTORY QUALITY EVALUATION")
    print("="*60 + "\n")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all user trajectories
    user_trajectories = load_all_user_trajectories(trajectories_dir)
    
    if not user_trajectories:
        print("No trajectories to process. Exiting.")
        return
    
    print(f"\nProcessing {len(user_trajectories)} users...")
    
    # Compute statistics for each user
    all_user_stats = {}
    all_trajectories = []
    
    # Check if we need to convert relative timestamps
    first_user_df = next(iter(user_trajectories.values()))
    if 'time' in first_user_df.columns:
        max_time = first_user_df['time'].max()
        if max_time < 100000 and 'day_date' in first_user_df.columns:
            print(f"  → Detected relative timestamps, converting using day_date column...")
    
    for user_id, traj_df in user_trajectories.items():
        stats = compute_trajectory_statistics(traj_df)
        stats['user_id'] = user_id
        all_user_stats[user_id] = stats
        all_trajectories.append(stats)
    
    # Combine all statistics
    combined_stats = pd.concat(all_trajectories, ignore_index=True)
    
    print(f"✓ Computed statistics for {len(combined_stats)} trajectories across {len(user_trajectories)} users\n")
    
    # Save combined statistics
    stats_path = os.path.join(output_dir, "trajectory_statistics.csv")
    combined_stats.to_csv(stats_path, index=False)
    print(f"✓ Saved trajectory statistics: {os.path.basename(stats_path)}\n")
    
    # Compute spatial metrics
    print("Computing spatial metrics...")
    od_matrix = compute_od_matrix(combined_stats, grid_size_m)
    origin_density = compute_origin_density(combined_stats, grid_size_m)
    dest_density = compute_destination_density(combined_stats, grid_size_m)
    spatial_coverage = compute_spatial_coverage(combined_stats)
    
    # Save spatial metrics
    od_matrix.to_csv(os.path.join(output_dir, "od_matrix.csv"), index=False)
    origin_density.to_csv(os.path.join(output_dir, "origin_density.csv"), index=False)
    dest_density.to_csv(os.path.join(output_dir, "destination_density.csv"), index=False)
    
    with open(os.path.join(output_dir, "spatial_coverage.json"), 'w') as f:
        json.dump(spatial_coverage, f, indent=2)
    
    print(f"✓ Saved OD matrix ({len(od_matrix)} pairs)")
    print(f"✓ Saved origin density ({len(origin_density)} cells)")
    print(f"✓ Saved destination density ({len(dest_density)} cells)")
    print(f"✓ Saved spatial coverage\n")
    
    # Generate summary statistics
    summary = {
        'total_users': len(user_trajectories),
        'total_trajectories': len(combined_stats),
        'avg_trajectories_per_user': len(combined_stats) / len(user_trajectories),
        'duration_statistics': {
            'mean_seconds': float(combined_stats['duration_seconds'].mean()),
            'median_seconds': float(combined_stats['duration_seconds'].median()),
            'std_seconds': float(combined_stats['duration_seconds'].std()),
            'min_seconds': float(combined_stats['duration_seconds'].min()),
            'max_seconds': float(combined_stats['duration_seconds'].max()),
        },
        'length_statistics': {
            'mean_meters': float(combined_stats['od_distance_meters'].mean()),
            'median_meters': float(combined_stats['od_distance_meters'].median()),
            'std_meters': float(combined_stats['od_distance_meters'].std()),
            'min_meters': float(combined_stats['od_distance_meters'].min()),
            'max_meters': float(combined_stats['od_distance_meters'].max()),
        },
        'path_length_statistics': {
            'mean_meters': float(combined_stats['path_length_meters'].mean()),
            'median_meters': float(combined_stats['path_length_meters'].median()),
        },
        'spatial_coverage': spatial_coverage,
        'grid_size_meters': grid_size_m,
        'unique_origin_cells': len(origin_density),
        'unique_destination_cells': len(dest_density),
        'unique_od_pairs': len(od_matrix),
    }
    
    summary_path = os.path.join(output_dir, "quality_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"✓ Saved quality summary: {os.path.basename(summary_path)}\n")
    
    # Load and compare with original data if provided
    original_stats = None
    if original_trajectories_dir:
        print(f"\nLoading original trajectories from: {original_trajectories_dir}")
        print(f"  → Only loading users that exist in synthetic data (matching {len(user_trajectories)} users)")
        try:
            # Load only users that exist in synthetic trajectories (for fair comparison)
            original_user_trajs = {}
            synthetic_user_ids = set(user_trajectories.keys())
            
            # Check for JSON files matching synthetic users
            json_count = 0
            csv_count = 0
            for user_id in synthetic_user_ids:
                # Try JSON first
                json_path = Path(original_trajectories_dir) / f"user_{user_id}_trajectories.json"
                csv_path_porto = Path(original_trajectories_dir) / f"user_{user_id}_porto_trajectory.csv"
                csv_path = Path(original_trajectories_dir) / f"user_{user_id}_trajectory.csv"
                
                if json_path.exists():
                    try:
                        df = load_trajectory_from_json(json_path)
                        if len(df) > 0:
                            original_user_trajs[user_id] = df
                            json_count += 1
                    except Exception as e:
                        print(f"  ⚠ Error loading user {user_id}: {e}")
                elif csv_path_porto.exists():
                    try:
                        df = pd.read_csv(csv_path_porto)
                        if all(col in df.columns for col in ['trajectory_id', 'lat', 'lon', 'time']):
                            original_user_trajs[user_id] = df
                            csv_count += 1
                    except Exception as e:
                        print(f"  ⚠ Error loading user {user_id}: {e}")
                elif csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        if all(col in df.columns for col in ['trajectory_id', 'lat', 'lon', 'time']):
                            original_user_trajs[user_id] = df
                            csv_count += 1
                    except Exception as e:
                        print(f"  ⚠ Error loading user {user_id}: {e}")
            
            print(f"✓ Loaded {len(original_user_trajs)} matching users ({json_count} JSON, {csv_count} CSV)")
            
            if original_user_trajs:
                print("Computing original trajectory statistics...")
                original_traj_list = []
                for user_id, traj_df in original_user_trajs.items():
                    stats = compute_trajectory_statistics(traj_df)
                    stats['user_id'] = user_id
                    original_traj_list.append(stats)
                original_stats = pd.concat(original_traj_list, ignore_index=True)
                print(f"✓ Computed statistics for {len(original_stats)} original trajectories")
                
                # Save original statistics
                orig_stats_path = os.path.join(output_dir, "original_trajectory_statistics.csv")
                original_stats.to_csv(orig_stats_path, index=False)
                print(f"✓ Saved original statistics: {os.path.basename(orig_stats_path)}")
        except Exception as e:
            print(f"⚠ Could not load original trajectories: {e}")
    
    # Generate visualizations (all as comparisons if original data available)
    print("\nGenerating visualizations...")
    
    if original_stats is not None:
        print("  Generating comparison plots (Original vs Synthetic)...")
        plot_distribution_comparison(combined_stats, original_stats, output_dir)
    else:
        print("  Generating synthetic-only plots (no original data for comparison)...")
        plot_trip_duration_distribution(
            combined_stats, 
            os.path.join(output_dir, "trip_duration_distribution.png")
        )
        
        plot_trip_length_distribution(
            combined_stats,
            os.path.join(output_dir, "trip_length_distribution.png")
        )
        
        plot_temporal_distribution(
            combined_stats,
            os.path.join(output_dir, "temporal_distribution.png")
        )
    
    # These plots are always synthetic-only (no original comparison needed)
    plot_trip_length_boxplot(
        all_user_stats,
        os.path.join(output_dir, "trip_length_boxplot.png")
    )
    
    plot_day_of_week_distribution(
        combined_stats,
        os.path.join(output_dir, "day_of_week_distribution.png")
    )
    
    plot_path_vs_od_distance(
        combined_stats,
        os.path.join(output_dir, "path_vs_od_distance.png")
    )
    
    # Generate interactive maps (synthetic only - spatial)
    print("\nGenerating interactive maps...")
    
    plot_origin_density_interactive_map(
        combined_stats,
        os.path.join(output_dir, "origin_density_map.html"),
        grid_size_m=grid_size_m
    )
    
    plot_od_pairs_interactive_map(
        combined_stats,
        od_matrix,
        os.path.join(output_dir, "top_od_pairs_map.html"),
        top_n=20
    )
    
    print("\n" + "="*60)
    print("QUALITY EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nData Files:")
    print(f"  - trajectory_statistics.csv (synthetic)")
    if original_stats is not None:
        print(f"  - original_trajectory_statistics.csv")
    print(f"  - od_matrix.csv")
    print(f"  - origin_density.csv")
    print(f"  - destination_density.csv")
    print(f"  - spatial_coverage.json")
    print(f"  - quality_summary.json")
    
    print(f"\nVisualizations:")
    if original_stats is not None:
        print(f"  Comparison Plots (Original vs Synthetic):")
        print(f"    - duration_comparison.png")
        print(f"    - length_comparison.png")
        print(f"    - temporal_comparison.png")
    else:
        print(f"  Synthetic-only distributions:")
        print(f"    - trip_duration_distribution.png")
        print(f"    - trip_length_distribution.png")
        print(f"    - temporal_distribution.png")
    
    print(f"  Other plots:")
    print(f"    - trip_length_boxplot.png")
    print(f"    - day_of_week_distribution.png")
    print(f"    - path_vs_od_distance.png")
    
    print(f"\nInteractive Maps:")
    print(f"  - origin_density_map.html (heatmap + toggleable markers)")
    print(f"  - top_od_pairs_map.html (flow lines between origins/destinations)")
    
    if original_stats is None:
        print(f"\n💡 Tip: Run with --original data/original_trajectories for comparison plots!")
    print()

