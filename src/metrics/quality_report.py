"""
Main orchestrator for generating comprehensive quality evaluation reports.
"""

import os
import json
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
    plot_origin_density_heatmap,
    plot_od_flow_summary,
    plot_day_of_week_distribution,
    plot_path_vs_od_distance,
)


def load_all_user_trajectories(trajectories_dir: str) -> Dict[int, pd.DataFrame]:
    """
    Load all user trajectory CSV files from a directory.
    
    Args:
        trajectories_dir: Path to directory containing user_*_porto_trajectory.csv files
        
    Returns:
        Dictionary mapping user_id to trajectory DataFrame
    """
    user_trajectories = {}
    
    csv_files = list(Path(trajectories_dir).glob("user_*_porto_trajectory.csv"))
    
    if not csv_files:
        print(f"⚠ No trajectory CSV files found in {trajectories_dir}")
        return user_trajectories
    
    print(f"Loading {len(csv_files)} user trajectory files...")
    
    for csv_path in csv_files:
        # Extract user_id from filename: user_123_porto_trajectory.csv
        filename = csv_path.stem
        try:
            user_id = int(filename.split('_')[1])
        except (IndexError, ValueError):
            print(f"  ⚠ Could not extract user_id from {filename}, skipping")
            continue
        
        try:
            df = pd.read_csv(csv_path)
            
            # Validate required columns
            required_cols = ['trajectory_id', 'lat', 'lon', 'time']
            if not all(col in df.columns for col in required_cols):
                print(f"  ⚠ Missing required columns in {filename}, skipping")
                continue
            
            user_trajectories[user_id] = df
            
        except Exception as e:
            print(f"  ⚠ Error loading {filename}: {e}")
            continue
    
    print(f"✓ Loaded {len(user_trajectories)} user trajectories")
    return user_trajectories


def generate_quality_report(trajectories_dir: str, output_dir: str, grid_size_m: float = 500.0):
    """
    Generate comprehensive quality evaluation report for synthetic trajectories.
    
    Args:
        trajectories_dir: Path to directory with trajectory CSV files
        output_dir: Path to save metrics and plots
        grid_size_m: Grid cell size for spatial aggregation (meters)
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
    
    # Generate visualizations
    print("Generating visualizations...")
    
    plot_trip_duration_distribution(
        combined_stats, 
        os.path.join(output_dir, "trip_duration_distribution.png")
    )
    
    plot_trip_length_distribution(
        combined_stats,
        os.path.join(output_dir, "trip_length_distribution.png")
    )
    
    plot_trip_length_boxplot(
        all_user_stats,
        os.path.join(output_dir, "trip_length_boxplot.png")
    )
    
    plot_temporal_distribution(
        combined_stats,
        os.path.join(output_dir, "temporal_distribution.png")
    )
    
    plot_day_of_week_distribution(
        combined_stats,
        os.path.join(output_dir, "day_of_week_distribution.png")
    )
    
    plot_origin_density_heatmap(
        origin_density,
        os.path.join(output_dir, "origin_density_heatmap.png")
    )
    
    plot_od_flow_summary(
        od_matrix,
        os.path.join(output_dir, "top_od_pairs.png"),
        top_n=20
    )
    
    plot_path_vs_od_distance(
        combined_stats,
        os.path.join(output_dir, "path_vs_od_distance.png")
    )
    
    print("\n" + "="*60)
    print("QUALITY EVALUATION COMPLETE")
    print("="*60)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - trajectory_statistics.csv")
    print(f"  - od_matrix.csv")
    print(f"  - origin_density.csv")
    print(f"  - destination_density.csv")
    print(f"  - spatial_coverage.json")
    print(f"  - quality_summary.json")
    print(f"  - 8 visualization plots (PNG)")
    print()

