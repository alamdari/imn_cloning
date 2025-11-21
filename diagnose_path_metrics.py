#!/usr/bin/env python3
"""
Diagnostic script to compare path metrics between original and synthetic trajectories.

This script computes:
- actual_path_length: Sum of all segment distances in the trajectory
- shortest_path_length: Shortest path distance on OSM graph
- straight_line_distance: Haversine distance between origin and destination

And derived metrics:
- detour_ratio = actual_path / shortest_path
- efficiency = straight_line / actual_path

Creates side-by-side comparison plots for each metric.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
import matplotlib.pyplot as plt
import seaborn as sns

# Import existing utilities
from src.metrics.trajectory_stats import haversine_distance, compute_trip_path_length, compute_trip_length
from src.metrics.quality_report import load_trajectory_from_json, load_all_user_trajectories
from src.spatial.resources import ensure_spatial_resources
from src.population.utils import generate_cumulative_map


def compute_shortest_path_length(G, origin_lat: float, origin_lon: float, 
                                 dest_lat: float, dest_lon: float) -> Optional[float]:
    """
    Compute shortest path length on OSM graph between two GPS coordinates.
    
    Args:
        G: OSM graph
        origin_lat, origin_lon: Origin coordinates
        dest_lat, dest_lon: Destination coordinates
        
    Returns:
        Shortest path length in meters, or None if no path found
    """
    try:
        # Find nearest nodes
        origin_node = ox.distance.nearest_nodes(G, origin_lon, origin_lat)
        dest_node = ox.distance.nearest_nodes(G, dest_lon, dest_lat)
        
        if origin_node == dest_node:
            return 0.0
        
        # Compute shortest path by length (not travel_time)
        try:
            path = nx.shortest_path(G, origin_node, dest_node, weight='length')
        except nx.NetworkXNoPath:
            return None
        
        # Compute total path length using OSMnx route_to_gdf
        try:
            gdf_route = ox.routing.route_to_gdf(G, path)
            total_length = gdf_route['length'].sum()
        except Exception:
            # Fallback: compute manually
            total_length = 0.0
            for i in range(len(path) - 1):
                edge_data = G[path[i]][path[i+1]]
                # Handle multiple edges between nodes
                if isinstance(edge_data, dict):
                    length = edge_data.get('length', 0.0)
                else:
                    # Multiple edges - take first one
                    length = list(edge_data.values())[0].get('length', 0.0) if edge_data else 0.0
                total_length += length
        
        return total_length
    except Exception as e:
        return None


def compute_trip_metrics(df: pd.DataFrame, G, trajectory_type: str = "unknown") -> pd.DataFrame:
    """
    Compute path metrics for all trips in a trajectory DataFrame.
    
    Args:
        df: DataFrame with columns [trajectory_id, lat, lon, time]
        G: OSM graph for computing shortest paths
        trajectory_type: Label for the trajectory type ("original" or "synthetic")
        
    Returns:
        DataFrame with metrics for each trip
    """
    grouped = df.groupby('trajectory_id')
    metrics = []
    total_trips = len(grouped)
    
    print(f"  Computing metrics for {total_trips} {trajectory_type} trips...")
    
    for idx, (traj_id, group) in enumerate(grouped, 1):
        if idx % 100 == 0:
            print(f"    Progress: {idx}/{total_trips} trips processed...")
        if len(group) < 2:
            continue
        
        # Get origin and destination
        origin = group.iloc[0]
        dest = group.iloc[-1]
        origin_lat, origin_lon = origin['lat'], origin['lon']
        dest_lat, dest_lon = dest['lat'], dest['lon']
        
        # 1. Straight-line distance (OD distance)
        straight_line_dist = haversine_distance(origin_lat, origin_lon, dest_lat, dest_lon)
        
        # 2. Actual path length (sum of all segments)
        coords = group[['lat', 'lon']].values
        actual_path_length = 0.0
        for i in range(len(coords) - 1):
            actual_path_length += haversine_distance(
                coords[i][0], coords[i][1],
                coords[i+1][0], coords[i+1][1]
            )
        
        # 3. Shortest path length on OSM graph
        shortest_path_length = compute_shortest_path_length(
            G, origin_lat, origin_lon, dest_lat, dest_lon
        )
        
        if shortest_path_length is None:
            # Skip if we can't compute shortest path
            continue
        
        # Derived metrics
        detour_ratio = actual_path_length / shortest_path_length if shortest_path_length > 0 else np.nan
        efficiency = straight_line_dist / actual_path_length if actual_path_length > 0 else np.nan
        
        metrics.append({
            'trajectory_id': traj_id,
            'straight_line_distance': straight_line_dist,
            'actual_path_length': actual_path_length,
            'shortest_path_length': shortest_path_length,
            'detour_ratio': detour_ratio,
            'efficiency': efficiency,
        })
    
    return pd.DataFrame(metrics)


def plot_metric_comparison(original_metrics: pd.DataFrame, synthetic_metrics: pd.DataFrame,
                          metric_name: str, metric_label: str, unit: str, output_path: str):
    """
    Create side-by-side comparison plot for a single metric.
    
    Args:
        original_metrics: DataFrame with original trajectory metrics
        synthetic_metrics: DataFrame with synthetic trajectory metrics
        metric_name: Column name in the DataFrames
        metric_label: Display label for the metric
        unit: Unit for the metric (e.g., "meters", "ratio")
        output_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original distribution
    orig_data = original_metrics[metric_name].dropna()
    ax1.hist(orig_data, bins=50, edgecolor='black', alpha=0.7, color='blue')
    ax1.set_xlabel(f'{metric_label} ({unit})', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Original City - {metric_label}', fontsize=14, fontweight='bold')
    mean_val = orig_data.mean()
    ax1.axvline(mean_val, color='red', linestyle='--', 
                label=f'Mean: {mean_val:.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Synthetic distribution
    syn_data = synthetic_metrics[metric_name].dropna()
    ax2.hist(syn_data, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax2.set_xlabel(f'{metric_label} ({unit})', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Target City (Synthetic) - {metric_label}', fontsize=14, fontweight='bold')
    mean_val = syn_data.mean()
    ax2.axvline(mean_val, color='red', linestyle='--', 
                label=f'Mean: {mean_val:.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(output_path)}")


def main():
    parser = argparse.ArgumentParser(
        description='Diagnose path metrics differences between original and synthetic trajectories'
    )
    parser.add_argument(
        '--synthetic',
        type=str,
        required=True,
        help='Directory containing synthetic trajectory CSV files'
    )
    parser.add_argument(
        '--original',
        type=str,
        required=True,
        help='Directory containing original trajectory JSON or CSV files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='results/path_metrics_diagnosis',
        help='Output directory for plots and metrics (default: results/path_metrics_diagnosis)'
    )
    parser.add_argument(
        '--target-city',
        type=str,
        default='porto',
        choices=['porto', 'milan'],
        help='Target city for OSM graph (default: porto)'
    )
    parser.add_argument(
        '--limit-users',
        type=int,
        default=None,
        help='Limit number of users to process (for faster testing)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    print("=" * 60)
    print("PATH METRICS DIAGNOSIS")
    print("=" * 60)
    print(f"Synthetic trajectories: {args.synthetic}")
    print(f"Original trajectories: {args.original}")
    print(f"Output directory: {args.output}")
    print(f"Target city: {args.target_city}")
    print()
    
    # Load OSM graph for shortest path computation
    print("Loading OSM graph...")
    try:
        G, _, _ = ensure_spatial_resources("data", generate_cumulative_map, target_city=args.target_city)
        print("✓ OSM graph loaded")
    except Exception as e:
        print(f"⚠ Error loading OSM graph: {e}")
        print("  Continuing without shortest path computation...")
        G = None
    
    # Load trajectories
    print("\nLoading synthetic trajectories...")
    synthetic_trajs = load_all_user_trajectories(args.synthetic, limit_users=args.limit_users)
    print(f"✓ Loaded {len(synthetic_trajs)} synthetic users")
    
    print("\nLoading original trajectories...")
    original_trajs = load_all_user_trajectories(args.original, limit_users=args.limit_users)
    print(f"✓ Loaded {len(original_trajs)} original users")
    
    if not synthetic_trajs or not original_trajs:
        print("⚠ Error: Need both synthetic and original trajectories")
        return
    
    # Find common users
    common_users = set(synthetic_trajs.keys()) & set(original_trajs.keys())
    print(f"\nFound {len(common_users)} common users")
    
    if len(common_users) == 0:
        print("⚠ Error: No common users found between synthetic and original trajectories")
        return
    
    # Combine all trajectories
    print("\nCombining trajectories...")
    synthetic_df = pd.concat([df.assign(user_id=uid) for uid, df in synthetic_trajs.items() 
                             if uid in common_users], ignore_index=True)
    original_df = pd.concat([df.assign(user_id=uid) for uid, df in original_trajs.items() 
                            if uid in common_users], ignore_index=True)
    
    print(f"  Synthetic: {len(synthetic_df)} points, {synthetic_df['trajectory_id'].nunique()} trips")
    print(f"  Original: {len(original_df)} points, {original_df['trajectory_id'].nunique()} trips")
    
    # Compute metrics
    print("\nComputing path metrics...")
    if G is not None:
        synthetic_metrics = compute_trip_metrics(synthetic_df, G, "synthetic")
        original_metrics = compute_trip_metrics(original_df, G, "original")
    else:
        print("  ⚠ Skipping shortest path computation (no OSM graph)")
        # Compute only actual path and straight-line distance
        synthetic_metrics = compute_trip_metrics_basic(synthetic_df, "synthetic")
        original_metrics = compute_trip_metrics_basic(original_df, "original")
    
    print(f"  ✓ Computed metrics for {len(synthetic_metrics)} synthetic trips")
    print(f"  ✓ Computed metrics for {len(original_metrics)} original trips")
    
    # Save metrics to CSV
    synthetic_metrics.to_csv(os.path.join(args.output, 'synthetic_path_metrics.csv'), index=False)
    original_metrics.to_csv(os.path.join(args.output, 'original_path_metrics.csv'), index=False)
    print(f"\n✓ Saved metrics to CSV files")
    
    # Create comparison plots
    print("\nGenerating comparison plots...")
    
    if G is not None:
        # Plot all metrics
        plot_metric_comparison(
            original_metrics, synthetic_metrics,
            'straight_line_distance', 'Straight-Line Distance', 'meters',
            os.path.join(args.output, 'straight_line_distance_comparison.png')
        )
        
        plot_metric_comparison(
            original_metrics, synthetic_metrics,
            'actual_path_length', 'Actual Path Length', 'meters',
            os.path.join(args.output, 'actual_path_length_comparison.png')
        )
        
        plot_metric_comparison(
            original_metrics, synthetic_metrics,
            'shortest_path_length', 'Shortest Path Length', 'meters',
            os.path.join(args.output, 'shortest_path_length_comparison.png')
        )
        
        plot_metric_comparison(
            original_metrics, synthetic_metrics,
            'detour_ratio', 'Detour Ratio', 'ratio',
            os.path.join(args.output, 'detour_ratio_comparison.png')
        )
        
        plot_metric_comparison(
            original_metrics, synthetic_metrics,
            'efficiency', 'Path Efficiency', 'ratio',
            os.path.join(args.output, 'efficiency_comparison.png')
        )
    else:
        # Plot only available metrics
        plot_metric_comparison(
            original_metrics, synthetic_metrics,
            'straight_line_distance', 'Straight-Line Distance', 'meters',
            os.path.join(args.output, 'straight_line_distance_comparison.png')
        )
        
        plot_metric_comparison(
            original_metrics, synthetic_metrics,
            'actual_path_length', 'Actual Path Length', 'meters',
            os.path.join(args.output, 'actual_path_length_comparison.png')
        )
    
    # Print summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    if G is not None:
        print("\nDetour Ratio (actual_path / shortest_path):")
        print(f"  Original - Mean: {original_metrics['detour_ratio'].mean():.3f}, "
              f"Median: {original_metrics['detour_ratio'].median():.3f}")
        print(f"  Synthetic - Mean: {synthetic_metrics['detour_ratio'].mean():.3f}, "
              f"Median: {synthetic_metrics['detour_ratio'].median():.3f}")
        
        print("\nPath Efficiency (straight_line / actual_path):")
        print(f"  Original - Mean: {original_metrics['efficiency'].mean():.3f}, "
              f"Median: {original_metrics['efficiency'].median():.3f}")
        print(f"  Synthetic - Mean: {synthetic_metrics['efficiency'].mean():.3f}, "
              f"Median: {synthetic_metrics['efficiency'].median():.3f}")
    
    print("\nActual Path Length:")
    print(f"  Original - Mean: {original_metrics['actual_path_length'].mean()/1000:.2f} km, "
          f"Median: {original_metrics['actual_path_length'].median()/1000:.2f} km")
    print(f"  Synthetic - Mean: {synthetic_metrics['actual_path_length'].mean()/1000:.2f} km, "
          f"Median: {synthetic_metrics['actual_path_length'].median()/1000:.2f} km")
    
    print("\n" + "=" * 60)
    print(f"Diagnosis complete! Results saved to: {args.output}")
    print("=" * 60)


def compute_trip_metrics_basic(df: pd.DataFrame, trajectory_type: str = "unknown") -> pd.DataFrame:
    """
    Compute basic path metrics without shortest path (when OSM graph unavailable).
    """
    grouped = df.groupby('trajectory_id')
    metrics = []
    
    for traj_id, group in grouped:
        if len(group) < 2:
            continue
        
        origin = group.iloc[0]
        dest = group.iloc[-1]
        
        straight_line_dist = haversine_distance(origin['lat'], origin['lon'], 
                                                dest['lat'], dest['lon'])
        
        coords = group[['lat', 'lon']].values
        actual_path_length = 0.0
        for i in range(len(coords) - 1):
            actual_path_length += haversine_distance(
                coords[i][0], coords[i][1],
                coords[i+1][0], coords[i+1][1]
            )
        
        metrics.append({
            'trajectory_id': traj_id,
            'straight_line_distance': straight_line_dist,
            'actual_path_length': actual_path_length,
            'shortest_path_length': np.nan,
            'detour_ratio': np.nan,
            'efficiency': straight_line_dist / actual_path_length if actual_path_length > 0 else np.nan,
        })
    
    return pd.DataFrame(metrics)


if __name__ == "__main__":
    main()

