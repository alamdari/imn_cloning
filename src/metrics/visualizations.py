"""
Generate visualizations for trajectory quality metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional
import os


def plot_trip_duration_distribution(trajectory_stats: pd.DataFrame, output_path: str):
    """
    Plot histogram of trip durations.
    """
    durations_minutes = trajectory_stats['duration_seconds'] / 60.0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(durations_minutes, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Trip Duration (minutes)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Trip Durations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_dur = durations_minutes.mean()
    median_dur = durations_minutes.median()
    ax.axvline(mean_dur, color='red', linestyle='--', label=f'Mean: {mean_dur:.1f} min')
    ax.axvline(median_dur, color='green', linestyle='--', label=f'Median: {median_dur:.1f} min')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(output_path)}")


def plot_trip_length_distribution(trajectory_stats: pd.DataFrame, output_path: str):
    """
    Plot histogram of trip lengths (OD distance).
    """
    lengths_km = trajectory_stats['od_distance_meters'] / 1000.0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(lengths_km, bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Trip Length (km)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Trip Lengths (Origin-Destination Distance)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mean_len = lengths_km.mean()
    median_len = lengths_km.median()
    ax.axvline(mean_len, color='red', linestyle='--', label=f'Mean: {mean_len:.2f} km')
    ax.axvline(median_len, color='green', linestyle='--', label=f'Median: {median_len:.2f} km')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(output_path)}")


def plot_trip_length_boxplot(all_user_stats: Dict[int, pd.DataFrame], output_path: str):
    """
    Box-whisker plot of trip lengths across all users.
    """
    # Prepare data
    data_for_plot = []
    for user_id, stats in all_user_stats.items():
        lengths_km = stats['od_distance_meters'] / 1000.0
        for length in lengths_km:
            data_for_plot.append({'user_id': user_id, 'trip_length_km': length})
    
    df_plot = pd.DataFrame(data_for_plot)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # If too many users, aggregate
    if len(all_user_stats) > 20:
        # Show overall distribution
        ax.boxplot(df_plot['trip_length_km'], vert=True, widths=0.5)
        ax.set_xlabel('All Users', fontsize=12)
    else:
        # Show per-user boxplots
        user_ids = sorted(all_user_stats.keys())
        data_by_user = [all_user_stats[uid]['od_distance_meters'] / 1000.0 for uid in user_ids]
        ax.boxplot(data_by_user, labels=[str(uid) for uid in user_ids], vert=True)
        ax.set_xlabel('User ID', fontsize=12)
        plt.xticks(rotation=45)
    
    ax.set_ylabel('Trip Length (km)', fontsize=12)
    ax.set_title('Distribution of Trip Lengths Across Users', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(output_path)}")


def plot_temporal_distribution(trajectory_stats: pd.DataFrame, output_path: str):
    """
    Plot histogram of trip start times in 1-hour bins.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.hist(trajectory_stats['start_hour'], bins=24, range=(0, 24), 
            edgecolor='black', alpha=0.7)
    ax.set_xlabel('Hour of Day', fontsize=12)
    ax.set_ylabel('Number of Trips', fontsize=12)
    ax.set_title('Temporal Distribution of Trip Start Times', fontsize=14, fontweight='bold')
    ax.set_xticks(range(0, 24))
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(output_path)}")


def plot_origin_density_heatmap(origin_density: pd.DataFrame, output_path: str):
    """
    Plot 2D heatmap of origin density.
    """
    # Pivot for heatmap
    pivot = origin_density.pivot_table(
        values='origin_count', 
        index='cell_y', 
        columns='cell_x', 
        fill_value=0
    )
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(pivot, cmap='YlOrRd', cbar_kws={'label': 'Number of Origins'}, 
                ax=ax, linewidths=0, square=False)
    
    ax.set_xlabel('Grid Cell X', fontsize=12)
    ax.set_ylabel('Grid Cell Y', fontsize=12)
    ax.set_title('Spatial Density of Trip Origins', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(output_path)}")


def plot_od_flow_summary(od_matrix: pd.DataFrame, output_path: str, top_n: int = 20):
    """
    Plot bar chart of top N OD pairs by trip count.
    """
    # Sort by trip count and take top N
    top_od = od_matrix.nlargest(top_n, 'trip_count')
    
    # Create labels
    top_od['od_label'] = top_od.apply(
        lambda x: f"({x['origin_cell_x']},{x['origin_cell_y']})→({x['dest_cell_x']},{x['dest_cell_y']})",
        axis=1
    )
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.barh(range(len(top_od)), top_od['trip_count'], color='steelblue', edgecolor='black')
    ax.set_yticks(range(len(top_od)))
    ax.set_yticklabels(top_od['od_label'], fontsize=8)
    ax.set_xlabel('Number of Trips', fontsize=12)
    ax.set_ylabel('Origin → Destination (Grid Cells)', fontsize=12)
    ax.set_title(f'Top {top_n} Origin-Destination Pairs', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(output_path)}")


def plot_day_of_week_distribution(trajectory_stats: pd.DataFrame, output_path: str):
    """
    Plot bar chart of trips by day of week.
    """
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    day_counts = trajectory_stats['start_day_of_week'].value_counts().sort_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(range(7), [day_counts.get(i, 0) for i in range(7)], 
           color='steelblue', edgecolor='black', alpha=0.7)
    ax.set_xticks(range(7))
    ax.set_xticklabels(day_names, rotation=45, ha='right')
    ax.set_xlabel('Day of Week', fontsize=12)
    ax.set_ylabel('Number of Trips', fontsize=12)
    ax.set_title('Trip Distribution by Day of Week', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(output_path)}")


def plot_path_vs_od_distance(trajectory_stats: pd.DataFrame, output_path: str):
    """
    Scatter plot comparing path length vs OD distance (circuity).
    """
    od_km = trajectory_stats['od_distance_meters'] / 1000.0
    path_km = trajectory_stats['path_length_meters'] / 1000.0
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(od_km, path_km, alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    
    # Add diagonal line (where path = OD)
    max_val = max(od_km.max(), path_km.max())
    ax.plot([0, max_val], [0, max_val], 'r--', label='Path = OD (straight line)')
    
    ax.set_xlabel('Origin-Destination Distance (km)', fontsize=12)
    ax.set_ylabel('Actual Path Length (km)', fontsize=12)
    ax.set_title('Path Length vs. OD Distance (Circuity)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(output_path)}")

