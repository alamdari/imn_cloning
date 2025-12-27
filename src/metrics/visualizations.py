"""
Generate visualizations for trajectory quality metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional, Tuple
import os
import folium


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


def plot_origin_density_interactive_map(trajectory_stats: pd.DataFrame, output_path: str, 
                                       grid_size_m: float = 500.0):
    """
    Create interactive Folium map with origin density shown as grid cells (rectangles).
    
    Args:
        trajectory_stats: DataFrame with origin_lat, origin_lon columns
        output_path: Path to save HTML file
        grid_size_m: Grid cell size in meters (default 500m)
    """
    if trajectory_stats.empty or 'origin_lat' not in trajectory_stats.columns:
        print(f"  ⚠ Skipped: {os.path.basename(output_path)} (no origin data)")
        return
    
    # Import spatial stats for grid assignment
    from .spatial_stats import assign_to_grid
    
    # Calculate map center
    center_lat = trajectory_stats['origin_lat'].mean()
    center_lon = trajectory_stats['origin_lon'].mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Aggregate origins by grid cell
    from collections import defaultdict
    cell_counts = defaultdict(int)
    cell_coords = {}  # Store representative lat/lon for each cell
    
    for _, row in trajectory_stats.iterrows():
        cell = assign_to_grid(row['origin_lat'], row['origin_lon'], grid_size_m)
        cell_counts[cell] += 1
        if cell not in cell_coords:
            cell_coords[cell] = (row['origin_lat'], row['origin_lon'])
    
    # Find min/max counts for color scaling
    if cell_counts:
        max_count = max(cell_counts.values())
        min_count = min(cell_counts.values())
    else:
        max_count = 1
        min_count = 0
    
    # Create grid cell rectangles as a toggleable group
    grid_group = folium.FeatureGroup(name='Grid Cells (Origin Density)', show=True)
    
    # Approximate conversion for grid cell boundaries
    meters_per_degree_lat = 111000.0
    
    for cell, count in cell_counts.items():
        cell_x, cell_y = cell
        # Get representative coordinates
        rep_lat, rep_lon = cell_coords[cell]
        meters_per_degree_lon = 111000.0 * np.cos(np.radians(rep_lat))
        
        # Calculate cell boundaries
        lat_offset = grid_size_m / meters_per_degree_lat
        lon_offset = grid_size_m / meters_per_degree_lon
        
        # Calculate corner coordinates of the grid cell
        min_lat = cell_y * grid_size_m / meters_per_degree_lat
        max_lat = (cell_y + 1) * grid_size_m / meters_per_degree_lat
        min_lon = cell_x * grid_size_m / meters_per_degree_lon
        max_lon = (cell_x + 1) * grid_size_m / meters_per_degree_lon
        
        # Normalize count to 0-1 for color intensity
        if max_count > min_count:
            normalized = (count - min_count) / (max_count - min_count)
        else:
            normalized = 1.0
        
        # Color gradient: blue (low) -> yellow -> red (high)
        if normalized < 0.33:
            # Blue to cyan
            r = int(0 + (0 * normalized / 0.33))
            g = int(0 + (255 * normalized / 0.33))
            b = 255
        elif normalized < 0.66:
            # Cyan to yellow
            norm_local = (normalized - 0.33) / 0.33
            r = int(0 + (255 * norm_local))
            g = 255
            b = int(255 - (255 * norm_local))
        else:
            # Yellow to red
            norm_local = (normalized - 0.66) / 0.34
            r = 255
            g = int(255 - (255 * norm_local))
            b = 0
        
        color = f'#{r:02x}{g:02x}{b:02x}'
        
        # Create rectangle for this cell
        bounds = [[min_lat, min_lon], [max_lat, max_lon]]
        
        folium.Rectangle(
            bounds=bounds,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.5,
            weight=1,
            opacity=0.8,
            popup=f"<b>Grid Cell</b><br>Origins: {count}<br>Cell: ({cell_x}, {cell_y})",
            tooltip=f"{count} origins"
        ).add_to(grid_group)
    
    grid_group.add_to(m)
    
    # Add individual origin markers as a toggleable group
    marker_group = folium.FeatureGroup(name='Origin Points', show=False)
    
    # Aggregate origins by location to show counts
    origin_counts = trajectory_stats.groupby(['origin_lat', 'origin_lon']).size().reset_index(name='count')
    
    for _, row in origin_counts.iterrows():
        folium.CircleMarker(
            location=[row['origin_lat'], row['origin_lon']],
            radius=min(3 + np.log1p(row['count']), 10),
            popup=f"<b>Origin Point</b><br>Trips: {row['count']}<br>Lat: {row['origin_lat']:.5f}<br>Lon: {row['origin_lon']:.5f}",
            color='black',
            fill=True,
            fill_color='white',
            fill_opacity=0.8,
            weight=2
        ).add_to(marker_group)
    
    marker_group.add_to(m)
    
    # Add layer control (allows toggling grid cells and markers on/off)
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Save map
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    print(f"  ✓ Saved: {os.path.basename(output_path)} ({len(cell_counts)} grid cells)")


def plot_od_pairs_interactive_map(trajectory_stats: pd.DataFrame, od_matrix: pd.DataFrame, 
                                   output_path: str, top_n: int = 20):
    """
    Create interactive Folium map showing top OD pairs with flow lines.
    
    Args:
        trajectory_stats: DataFrame with trajectory statistics
        od_matrix: OD matrix with trip counts
        output_path: Path to save HTML file
        top_n: Number of top OD pairs to show
    """
    if od_matrix.empty:
        print(f"  ⚠ Skipped: {os.path.basename(output_path)} (no OD data)")
        return
    
    # Get top OD pairs
    top_od = od_matrix.nlargest(top_n, 'trip_count')
    
    # Calculate map center from all origins/destinations in trajectory stats
    center_lat = trajectory_stats[['origin_lat', 'dest_lat']].mean().mean()
    center_lon = trajectory_stats[['origin_lon', 'dest_lon']].mean().mean()
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # For each top OD pair, draw flow line using coordinates from od_matrix
    for idx, row in top_od.iterrows():
        # Use coordinates directly from OD matrix (already includes representative coords)
        origin_coords = [row['origin_lat'], row['origin_lon']]
        dest_coords = [row['dest_lat'], row['dest_lon']]
        
        # Calculate line weight based on trip count
        weight = 2 + (row['trip_count'] / top_od['trip_count'].max()) * 5
        
        # Draw flow line
        folium.PolyLine(
            locations=[origin_coords, dest_coords],
            color='red',
            weight=weight,
            opacity=0.6,
            popup=f"<b>Trips:</b> {row['trip_count']}<br><b>Origin:</b> ({origin_coords[0]:.4f}, {origin_coords[1]:.4f})<br><b>Dest:</b> ({dest_coords[0]:.4f}, {dest_coords[1]:.4f})",
            tooltip=f"{row['trip_count']} trips"
        ).add_to(m)
        
        # Add origin marker
        folium.CircleMarker(
            location=origin_coords,
            radius=5 + np.log1p(row['trip_count']),
            color='green',
            fill=True,
            fill_color='lightgreen',
            fill_opacity=0.7,
            popup=f"<b>Origin</b><br>Trips: {row['trip_count']}<br>Coords: ({origin_coords[0]:.4f}, {origin_coords[1]:.4f})"
        ).add_to(m)
        
        # Add destination marker
        folium.CircleMarker(
            location=dest_coords,
            radius=5 + np.log1p(row['trip_count']),
            color='blue',
            fill=True,
            fill_color='lightblue',
            fill_opacity=0.7,
            popup=f"<b>Destination</b><br>Trips: {row['trip_count']}<br>Coords: ({dest_coords[0]:.4f}, {dest_coords[1]:.4f})"
        ).add_to(m)
    
    # Save map
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    m.save(output_path)
    print(f"  ✓ Saved: {os.path.basename(output_path)}")


def plot_distribution_comparison(synthetic_stats: pd.DataFrame, 
                                 original_stats: Optional[pd.DataFrame],
                                 output_dir: str,
                                 source_city: str = 'milan',
                                 target_city: str = 'porto'):
    """
    Create side-by-side comparison plots of synthetic vs original distributions.
    Also creates overlay versions with alpha transparency.
    Uses shared bins and scales for easy comparison.
    
    Args:
        synthetic_stats: Statistics from synthetic trajectories
        original_stats: Statistics from original trajectories (if available)
        output_dir: Directory to save comparison plots directly to output_dir
        source_city: Source city name (e.g., 'milan') for plot titles
        target_city: Target city name (e.g., 'porto') for plot titles
    """
    if original_stats is None or original_stats.empty:
        print("  ⚠ No original data for comparison")
        return
    
    # 1. Trip Duration Comparison
    syn_dur_min = synthetic_stats['duration_seconds'] / 60.0
    orig_dur_min = original_stats['duration_seconds'] / 60.0
    
    # Compute shared bins and range
    all_dur = pd.concat([orig_dur_min, syn_dur_min])
    dur_min, dur_max = all_dur.min(), all_dur.max()
    dur_bins = 50
    
    # Side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.hist(orig_dur_min, bins=dur_bins, range=(dur_min, dur_max), edgecolor='black', alpha=0.7, color='blue')
    ax1.set_xlabel('Trip Duration (minutes)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Original City - Trip Duration', fontsize=14, fontweight='bold')
    ax1.axvline(orig_dur_min.mean(), color='red', linestyle='--', 
                label=f'Mean: {orig_dur_min.mean():.1f} min')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(syn_dur_min, bins=dur_bins, range=(dur_min, dur_max), edgecolor='black', alpha=0.7, color='green')
    ax2.set_xlabel('Trip Duration (minutes)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Target City ({target_city.upper()}) - Trip Duration', fontsize=14, fontweight='bold')
    ax2.axvline(syn_dur_min.mean(), color='red', linestyle='--', 
                label=f'Mean: {syn_dur_min.mean():.1f} min')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: duration_comparison.png")
    
    # Overlay version
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(orig_dur_min, bins=dur_bins, range=(dur_min, dur_max), edgecolor='black', alpha=0.6, color='blue', label=f'Original ({source_city.upper()})')
    ax.hist(syn_dur_min, bins=dur_bins, range=(dur_min, dur_max), edgecolor='black', alpha=0.6, color='green', label=f'Synthetic ({target_city.upper()})')
    ax.set_xlabel('Trip Duration (minutes)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Trip Duration Comparison (Overlay)', fontsize=14, fontweight='bold')
    ax.axvline(orig_dur_min.mean(), color='blue', linestyle='--', alpha=0.7, label=f'Original Mean: {orig_dur_min.mean():.1f} min')
    ax.axvline(syn_dur_min.mean(), color='green', linestyle='--', alpha=0.7, label=f'Synthetic Mean: {syn_dur_min.mean():.1f} min')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'duration_comparison_overlay.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: duration_comparison_overlay.png")
    
    # 2. Trip Length Comparison
    syn_len_km = synthetic_stats['od_distance_meters'] / 1000.0
    orig_len_km = original_stats['od_distance_meters'] / 1000.0
    
    # Compute shared bins and range
    all_len = pd.concat([orig_len_km, syn_len_km])
    len_min, len_max = all_len.min(), all_len.max()
    len_bins = 50
    
    # Side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.hist(orig_len_km, bins=len_bins, range=(len_min, len_max), edgecolor='black', alpha=0.7, color='blue')
    ax1.set_xlabel('Trip Length (km)', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Original City - Trip Length', fontsize=14, fontweight='bold')
    ax1.axvline(orig_len_km.mean(), color='red', linestyle='--', 
                label=f'Mean: {orig_len_km.mean():.2f} km')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.hist(syn_len_km, bins=len_bins, range=(len_min, len_max), edgecolor='black', alpha=0.7, color='green')
    ax2.set_xlabel('Trip Length (km)', fontsize=12)
    ax2.set_ylabel('Frequency', fontsize=12)
    ax2.set_title(f'Target City ({target_city.upper()}) - Trip Length', fontsize=14, fontweight='bold')
    ax2.axvline(syn_len_km.mean(), color='red', linestyle='--', 
                label=f'Mean: {syn_len_km.mean():.2f} km')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: length_comparison.png")
    
    # Overlay version
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(orig_len_km, bins=len_bins, range=(len_min, len_max), edgecolor='black', alpha=0.6, color='blue', label=f'Original ({source_city.upper()})')
    ax.hist(syn_len_km, bins=len_bins, range=(len_min, len_max), edgecolor='black', alpha=0.6, color='green', label=f'Synthetic ({target_city.upper()})')
    ax.set_xlabel('Trip Length (km)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Trip Length Comparison (Overlay)', fontsize=14, fontweight='bold')
    ax.axvline(orig_len_km.mean(), color='blue', linestyle='--', alpha=0.7, label=f'Original Mean: {orig_len_km.mean():.2f} km')
    ax.axvline(syn_len_km.mean(), color='green', linestyle='--', alpha=0.7, label=f'Synthetic Mean: {syn_len_km.mean():.2f} km')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_comparison_overlay.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: length_comparison_overlay.png")
    
    # 2b. Path Length Comparison (separate folder) - actual path traveled along roads
    # Create separate folder for path length plots
    path_output_dir = os.path.join(output_dir, 'od_distance_plots')
    os.makedirs(path_output_dir, exist_ok=True)
    
    # Use path_length_meters (actual path traveled) instead of od_distance_meters.
    # Use the SAME number of bins (50) and SAME range (len_min, len_max) as length_comparison_overlay.png
    # so the histograms are directly comparable on the same x‑axis scale.
    if 'path_length_meters' in synthetic_stats.columns and 'path_length_meters' in original_stats.columns:
        syn_path_km = synthetic_stats['path_length_meters'] / 1000.0
        orig_path_km = original_stats['path_length_meters'] / 1000.0
        
        # Use SAME bins (range and count) as OD distance plot for direct comparison
        # This allows side-by-side comparison on the same x-axis scale
        # Note: Path lengths are typically 1.2-1.5x OD distances, so some values may exceed len_max
        # and will be grouped in the rightmost bin (this is expected and correct)
        path_bins = len_bins
        path_min, path_max = len_min, len_max
        
        # Verify data integrity: path_length should always be >= od_distance
        syn_path_lt_od = (syn_path_km < syn_len_km).sum()
        orig_path_lt_od = (orig_path_km < orig_len_km).sum()
        if syn_path_lt_od > 0 or orig_path_lt_od > 0:
            print(f"  ⚠ WARNING: {syn_path_lt_od} synthetic and {orig_path_lt_od} original path_length values are less than od_distance (data error!)")
        
        # Count how many path_length values exceed the OD distance range
        syn_exceed = (syn_path_km > len_max).sum()
        orig_exceed = (orig_path_km > len_max).sum()
        if syn_exceed > 0 or orig_exceed > 0:
            print(f"  Note: {syn_exceed} synthetic and {orig_exceed} original path_length values exceed OD distance range (will be in rightmost bin)")
        
        # Overlay version for path length
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(orig_path_km, bins=path_bins, range=(path_min, path_max), 
                edgecolor='black', alpha=0.6, color='blue', 
                label=f'Original ({source_city.upper()})')
        ax.hist(syn_path_km, bins=path_bins, range=(path_min, path_max), 
                edgecolor='black', alpha=0.6, color='green', 
                label=f'Synthetic ({target_city.upper()})')
        ax.set_xlabel('Path Length (km)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Path Length Comparison (Overlay)', fontsize=14, fontweight='bold')
        ax.axvline(orig_path_km.mean(), color='blue', linestyle='--', alpha=0.7, 
                  label=f'Original Mean: {orig_path_km.mean():.2f} km')
        ax.axvline(syn_path_km.mean(), color='green', linestyle='--', alpha=0.7, 
                  label=f'Synthetic Mean: {syn_path_km.mean():.2f} km')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(path_output_dir, 'od_distance_comparison_overlay.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: od_distance_plots/od_distance_comparison_overlay.png")
    else:
        print(f"  ⚠ path_length_meters column not found, skipping path length comparison")
    
    # 3. Temporal Distribution Comparison
    if 'start_hour' in original_stats.columns and 'start_hour' in synthetic_stats.columns:
        # Shared bins for temporal (always 0-24)
        temp_bins = 24
        temp_range = (0, 24)
        
        # Side-by-side plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        ax1.hist(original_stats['start_hour'], bins=temp_bins, range=temp_range, 
                edgecolor='black', alpha=0.7, color='blue')
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Number of Trips', fontsize=12)
        ax1.set_title('Original City - Temporal Distribution', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(0, 24, 2))
        ax1.grid(True, alpha=0.3, axis='y')
        
        ax2.hist(synthetic_stats['start_hour'], bins=temp_bins, range=temp_range, 
                edgecolor='black', alpha=0.7, color='green')
        ax2.set_xlabel('Hour of Day', fontsize=12)
        ax2.set_ylabel('Number of Trips', fontsize=12)
        ax2.set_title(f'Target City ({target_city.upper()}) - Temporal Distribution', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(0, 24, 2))
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temporal_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: temporal_comparison.png")
        
        # Overlay version
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(original_stats['start_hour'], bins=temp_bins, range=temp_range, 
                edgecolor='black', alpha=0.6, color='blue', label=f'Original ({source_city.upper()})')
        ax.hist(synthetic_stats['start_hour'], bins=temp_bins, range=temp_range, 
                edgecolor='black', alpha=0.6, color='green', label=f'Synthetic ({target_city.upper()})')
        ax.set_xlabel('Hour of Day', fontsize=12)
        ax.set_ylabel('Number of Trips', fontsize=12)
        ax.set_title('Temporal Distribution Comparison (Overlay)', fontsize=14, fontweight='bold')
        ax.set_xticks(range(0, 24, 2))
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'temporal_comparison_overlay.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved: temporal_comparison_overlay.png")


def plot_path_metrics_comparison(original_metrics: pd.DataFrame, synthetic_metrics: pd.DataFrame,
                                 output_dir: str):
    """
    Create side-by-side comparison plots for path metrics.
    Also creates overlay versions with alpha transparency.
    Uses shared bins and scales for easy comparison.
    
    Args:
        original_metrics: DataFrame with original trajectory path metrics
        synthetic_metrics: DataFrame with synthetic trajectory path metrics
        output_dir: Directory to save plots
    """
    print("  Generating path metrics comparison plots...")
    
    # Plot each metric
    metrics_to_plot = [
        ('straight_line_distance', 'Straight-Line Distance', 'meters'),
        ('actual_path_length', 'Actual Path Length', 'meters'),
        ('shortest_path_length', 'Shortest Path Length', 'meters'),
        ('detour_ratio', 'Detour Ratio', 'ratio'),
        ('efficiency', 'Path Efficiency', 'ratio'),
    ]
    
    for metric_name, metric_label, unit in metrics_to_plot:
        if metric_name not in original_metrics.columns or metric_name not in synthetic_metrics.columns:
            continue
        
        # Skip if all values are NaN
        if original_metrics[metric_name].isna().all() and synthetic_metrics[metric_name].isna().all():
            continue
        
        orig_data = original_metrics[metric_name].dropna()
        syn_data = synthetic_metrics[metric_name].dropna()
        
        if len(orig_data) == 0 and len(syn_data) == 0:
            continue
        
        # Compute shared bins and range
        all_data = pd.concat([orig_data, syn_data])
        data_min, data_max = all_data.min(), all_data.max()
        # Use 50 bins, but ensure we have valid range
        if data_max > data_min:
            bins = 50
        else:
            bins = 10  # Fallback for edge case
        
        # Side-by-side plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Original distribution
        if len(orig_data) > 0:
            ax1.hist(orig_data, bins=bins, range=(data_min, data_max), edgecolor='black', alpha=0.7, color='blue')
            mean_val = orig_data.mean()
            ax1.axvline(mean_val, color='red', linestyle='--', 
                       label=f'Mean: {mean_val:.3f}')
            ax1.legend()
        
        ax1.set_xlabel(f'{metric_label} ({unit})', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title(f'Original City - {metric_label}', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Synthetic distribution
        if len(syn_data) > 0:
            ax2.hist(syn_data, bins=bins, range=(data_min, data_max), edgecolor='black', alpha=0.7, color='green')
            mean_val = syn_data.mean()
            ax2.axvline(mean_val, color='red', linestyle='--', 
                       label=f'Mean: {mean_val:.3f}')
            ax2.legend()
        
        ax2.set_xlabel(f'{metric_label} ({unit})', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title(f'Target City (Synthetic) - {metric_label}', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'path_metrics_{metric_name}_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    ✓ Saved: path_metrics_{metric_name}_comparison.png")
        
        # Overlay version
        if len(orig_data) > 0 or len(syn_data) > 0:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            if len(orig_data) > 0:
                ax.hist(orig_data, bins=bins, range=(data_min, data_max), 
                       edgecolor='black', alpha=0.6, color='blue', label='Original')
                orig_mean = orig_data.mean()
                ax.axvline(orig_mean, color='blue', linestyle='--', alpha=0.7, 
                          label=f'Original Mean: {orig_mean:.3f}')
            
            if len(syn_data) > 0:
                ax.hist(syn_data, bins=bins, range=(data_min, data_max), 
                       edgecolor='black', alpha=0.6, color='green', label='Synthetic')
                syn_mean = syn_data.mean()
                ax.axvline(syn_mean, color='green', linestyle='--', alpha=0.7, 
                          label=f'Synthetic Mean: {syn_mean:.3f}')
            
            ax.set_xlabel(f'{metric_label} ({unit})', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title(f'{metric_label} Comparison (Overlay)', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            overlay_path = os.path.join(output_dir, f'path_metrics_{metric_name}_comparison_overlay.png')
            plt.savefig(overlay_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"    ✓ Saved: path_metrics_{metric_name}_comparison_overlay.png")

