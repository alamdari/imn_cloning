#!/usr/bin/env python3
"""
Separate script to analyze activity pools effectiveness.

This script:
1. Loads OSM graph for a target city
2. Builds activity pools (or loads from cache)
3. Analyzes pool statistics (node counts, distributions)
4. Creates visualizations (maps, histograms)
5. Saves everything to results/pools_test/

Usage:
    python3 test_activity_pools.py --city porto
    python3 test_activity_pools.py --city milan
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from collections import Counter
from typing import Dict, List

import osmnx as ox

from src.spatial.resources import CITY_CONFIGS
from src.spatial.activity_pools import build_activity_node_pools


def load_osm_graph(city_name: str, data_dir: str = "data"):
    """Load OSM graph for the specified city."""
    if city_name not in CITY_CONFIGS:
        raise ValueError(f"Unknown city: {city_name}. Available: {list(CITY_CONFIGS.keys())}")
    
    city_config = CITY_CONFIGS[city_name]
    graphml_path = os.path.join(data_dir, f"{city_name}_drive.graphml")
    
    if os.path.exists(graphml_path):
        print(f"  → Loading cached OSM graph for {city_config['display_name']}...")
        G = ox.load_graphml(graphml_path)
    else:
        print(f"  → Downloading OSM graph for {city_config['display_name']}...")
        G = ox.graph.graph_from_point(
            city_config['center'], 
            dist=city_config['radius_m'], 
            network_type="drive", 
            simplify=False
        )
        G = ox.routing.add_edge_speeds(G)
        G = ox.routing.add_edge_travel_times(G)
        ox.save_graphml(G, graphml_path)
        print(f"  ✓ OSM graph cached")
    
    return G


def analyze_pool_statistics(activity_pools: Dict[str, List[int]], G) -> pd.DataFrame:
    """Compute statistics for each activity pool."""
    stats = []
    
    all_nodes = set(G.nodes())
    
    for activity, node_list in activity_pools.items():
        node_set = set(node_list)
        stats.append({
            'activity': activity,
            'node_count': len(node_set),
            'percentage_of_all_nodes': (len(node_set) / len(all_nodes)) * 100,
            'unique_nodes': len(node_set),
            'duplicate_count': len(node_list) - len(node_set) if len(node_list) > len(node_set) else 0
        })
    
    return pd.DataFrame(stats).sort_values('node_count', ascending=False)


def compute_spatial_distribution(activity_pools: Dict[str, List[int]], G, activity: str) -> pd.DataFrame:
    """Compute spatial distribution (lat/lon) of nodes for an activity."""
    if activity not in activity_pools:
        return pd.DataFrame()
    
    nodes = list(set(activity_pools[activity]))  # Remove duplicates
    coords = []
    
    for node in nodes:
        node_data = G.nodes[node]
        coords.append({
            'node_id': node,
            'lat': node_data['y'],
            'lon': node_data['x']
        })
    
    return pd.DataFrame(coords)


def plot_pool_size_distribution(stats_df: pd.DataFrame, output_path: str):
    """Plot bar chart of pool sizes."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    stats_sorted = stats_df.sort_values('node_count', ascending=True)
    
    bars = ax.barh(range(len(stats_sorted)), stats_sorted['node_count'], 
                   color='steelblue', edgecolor='black', alpha=0.7)
    
    ax.set_yticks(range(len(stats_sorted)))
    ax.set_yticklabels(stats_sorted['activity'])
    ax.set_xlabel('Number of Candidate Nodes', fontsize=12)
    ax.set_ylabel('Activity Type', fontsize=12)
    ax.set_title('Activity Pool Sizes', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(stats_sorted.iterrows()):
        ax.text(row['node_count'] + max(stats_sorted['node_count']) * 0.01, 
                i, f"{int(row['node_count'])}", 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(output_path)}")


def plot_pool_percentage_distribution(stats_df: pd.DataFrame, output_path: str):
    """Plot bar chart showing percentage of all nodes."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    stats_sorted = stats_df.sort_values('percentage_of_all_nodes', ascending=True)
    
    bars = ax.barh(range(len(stats_sorted)), stats_sorted['percentage_of_all_nodes'], 
                   color='coral', edgecolor='black', alpha=0.7)
    
    ax.set_yticks(range(len(stats_sorted)))
    ax.set_yticklabels(stats_sorted['activity'])
    ax.set_xlabel('Percentage of All Nodes (%)', fontsize=12)
    ax.set_ylabel('Activity Type', fontsize=12)
    ax.set_title('Activity Pool Coverage (% of Total Nodes)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (idx, row) in enumerate(stats_sorted.iterrows()):
        ax.text(row['percentage_of_all_nodes'] + max(stats_sorted['percentage_of_all_nodes']) * 0.01, 
                i, f"{row['percentage_of_all_nodes']:.1f}%", 
                va='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(output_path)}")


def plot_home_work_comparison(home_stats: pd.DataFrame, work_stats: pd.DataFrame, 
                              output_path: str):
    """Plot comparison of home vs work pool sizes."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Home distribution
    if len(home_stats) > 0:
        ax1.scatter(home_stats['lon'], home_stats['lat'], 
                   alpha=0.3, s=10, color='blue', label='Home nodes')
        ax1.set_xlabel('Longitude', fontsize=12)
        ax1.set_ylabel('Latitude', fontsize=12)
        ax1.set_title(f'Home Pool Distribution\n({len(home_stats)} nodes)', 
                     fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Work distribution
    if len(work_stats) > 0:
        ax2.scatter(work_stats['lon'], work_stats['lat'], 
                   alpha=0.3, s=10, color='red', label='Work nodes')
        ax2.set_xlabel('Longitude', fontsize=12)
        ax2.set_ylabel('Latitude', fontsize=12)
        ax2.set_title(f'Work Pool Distribution\n({len(work_stats)} nodes)', 
                     fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(output_path)}")


def create_interactive_map(activity_pools: Dict[str, List[int]], G, 
                          city_name: str, output_path: str):
    """Create interactive Folium map showing home and work node distributions."""
    city_config = CITY_CONFIGS[city_name]
    center_lat, center_lon = city_config['center']
    
    # Create base map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='OpenStreetMap'
    )
    
    # Add home nodes
    if 'home' in activity_pools:
        home_nodes = list(set(activity_pools['home']))
        home_group = folium.FeatureGroup(name='Home Nodes', show=True)
        for node_id in home_nodes[:5000]:  # Limit to 5000 for performance
            node_data = G.nodes[node_id]
            folium.CircleMarker(
                location=[node_data['y'], node_data['x']],
                radius=2,
                popup=f"Home Node: {node_id}",
                color='blue',
                fill=True,
                fill_color='blue',
                fill_opacity=0.6,
                weight=1
            ).add_to(home_group)
        home_group.add_to(m)
        print(f"    Added {min(len(home_nodes), 5000)} home nodes to map")
    
    # Add work nodes
    if 'work' in activity_pools:
        work_nodes = list(set(activity_pools['work']))
        work_group = folium.FeatureGroup(name='Work Nodes', show=True)
        for node_id in work_nodes[:5000]:  # Limit to 5000 for performance
            node_data = G.nodes[node_id]
            folium.CircleMarker(
                location=[node_data['y'], node_data['x']],
                radius=2,
                popup=f"Work Node: {node_id}",
                color='red',
                fill=True,
                fill_color='red',
                fill_opacity=0.6,
                weight=1
            ).add_to(work_group)
        work_group.add_to(m)
        print(f"    Added {min(len(work_nodes), 5000)} work nodes to map")
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Save map
    m.save(output_path)
    print(f"  ✓ Saved: {os.path.basename(output_path)}")


def analyze_overlaps(activity_pools: Dict[str, List[int]]) -> pd.DataFrame:
    """Analyze overlaps between different activity pools."""
    overlaps = []
    
    activities = list(activity_pools.keys())
    
    for i, act1 in enumerate(activities):
        set1 = set(activity_pools[act1])
        for act2 in activities[i+1:]:
            set2 = set(activity_pools[act2])
            intersection = set1 & set2
            union = set1 | set2
            
            if len(union) > 0:
                overlap_pct = (len(intersection) / len(union)) * 100
            else:
                overlap_pct = 0
            
            overlaps.append({
                'activity1': act1,
                'activity2': act2,
                'overlap_count': len(intersection),
                'overlap_percentage': overlap_pct,
                'activity1_size': len(set1),
                'activity2_size': len(set2)
            })
    
    return pd.DataFrame(overlaps).sort_values('overlap_percentage', ascending=False)


def plot_overlap_heatmap(overlap_df: pd.DataFrame, output_path: str):
    """Create heatmap of activity pool overlaps."""
    if len(overlap_df) == 0:
        print("  ⚠ No overlap data to plot")
        return
    
    # Create pivot table for heatmap
    activities = sorted(set(overlap_df['activity1'].unique()) | set(overlap_df['activity2'].unique()))
    overlap_matrix = pd.DataFrame(0, index=activities, columns=activities)
    
    for _, row in overlap_df.iterrows():
        overlap_matrix.loc[row['activity1'], row['activity2']] = row['overlap_percentage']
        overlap_matrix.loc[row['activity2'], row['activity1']] = row['overlap_percentage']
    
    # Set diagonal to 100% (self-overlap)
    for act in activities:
        overlap_matrix.loc[act, act] = 100.0
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(overlap_matrix, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': 'Overlap Percentage (%)'}, ax=ax)
    ax.set_title('Activity Pool Overlap Matrix', fontsize=14, fontweight='bold')
    ax.set_xlabel('Activity Type', fontsize=12)
    ax.set_ylabel('Activity Type', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved: {os.path.basename(output_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze activity pools effectiveness"
    )
    
    parser.add_argument(
        '--city',
        type=str,
        default='porto',
        choices=['porto', 'milan'],
        help='Target city to analyze (default: porto)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory for cached spatial resources (default: data)'
    )
    
    parser.add_argument(
        '--proximity',
        type=float,
        default=200.0,
        help='Proximity distance in meters for node selection (default: 200)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.join("results", "pools_test", args.city)
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("Activity Pools Analysis")
    print("=" * 60)
    print(f"City: {args.city}")
    print(f"Output directory: {output_dir}")
    print(f"Proximity: {args.proximity}m")
    print()
    
    # Step 1: Load OSM graph
    print("Step 1: Loading OSM graph...")
    G = load_osm_graph(args.city, args.data_dir)
    print(f"  ✓ Graph loaded: {len(G.nodes())} nodes, {len(G.edges())} edges")
    print()
    
    # Step 2: Build or load activity pools (use cache if available)
    print("Step 2: Loading activity pools...")
    activity_pools_pickle = os.path.join(args.data_dir, f"{args.city}_activity_pools.pkl")
    
    if os.path.exists(activity_pools_pickle):
        print(f"  → Loading cached activity pools from {activity_pools_pickle}...")
        with open(activity_pools_pickle, 'rb') as f:
            activity_pools = pickle.load(f)
        print(f"  ✓ Loaded activity pools from cache ({len(activity_pools)} activity types)")
    else:
        print("  → Building activity pools (this may take a while)...")
        activity_pools = build_activity_node_pools(
            G, 
            proximity_m=args.proximity, 
            cache_dir=args.data_dir, 
            city_name=args.city
        )
        # Cache the pools
        with open(activity_pools_pickle, 'wb') as f:
            pickle.dump(activity_pools, f)
        print(f"  ✓ Built and cached activity pools ({len(activity_pools)} activity types)")
    print()
    
    # Step 3: Analyze statistics
    print("Step 3: Computing pool statistics...")
    stats_df = analyze_pool_statistics(activity_pools, G)
    
    # Save statistics
    stats_path = os.path.join(output_dir, "pool_statistics.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"  ✓ Saved: {os.path.basename(stats_path)}")
    print()
    print("Pool Statistics:")
    print(stats_df.to_string(index=False))
    print()
    
    # Step 4: Create visualizations
    print("Step 4: Creating visualizations...")
    
    # Pool size distribution
    plot_pool_size_distribution(
        stats_df, 
        os.path.join(output_dir, "pool_size_distribution.png")
    )
    
    # Pool percentage distribution
    plot_pool_percentage_distribution(
        stats_df,
        os.path.join(output_dir, "pool_percentage_distribution.png")
    )
    
    # Home and work spatial distributions
    if 'home' in activity_pools and 'work' in activity_pools:
        home_coords = compute_spatial_distribution(activity_pools, G, 'home')
        work_coords = compute_spatial_distribution(activity_pools, G, 'work')
        
        plot_home_work_comparison(
            home_coords, 
            work_coords,
            os.path.join(output_dir, "home_work_spatial_comparison.png")
        )
        
        # Save coordinate data
        home_coords.to_csv(os.path.join(output_dir, "home_nodes_coordinates.csv"), index=False)
        work_coords.to_csv(os.path.join(output_dir, "work_nodes_coordinates.csv"), index=False)
        print(f"  ✓ Saved coordinate data for home and work nodes")
    
    # Interactive map
    create_interactive_map(
        activity_pools, 
        G, 
        args.city,
        os.path.join(output_dir, "activity_pools_map.html")
    )
    
    # Step 5: Analyze overlaps
    print()
    print("Step 5: Analyzing pool overlaps...")
    overlap_df = analyze_overlaps(activity_pools)
    
    if len(overlap_df) > 0:
        overlap_path = os.path.join(output_dir, "pool_overlaps.csv")
        overlap_df.to_csv(overlap_path, index=False)
        print(f"  ✓ Saved: {os.path.basename(overlap_path)}")
        
        # Focus on home-work overlap
        home_work_overlap = overlap_df[
            ((overlap_df['activity1'] == 'home') & (overlap_df['activity2'] == 'work')) |
            ((overlap_df['activity1'] == 'work') & (overlap_df['activity2'] == 'home'))
        ]
        if len(home_work_overlap) > 0:
            print()
            print("Home-Work Overlap:")
            print(home_work_overlap.to_string(index=False))
        
        # Create overlap heatmap
        plot_overlap_heatmap(
            overlap_df,
            os.path.join(output_dir, "pool_overlap_heatmap.png")
        )
    
    # Step 6: Summary report
    print()
    print("=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {output_dir}")
    print(f"\nGenerated files:")
    print(f"  - pool_statistics.csv")
    print(f"  - pool_size_distribution.png")
    print(f"  - pool_percentage_distribution.png")
    if 'home' in activity_pools and 'work' in activity_pools:
        print(f"  - home_work_spatial_comparison.png")
        print(f"  - home_nodes_coordinates.csv")
        print(f"  - work_nodes_coordinates.csv")
    print(f"  - activity_pools_map.html")
    if len(overlap_df) > 0:
        print(f"  - pool_overlaps.csv")
        print(f"  - pool_overlap_heatmap.png")
    
    # Key insights
    print(f"\nKey Insights:")
    if 'home' in activity_pools:
        home_count = len(set(activity_pools['home']))
        home_pct = stats_df[stats_df['activity'] == 'home']['percentage_of_all_nodes'].values[0]
        print(f"  - Home pool: {home_count} nodes ({home_pct:.1f}% of all nodes)")
    
    if 'work' in activity_pools:
        work_count = len(set(activity_pools['work']))
        work_pct = stats_df[stats_df['activity'] == 'work']['percentage_of_all_nodes'].values[0]
        print(f"  - Work pool: {work_count} nodes ({work_pct:.1f}% of all nodes)")
    
    if 'home' in activity_pools and 'work' in activity_pools:
        home_set = set(activity_pools['home'])
        work_set = set(activity_pools['work'])
        overlap = len(home_set & work_set)
        print(f"  - Home-Work overlap: {overlap} nodes")
        if len(home_set | work_set) > 0:
            overlap_pct = (overlap / len(home_set | work_set)) * 100
            print(f"    ({overlap_pct:.1f}% of combined pool)")
    
    print()


if __name__ == "__main__":
    main()

