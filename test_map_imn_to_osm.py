#!/usr/bin/env python3
"""
Test script for isolating and visualizing the map_imn_to_osm function.

This script:
1. Loads IMN and POI data for 10 random users + user 302397
2. Prepares spatial resources (Porto OSM graph, population data)
3. Enriches IMNs with POI activity labels
4. Saves all inputs as pickle for reuse
5. Calls map_imn_to_osm for each user
6. Visualizes original IMN locations vs mapped Porto locations
"""

import os
import pickle
import random
import gzip
import json
from typing import Dict, List, Tuple
import numpy as np
import folium
from folium import FeatureGroup, Marker, CircleMarker, PolyLine

# Import required modules
from src.io.paths import PathsConfig
from src.io.data import read_poi_data
from src.io import imn_loading
from src.population.utils import generate_cumulative_map
from src.spatial.resources import ensure_spatial_resources
from src.synthetic.enrich import enrich_imn_with_poi
from src.spatial.mapping import map_imn_to_osm


# Configuration
TEST_DATA_DIR = "results/test_mapping"
PICKLE_PATH = os.path.join(TEST_DATA_DIR, "test_inputs.pkl")
RESULTS_DIR = TEST_DATA_DIR
NUM_RANDOM_USERS = 10
SPECIFIC_USER = 302397
RANDOM_SEED = 42


def load_test_users(imn_path: str, poi_path: str) -> Tuple[Dict, Dict]:
    """Load 10 random users + user 302397 from the datasets."""
    print("Loading IMN and POI data...")
    
    # Load all IMNs
    full_imns = imn_loading.read_imn(imn_path)
    print(f"  ✓ Loaded {len(full_imns)} IMNs")
    
    # Load all POI data
    poi_data = read_poi_data(poi_path)
    print(f"  ✓ Loaded POI data for {len(poi_data)} users")
    
    # Find common users
    common_users = set(full_imns.keys()) & set(poi_data.keys())
    print(f"  ✓ Found {len(common_users)} users with both IMN and POI data")
    
    # Select users: 10 random + specific user
    random.seed(RANDOM_SEED)
    available_users = sorted(common_users)
    
    selected_users = set()
    
    # Add specific user if available
    if SPECIFIC_USER in common_users:
        selected_users.add(SPECIFIC_USER)
        print(f"  ✓ Added specific user {SPECIFIC_USER}")
    else:
        print(f"  ⚠ User {SPECIFIC_USER} not found in common users")
    
    # Add 10 random users (excluding specific user if already added)
    remaining_users = [u for u in available_users if u not in selected_users]
    random_users = random.sample(remaining_users, min(NUM_RANDOM_USERS, len(remaining_users)))
    selected_users.update(random_users)
    
    print(f"  ✓ Selected {len(selected_users)} users total")
    print(f"    Random users: {sorted(random_users)[:5]}{'...' if len(random_users) > 5 else ''}")
    
    # Filter datasets
    imns = {uid: full_imns[uid] for uid in selected_users}
    poi_data_filtered = {uid: poi_data[uid] for uid in selected_users}
    
    return imns, poi_data_filtered


def prepare_test_data(force_reload: bool = False) -> Dict:
    """
    Prepare all inputs needed for map_imn_to_osm.
    
    Returns a dictionary with:
    - enriched_imns: Dict of enriched IMNs
    - G: OSM graph
    - gdf_cumulative_p: Population cumulative probability DataFrame
    - user_ids: List of user IDs
    """
    os.makedirs(TEST_DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Check if pickle exists and load if not forcing reload
    if os.path.exists(PICKLE_PATH) and not force_reload:
        print(f"Loading cached test data from {PICKLE_PATH}...")
        with open(PICKLE_PATH, 'rb') as f:
            data = pickle.load(f)
        print("  ✓ Cached data loaded")
        return data
    
    print("Preparing fresh test data...")
    print("=" * 60)
    
    # Load IMN and POI data
    imns, poi_data = load_test_users(
        "data/milano_2007_full_imns.json.gz",
        "data/milano_2007_full_imns_pois.json.gz"
    )
    
    # Prepare spatial resources (Porto)
    print("\nPreparing spatial resources (Porto OSM + population)...")
    G, gdf_cumulative_p, activity_pools = ensure_spatial_resources("data", generate_cumulative_map)
    print("  ✓ Spatial resources ready")
    
    # Enrich IMNs with POI activity labels
    print("\nEnriching IMNs with POI activity labels...")
    enriched_imns = {}
    for uid, imn in imns.items():
        enriched = enrich_imn_with_poi(imn, poi_data[uid])
        enriched_imns[uid] = enriched
    print(f"  ✓ Enriched {len(enriched_imns)} IMNs")
    
    # Prepare data structure
    data = {
        'enriched_imns': enriched_imns,
        'G': G,
        'gdf_cumulative_p': gdf_cumulative_p,
        'activity_pools': activity_pools,
        'user_ids': sorted(enriched_imns.keys()),
    }
    
    # Save to pickle
    print(f"\nSaving test data to {PICKLE_PATH}...")
    with open(PICKLE_PATH, 'wb') as f:
        pickle.dump(data, f)
    print("  ✓ Test data saved")
    
    return data


def visualize_mapping(user_id: int, enriched_imn: Dict, map_loc_imn: Dict, rmse: float, G) -> str:
    """
    Create a visualization comparing original IMN locations with mapped Porto locations.
    
    Returns the path to the saved HTML file.
    """
    # Create a folium map with two panes
    output_path = os.path.join(RESULTS_DIR, f"user_{user_id}_mapping_comparison.html")
    
    # Get original IMN center
    locations = enriched_imn['locations']
    if not locations:
        print(f"  ⚠ No locations for user {user_id}")
        return None
    
    # Calculate original city center
    orig_lats = [loc['coordinates'][1] for loc in locations.values()]
    orig_lons = [loc['coordinates'][0] for loc in locations.values()]
    orig_center = (np.mean(orig_lats), np.mean(orig_lons))
    
    # Get Porto center from mapped nodes
    porto_lats = [G.nodes[node]['y'] for node in map_loc_imn.values()]
    porto_lons = [G.nodes[node]['x'] for node in map_loc_imn.values()]
    porto_center = (np.mean(porto_lats), np.mean(porto_lons))
    
    # Create HTML with side-by-side maps
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>User {user_id} - IMN Mapping Comparison (RMSE: {rmse:.2f} km)</title>
        <style>
            body {{
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
            }}
            .header {{
                background-color: #2c3e50;
                color: white;
                padding: 15px;
                text-align: center;
            }}
            .container {{
                display: flex;
                height: calc(100vh - 80px);
            }}
            .map-pane {{
                flex: 1;
                position: relative;
            }}
            .map-title {{
                position: absolute;
                top: 10px;
                left: 50%;
                transform: translateX(-50%);
                background-color: white;
                padding: 10px 20px;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.3);
                z-index: 1000;
                font-weight: bold;
            }}
            #map-original, #map-porto {{
                width: 100%;
                height: 100%;
            }}
        </style>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css"/>
        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/leaflet-measure@3.1.0/dist/leaflet-measure.css"/>
        <script src="https://cdn.jsdelivr.net/npm/leaflet-measure@3.1.0/dist/leaflet-measure.min.js"></script>
    </head>
    <body>
        <div class="header">
            <h2>User {user_id} - IMN to OSM Mapping Comparison</h2>
            <p>RMSE: {rmse:.2f} km | Locations: {len(locations)} | Home: {enriched_imn['home']} | Work: {enriched_imn['work']}</p>
        </div>
        <div class="container">
            <div class="map-pane">
                <div class="map-title">Original IMN (Source City)</div>
                <div id="map-original"></div>
            </div>
            <div class="map-pane">
                <div class="map-title">Mapped Locations (Porto)</div>
                <div id="map-porto"></div>
            </div>
        </div>
        
        <script>
            // Original IMN Map
            var mapOriginal = L.map('map-original').setView([{orig_center[0]}, {orig_center[1]}], 13);
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '© OpenStreetMap contributors',
                opacity: 0.5
            }}).addTo(mapOriginal);
            
            // Add distance measurement tool to original map
            var measureControlOriginal = new L.Control.Measure({{
                primaryLengthUnit: 'kilometers',
                secondaryLengthUnit: 'meters',
                primaryAreaUnit: 'sqkilometers',
                secondaryAreaUnit: 'sqmeters'
            }});
            measureControlOriginal.addTo(mapOriginal);
            
            // Porto Map
            var mapPorto = L.map('map-porto').setView([{porto_center[0]}, {porto_center[1]}], 13);
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '© OpenStreetMap contributors',
                opacity: 0.5
            }}).addTo(mapPorto);
            
            // Add distance measurement tool to Porto map
            var measureControlPorto = new L.Control.Measure({{
                primaryLengthUnit: 'kilometers',
                secondaryLengthUnit: 'meters',
                primaryAreaUnit: 'sqkilometers',
                secondaryAreaUnit: 'sqmeters'
            }});
            measureControlPorto.addTo(mapPorto);
            
            // Add original locations
    """
    
    # Add markers for each location
    for loc_id, loc_data in locations.items():
        lon, lat = loc_data['coordinates']
        activity = loc_data.get('activity_label', 'unknown')
        freq = loc_data.get('frequency', 0)
        
        # Determine color using same palette as maps.py
        color_map = {
            'home': 'darkblue', 'work': 'orange', 'eat': 'green', 'utility': 'purple', 
            'transit': 'red', 'unknown': 'gray', 'school': 'yellow', 'shop': 'pink', 
            'leisure': 'lightgreen', 'health': 'lightcoral', 'admin': 'lightblue', 'finance': 'gold'
        }
        color = color_map.get(str(activity).lower() if activity is not None else 'unknown', 'black')
        
        # Original location marker
        html_content += f"""
            L.circleMarker([{lat}, {lon}], {{
                radius: 8,
                fillColor: '{color}',
                color: 'white',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }}).addTo(mapOriginal).bindPopup(
                '<b>Location {loc_id}</b><br>' +
                'Activity: {activity}<br>' +
                'Frequency: {freq}<br>' +
                'Coords: ({lat:.5f}, {lon:.5f})'
            );
        """
        
        # Mapped Porto location
        if loc_id in map_loc_imn:
            porto_node = map_loc_imn[loc_id]
            porto_lat = G.nodes[porto_node]['y']
            porto_lon = G.nodes[porto_node]['x']
            
            html_content += f"""
            L.circleMarker([{porto_lat}, {porto_lon}], {{
                radius: 8,
                fillColor: '{color}',
                color: 'white',
                weight: 2,
                opacity: 1,
                fillOpacity: 0.8
            }}).addTo(mapPorto).bindPopup(
                '<b>Location {loc_id} (mapped)</b><br>' +
                'Activity: {activity}<br>' +
                'OSM Node: {porto_node}<br>' +
                'Coords: ({porto_lat:.5f}, {porto_lon:.5f})'
            );
            """
    
    html_content += """
        </script>
    </body>
    </html>
    """
    
    # Save HTML
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    return output_path


def test_map_imn_to_osm(data: Dict):
    """
    Test map_imn_to_osm function for all users and visualize results.
    """
    print("\n" + "=" * 60)
    print("Testing map_imn_to_osm function")
    print("=" * 60)
    
    enriched_imns = data['enriched_imns']
    G = data['G']
    gdf_cumulative_p = data['gdf_cumulative_p']
    activity_pools = data.get('activity_pools')
    user_ids = data['user_ids']
    
    results = {}
    
    for idx, user_id in enumerate(user_ids, 1):
        print(f"\n[{idx}/{len(user_ids)}] Processing user {user_id}...")
        enriched = enriched_imns[user_id]
        
        # Call the function we're testing with activity pools and random seed
        try:
            map_loc_imn, rmse = map_imn_to_osm(
                enriched, 
                G, 
                n_trials=10,
                gdf_cumulative_p=gdf_cumulative_p,
                activity_pools=activity_pools,
                random_seed=RANDOM_SEED + user_id  # Unique but deterministic seed per user
            )
            print(f"  ✓ Mapping complete - RMSE: {rmse:.2f} km")
            print(f"  ✓ Mapped {len(map_loc_imn)} locations")
            print(f"    Home: {enriched['home']} → OSM node {map_loc_imn.get(enriched['home'])}")
            print(f"    Work: {enriched['work']} → OSM node {map_loc_imn.get(enriched['work'])}")
            
            # Show activity breakdown
            activity_counts = {}
            for loc_id in enriched['locations']:
                act = enriched['locations'][loc_id].get('activity_label', 'unknown')
                activity_counts[act] = activity_counts.get(act, 0) + 1
            print(f"    Activities: {dict(sorted(activity_counts.items()))}")
            
            # Store results
            results[user_id] = {
                'map_loc_imn': map_loc_imn,
                'rmse': rmse,
                'num_locations': len(map_loc_imn)
            }
            
            # Create visualization
            viz_path = visualize_mapping(user_id, enriched, map_loc_imn, rmse, G)
            if viz_path:
                print(f"  ✓ Visualization saved: {os.path.basename(viz_path)}")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results summary
    summary_path = os.path.join(RESULTS_DIR, "mapping_results_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("IMN to OSM Mapping Results Summary\n")
        f.write("=" * 60 + "\n\n")
        
        for user_id in sorted(results.keys()):
            res = results[user_id]
            f.write(f"User {user_id}:\n")
            f.write(f"  RMSE: {res['rmse']:.2f} km\n")
            f.write(f"  Locations mapped: {res['num_locations']}\n")
            f.write(f"  Mapping: {res['map_loc_imn']}\n")
            f.write("\n")
        
        # Statistics
        rmses = [res['rmse'] for res in results.values()]
        f.write(f"\nOverall Statistics:\n")
        f.write(f"  Mean RMSE: {np.mean(rmses):.2f} km\n")
        f.write(f"  Median RMSE: {np.median(rmses):.2f} km\n")
        f.write(f"  Min RMSE: {np.min(rmses):.2f} km\n")
        f.write(f"  Max RMSE: {np.max(rmses):.2f} km\n")
        f.write(f"  Std Dev: {np.std(rmses):.2f} km\n")
        
        f.write(f"\nAlgorithm Details:\n")
        f.write(f"  - Activity-aware mapping using activity pools\n")
        f.write(f"  - Home and work mapped first as anchor points\n")
        f.write(f"  - Other locations mapped incrementally by frequency\n")
        f.write(f"  - Preserves spatial structure (pairwise distances)\n")
        f.write(f"  - Deterministic with random_seed={RANDOM_SEED}\n")
    
    print(f"\n✓ Results summary saved: {summary_path}")
    print(f"✓ All visualizations saved to: {RESULTS_DIR}")
    
    return results


def main():
    """Main test execution."""
    print("=" * 60)
    print("IMN to OSM Mapping Test")
    print("=" * 60)
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Target users: {NUM_RANDOM_USERS} random + user {SPECIFIC_USER}")
    print()
    
    # Prepare test data (loads from cache if available)
    data = prepare_test_data(force_reload=False)
    
    # Test the function
    results = test_map_imn_to_osm(data)
    
    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
    print(f"Tested {len(results)} users")
    print(f"Results and visualizations saved to: {RESULTS_DIR}")
    print("\nTo force reload data, run with force_reload=True in prepare_test_data()")


if __name__ == "__main__":
    main()

