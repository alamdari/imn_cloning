import os
import pickle
import requests
import osmnx as ox
from typing import Tuple, Dict, List
import pandas as pd

# Spatial constants/resources
SEDAC_TIFF_URL = "https://data.ghg.center/sedac-popdensity-yeargrid5yr-v4.11/gpw_v4_population_density_rev11_2020_30_sec_2020.tif"

# City configurations: (center_lat, center_lon, radius_m)
CITY_CONFIGS = {
    'porto': {
        'center': (41.1494512, -8.6107884),
        'radius_m': 10000,
        'display_name': 'Porto, Portugal'
    },
    'milan': {
        'center': (45.4642, 9.1900),  # Milan city center
        'radius_m': 15000,  # Larger radius for Milan
        'display_name': 'Milan, Italy'
    }
}


# ** Prepare OSM graph and population raster (used for node sampling)
def ensure_spatial_resources(
    data_dir: str,
    generate_cumulative_map_fn,
    target_city: str = 'porto'
) -> Tuple[object, pd.DataFrame, Dict[str, List[int]]]:
    """
    Load/build spatial resources: OSM graph, population map, and activity-aware node pools.
    
    Args:
        data_dir: Directory for caching spatial resources
        generate_cumulative_map_fn: Function to generate population cumulative map
        target_city: Target city name ('porto' or 'milan')
    
    Returns:
        Tuple of (G, gdf_cumulative_p, activity_pools)
    """
    # Get city configuration
    if target_city not in CITY_CONFIGS:
        raise ValueError(f"Unknown target city: {target_city}. Available: {list(CITY_CONFIGS.keys())}")
    
    city_config = CITY_CONFIGS[target_city]
    city_center = city_config['center']
    city_radius = city_config['radius_m']
    
    os.makedirs(data_dir, exist_ok=True)
    graphml_path = os.path.join(data_dir, f"{target_city}_drive.graphml")
    tiff_path = os.path.join(data_dir, "gpw_v4_population_density_rev11_2020_30_sec_2020.tif")
    pop_pickle = os.path.join(data_dir, f"{target_city}_population_cumulative.pkl")
    activity_pools_pickle = os.path.join(data_dir, f"{target_city}_activity_pools.pkl")

    # OSM graph
    if os.path.exists(graphml_path):
        print(f"  → Loading cached OSM graph for {city_config['display_name']}...")
        G = ox.load_graphml(graphml_path)
    else:
        print(f"  → Downloading OSM graph for {city_config['display_name']} (center: {city_center}, radius: {city_radius}m)...")
        G = ox.graph.graph_from_point(city_center, dist=city_radius, network_type="drive", simplify=False)
        G = ox.routing.add_edge_speeds(G)
        G = ox.routing.add_edge_travel_times(G)
        ox.save_graphml(G, graphml_path)
        print(f"  ✓ OSM graph cached to {graphml_path}")

    # TIFF
    if not os.path.exists(tiff_path):
        print(f"  → Downloading global population density data...")
        resp = requests.get(SEDAC_TIFF_URL)
        resp.raise_for_status()
        with open(tiff_path, 'wb') as f:
            f.write(resp.content)
        print(f"  ✓ Population data downloaded")

    # Cumulative population map
    if os.path.exists(pop_pickle):
        print(f"  → Loading cached population map for {target_city}...")
        with open(pop_pickle, 'rb') as f:
            gdf_cumulative_p = pickle.load(f)
    else:
        print(f"  → Generating population cumulative map for {target_city}...")
        gdf_cumulative_p = generate_cumulative_map_fn(tiff_path, G)
        with open(pop_pickle, 'wb') as f:
            pickle.dump(gdf_cumulative_p, f)
        print(f"  ✓ Population map cached")

    # Activity-aware node pools
    if os.path.exists(activity_pools_pickle):
        with open(activity_pools_pickle, 'rb') as f:
            activity_pools = pickle.load(f)
        print(f"  ✓ Loaded activity pools from cache ({len(activity_pools)} activity types)")
    else:
        from src.spatial.activity_pools import build_activity_node_pools
        print(f"  → Building activity-aware node pools for {target_city}...")
        activity_pools = build_activity_node_pools(G, proximity_m=200, cache_dir=data_dir, city_name=target_city)
        with open(activity_pools_pickle, 'wb') as f:
            pickle.dump(activity_pools, f)
        print(f"  ✓ Built and cached activity pools ({len(activity_pools)} activity types)")

    return G, gdf_cumulative_p, activity_pools


