"""
Activity-aware OSM node pools for spatial mapping.
Uses amenity tags to build candidate node pools for each activity type.
"""
from typing import Dict, List, Optional
import numpy as np
import osmnx as ox
from collections import defaultdict


# Amenity classes from add_poi_to_imns.py for consistency
AMENITY_CLASSES = {
    'transportation': [
        'parking', 'taxi', 'fuel', 'parking_entrance', 'parking_exit', 'parking_access', 'car_rental',
        'bicycle_rental', 'bus_station', 'bicycle_parking', 'motorcycle_parking', 'ferry_terminal',
        'kick-scooter_parking', 'motorcycle_rental', 'taxi_rank'
    ],
    'healthcare': [
        'pharmacy', 'clinic', 'hospital', 'dentist', 'doctors', 'veterinary', 'nursing_home',
        'health_post'
    ],
    'public_services': [
        'post_office', 'police', 'fire_station', 'townhall', 'courthouse', 'waste_disposal',
        'waste_basket', 'social_facility', 'shelter', 'community_centre', 'public_bookcase',
        'animal_shelter', 'events_venue', 'prison', 'archive', 'post_depot', 'mortuary',
        'public_building', 'crematorium', 'payment_centre', 'meeting_room', 'reception_desk',
        'reception_desk;lost_property_office', 'lost_property_office', 'group_home', 'dormitory'
    ],
    'finance': [
        'bank', 'atm', 'bureau_de_change', 'money_transfer'
    ],
    'food_and_drink': [
        'restaurant', 'cafe', 'fast_food', 'pub', 'bar', 'ice_cream', 'biergarten', 'food_court',
        'canteen', 'restaurant;cafe'
    ],
    'utilities': [
        'telephone', 'recycling', 'drinking_water', 'toilets', 'vending_machine', 'charging_station',
        'compressed_air', 'sanitary_dump_station', 'water_point', 'bicycle_repair_station',
        'foot_shower', 'vacuum_cleaner', 'parcel_locker', 'fixme'
    ],
    'education': [
        'library', 'school', 'kindergarten', 'college', 'university', 'music_school', 'driving_school',
        'childcare', 'research_institute', 'prep_school', 'language_school', 'training',
        'dancing_school'
    ],
    'entertainment_and_recreation': [
        'cinema', 'theatre', 'nightclub', 'club', 'studio', 'events_venue', 'gambling', 'dojo',
        'hookah_lounge', 'stripclub', 'concert_hall', 'planetarium', 'exhibition_centre',
        'love_hotel', 'public_bath', 'watering_place', 'stage', 'auditorium'
    ],
    'shopping': [
        'marketplace', 'shop', 'mall', 'supermarket'
    ]
}


# Map amenity classes to activity labels used in our system
AMENITY_TO_ACTIVITY = {
    'transportation': 'transit',
    'healthcare': 'health',
    'public_services': 'admin',
    'finance': 'finance',
    'food_and_drink': 'eat',
    'utilities': 'utility',
    'education': 'school',
    'entertainment_and_recreation': 'leisure',
    'shopping': 'shop'
}


def build_activity_node_pools(G, proximity_m: float = 200, cache_dir: str = "data", city_name: str = "porto") -> Dict[str, List[int]]:
    """
    Build activity-aware candidate node pools based on OSM amenities.
    
    For each activity type, finds nodes near relevant amenity features.
    Special handling for home (residential areas) and work (commercial/office/industrial).
    
    Args:
        G: OSMnx graph
        proximity_m: Maximum distance in meters from amenity to consider a node
        cache_dir: Directory to cache intermediate amenity GeoDataFrames
        city_name: Name of the city for cache file naming
        
    Returns:
        Dictionary mapping activity labels to lists of candidate OSM node IDs
    """
    import os
    import pickle
    
    print(f"  Building activity-aware node pools (proximity={proximity_m}m)...")
    
    # Get graph bounding box
    nodes_gdf = ox.graph_to_gdfs(G, nodes=True, edges=False)
    bbox = nodes_gdf.total_bounds  # [minx, miny, maxx, maxy]
    
    # Initialize pools with all nodes (fallback for activities without specific features)
    all_nodes = list(G.nodes())
    activity_pools: Dict[str, set] = defaultdict(set)
    
    # Cache paths for amenity GeoDataFrames
    os.makedirs(cache_dir, exist_ok=True)
    amenities_cache = os.path.join(cache_dir, f"{city_name}_amenities.pkl")
    amenities_geojson = os.path.join(cache_dir, f"{city_name}_amenities.geojson")
    
    # MINIMAL AMENITY QUERY: Only essential types for fast prototyping
    print("    Will query minimal essential amenity types (fast mode)")
    
    # Query amenities from OSM (with caching)
    try:
        if os.path.exists(amenities_cache):
            print("    Loading cached amenities...")
            with open(amenities_cache, 'rb') as f:
                gdf_amenities = pickle.load(f)
            print(f"    Loaded {len(gdf_amenities)} amenity features from cache")
        else:
            # MINIMAL QUERY: Only essential amenity types
            print("    Querying OSM amenities (minimal set)...")
            tags = {
                'amenity': [
                    'restaurant', 'cafe', 'bar',        # eat
                    'school', 'university',             # school
                    'hospital', 'clinic', 'doctors',    # health
                    'supermarket',                      # shop
                    'office', 'bank',                   # work/finance
                ]
            }
            gdf_amenities = ox.features.features_from_bbox(bbox=bbox, tags=tags)
            print(f"    Retrieved {len(gdf_amenities)} amenity features")
            
            # OLD CODE: Queries from AMENITY_CLASSES (111 types - too slow)
            # needed_amenity_types = []
            # for amenity_class, amenity_list in AMENITY_CLASSES.items():
            #     needed_amenity_types.extend(amenity_list)
            # needed_amenity_types = list(set(needed_amenity_types))
            # print(f"    Will query {len(needed_amenity_types)} specific amenity types")
            # tags = {'amenity': needed_amenity_types}
            # gdf_amenities = ox.features.features_from_bbox(bbox=bbox, tags=tags)
            
            # OLD CODE: Queries ALL amenities (27K+ features - very slow)
            # print("    Querying OSM amenities...")
            # tags = {'amenity': True}
            # gdf_amenities = ox.features.features_from_bbox(bbox=bbox, tags=tags)
            # print(f"    Retrieved {len(gdf_amenities)} amenity features")
            
            # Save to cache (pickle for speed)
            with open(amenities_cache, 'wb') as f:
                pickle.dump(gdf_amenities, f)
            print(f"    Cached amenities to {amenities_cache}")
            
            # Also save as GeoJSON for portability/transfer to other computers
            try:
                gdf_amenities.to_file(amenities_geojson, driver='GeoJSON')
                print(f"    Saved portable GeoJSON to {amenities_geojson}")
            except Exception as e:
                print(f"    ⚠ Could not save GeoJSON: {e}")
        
        # Build reverse mapping: amenity type -> activity label
        amenity_type_to_activity = {}
        for amenity_class, amenity_list in AMENITY_CLASSES.items():
            activity_label = AMENITY_TO_ACTIVITY.get(amenity_class, 'unknown')
            for amenity_type in amenity_list:
                amenity_type_to_activity[amenity_type] = activity_label
        
        # For each amenity, find nearby nodes (PARALLELIZED)
        print(f"    Processing {len(gdf_amenities)} amenities to find nearby nodes (parallelized)...")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import multiprocessing as mp
        
        # Prepare data for parallel processing
        amenity_rows = list(gdf_amenities.iterrows())
        
        # Use ThreadPoolExecutor (G is not pickleable for ProcessPoolExecutor)
        max_workers = min(mp.cpu_count(), 8)  # Limit to 8 threads to avoid overhead
        
        def process_amenity_thread(amenity_row):
            """Process amenity in thread (can access G directly)."""
            idx, amenity = amenity_row
            amenity_type = amenity.get('amenity')
            if not amenity_type:
                return None
            
            activity_label = amenity_type_to_activity.get(amenity_type, 'unknown')
            if activity_label == 'unknown':
                return None
            
            # Get amenity centroid
            try:
                geom = amenity['geometry']
                centroid = geom.centroid
                amenity_lat, amenity_lon = centroid.y, centroid.x
            except Exception:
                return None
            
            # Find nodes within proximity
            nearby_nodes_found = set()
            try:
                nearby_nodes = ox.nearest_nodes(G, amenity_lon, amenity_lat, return_dist=False)
                if isinstance(nearby_nodes, (list, tuple)):
                    for node in nearby_nodes:
                        node_data = G.nodes[node]
                        dist = ox.distance.great_circle(amenity_lat, amenity_lon, node_data['y'], node_data['x'])
                        if dist <= proximity_m:
                            nearby_nodes_found.add(node)
                else:
                    # Single node
                    node_data = G.nodes[nearby_nodes]
                    dist = ox.distance.great_circle(amenity_lat, amenity_lon, node_data['y'], node_data['x'])
                    if dist <= proximity_m:
                        nearby_nodes_found.add(nearby_nodes)
            except Exception:
                pass
            
            return (activity_label, nearby_nodes_found)
        
        # Process in parallel with progress tracking
        processed = 0
        last_progress = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_amenity_thread, row): row for row in amenity_rows}
            
            for future in as_completed(futures):
                result = future.result()
                if result:
                    activity_label, nearby_nodes = result
                    for node in nearby_nodes:
                        activity_pools[activity_label].add(node)
                
                # Progress indicator (every 10%)
                processed += 1
                progress = int((processed / len(gdf_amenities)) * 100)
                if progress >= last_progress + 10:
                    print(f"      {progress}% complete ({processed}/{len(gdf_amenities)} amenities)")
                    last_progress = progress
        
        print(f"    ✓ Processed all {len(gdf_amenities)} amenities (parallelized with {max_workers} threads)")
                
    except Exception as e:
        print(f"    ⚠ Failed to query amenities: {e}")
    
    # Query landuse for home and work (with caching)
    residential_cache = os.path.join(cache_dir, f"{city_name}_residential.pkl")
    residential_geojson = os.path.join(cache_dir, f"{city_name}_residential.geojson")
    work_cache = os.path.join(cache_dir, f"{city_name}_work.pkl")
    work_geojson = os.path.join(cache_dir, f"{city_name}_work.geojson")
    
    try:
        # Home: residential areas
        if os.path.exists(residential_cache):
            print("    Loading cached residential areas...")
            with open(residential_cache, 'rb') as f:
                gdf_residential = pickle.load(f)
            print(f"    Loaded {len(gdf_residential)} residential areas from cache")
        else:
            print("    Querying OSM landuse for home (residential)...")
            tags_residential = {'landuse': 'residential'}
            gdf_residential = ox.features.features_from_bbox(bbox=bbox, tags=tags_residential)
            print(f"    Retrieved {len(gdf_residential)} residential areas")
            with open(residential_cache, 'wb') as f:
                pickle.dump(gdf_residential, f)
            print(f"    Cached residential areas")
            
            # Also save as GeoJSON for portability
            try:
                gdf_residential.to_file(residential_geojson, driver='GeoJSON')
                print(f"    Saved portable GeoJSON to {residential_geojson}")
            except Exception as e:
                print(f"    ⚠ Could not save GeoJSON: {e}")
        
        print(f"    Processing {len(gdf_residential)} residential areas...")
        total_residential = len(gdf_residential)
        for i, (idx, area) in enumerate(gdf_residential.iterrows(), 1):
            try:
                geom = area['geometry']
                centroid = geom.centroid
                lat, lon = centroid.y, centroid.x
                nearby_nodes = ox.nearest_nodes(G, lon, lat, return_dist=False)
                if isinstance(nearby_nodes, (list, tuple)):
                    for node in nearby_nodes:
                        node_data = G.nodes[node]
                        dist = ox.distance.great_circle(lat, lon, node_data['y'], node_data['x'])
                        if dist <= proximity_m * 2:  # Larger radius for residential
                            activity_pools['home'].add(node)
                else:
                    node_data = G.nodes[nearby_nodes]
                    dist = ox.distance.great_circle(lat, lon, node_data['y'], node_data['x'])
                    if dist <= proximity_m * 2:
                        activity_pools['home'].add(nearby_nodes)
            except Exception:
                continue
            
            # Progress indicator every 10%
            if i % max(1, total_residential // 10) == 0 or i == total_residential:
                print(f"      {int(100 * i / total_residential)}% complete ({i}/{total_residential} residential areas)")
        print(f"    ✓ Processed residential areas")
                
    except Exception as e:
        print(f"    ⚠ Failed to query residential landuse: {e}")
    
    # Work: commercial, industrial, office areas
    try:
        if os.path.exists(work_cache):
            print("    Loading cached work areas...")
            with open(work_cache, 'rb') as f:
                gdf_work = pickle.load(f)
            print(f"    Loaded {len(gdf_work)} work-related areas from cache")
        else:
            print("    Querying OSM landuse for work (commercial/industrial/retail)...")
            tags_work = {'landuse': ['commercial', 'industrial', 'retail']}
            gdf_work = ox.features.features_from_bbox(bbox=bbox, tags=tags_work)
            print(f"    Retrieved {len(gdf_work)} work-related areas")
            with open(work_cache, 'wb') as f:
                pickle.dump(gdf_work, f)
            print(f"    Cached work areas")
            
            # Also save as GeoJSON for portability
            try:
                gdf_work.to_file(work_geojson, driver='GeoJSON')
                print(f"    Saved portable GeoJSON to {work_geojson}")
            except Exception as e:
                print(f"    ⚠ Could not save GeoJSON: {e}")
        
        print(f"    Processing {len(gdf_work)} work areas...")
        total_work = len(gdf_work)
        for i, (idx, area) in enumerate(gdf_work.iterrows(), 1):
            try:
                geom = area['geometry']
                centroid = geom.centroid
                lat, lon = centroid.y, centroid.x
                nearby_nodes = ox.nearest_nodes(G, lon, lat, return_dist=False)
                if isinstance(nearby_nodes, (list, tuple)):
                    for node in nearby_nodes:
                        node_data = G.nodes[node]
                        dist = ox.distance.great_circle(lat, lon, node_data['y'], node_data['x'])
                        if dist <= proximity_m * 2:
                            activity_pools['work'].add(node)
                else:
                    node_data = G.nodes[nearby_nodes]
                    dist = ox.distance.great_circle(lat, lon, node_data['y'], node_data['x'])
                    if dist <= proximity_m * 2:
                        activity_pools['work'].add(nearby_nodes)
            except Exception:
                continue
            
            # Progress indicator every 10%
            if i % max(1, total_work // 10) == 0 or i == total_work:
                print(f"      {int(100 * i / total_work)}% complete ({i}/{total_work} work areas)")
        print(f"    ✓ Processed work areas")
                
    except Exception as e:
        print(f"    ⚠ Failed to query work landuse: {e}")
    
    # Convert sets to lists and add fallback to all nodes for activities with no candidates
    result: Dict[str, List[int]] = {}
    activity_labels = ['home', 'work', 'eat', 'school', 'health', 'shop', 'leisure', 
                      'transit', 'utility', 'admin', 'finance', 'unknown']
    
    for activity in activity_labels:
        if activity in activity_pools and len(activity_pools[activity]) > 0:
            result[activity] = list(activity_pools[activity])
        else:
            # Fallback: use all nodes
            result[activity] = all_nodes.copy()
        print(f"    {activity}: {len(result[activity])} candidate nodes")
    
    return result


def sample_from_activity_pool(activity_label: str, activity_pools: Dict[str, List[int]], 
                              n: int = 1, all_nodes: Optional[List[int]] = None) -> List[int]:
    """
    Sample n nodes from the activity-specific candidate pool.
    
    Args:
        activity_label: Activity type (home, work, eat, etc.)
        activity_pools: Dictionary of activity -> candidate nodes
        n: Number of nodes to sample
        all_nodes: Fallback list of all nodes if activity pool is empty
        
    Returns:
        List of sampled node IDs
    """
    pool = activity_pools.get(activity_label, [])
    
    # Fallback to all nodes if pool is empty
    if not pool:
        if all_nodes:
            pool = all_nodes
        else:
            return []
    
    # Sample with replacement if n > pool size
    if n >= len(pool):
        return list(np.random.choice(pool, size=n, replace=True))
    else:
        return list(np.random.choice(pool, size=n, replace=False))

