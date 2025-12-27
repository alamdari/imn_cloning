from typing import Dict, Tuple, Any, Optional, List
import numpy as np
import networkx as nx
import osmnx as ox

osm_paths_cache: Dict[Tuple[int, int], Optional[List[int]]] = {}

# Optional spatial index support
try:
    from sklearn.neighbors import BallTree
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


def haversine_distance(x1, y1, x2, y2):
    R = 6371000
    lat_rad1 = np.radians(y1)
    lon_rad1 = np.radians(x1)
    lat_rad2 = np.radians(y2)
    lon_rad2 = np.radians(x2)
    return 2 * R * np.arcsin(np.sqrt(np.sin((lat_rad2 - lat_rad1) / 2) ** 2 + np.cos(lat_rad1) * np.cos(lat_rad2) * (np.sin((lon_rad2 - lon_rad1) / 2) ** 2)))


def create_trajectory(G, node_origin, node_destination, start_time, weight="travel_time", slow_factor=1, use_cache=True):
    global osm_paths_cache
    if node_origin == node_destination:
        return [
            (G.nodes[node_origin]["y"], G.nodes[node_origin]["x"], start_time),
            (G.nodes[node_destination]["y"], G.nodes[node_destination]["x"], start_time + 60),
        ], 60, []

    if use_cache:
        if (node_origin, node_destination) in osm_paths_cache:
            route = osm_paths_cache[(node_origin, node_destination)]
        else:
            try:
                route = nx.shortest_path(G, node_origin, node_destination, weight=weight)
            except nx.NetworkXNoPath:
                route = None
            osm_paths_cache[(node_origin, node_destination)] = route
    else:
        try:
            route = nx.shortest_path(G, node_origin, node_destination, weight=weight)
        except nx.NetworkXNoPath:
            route = None

    if route is None:
        return None, None, None

    gdf_info_routes = ox.routing.route_to_gdf(G, route)
    road_segments_osmid = list(gdf_info_routes[["osmid"]].index)
    gps_points = [(G.nodes[node]["y"], G.nodes[node]["x"]) for node in route]
    travel_time_edges = gdf_info_routes["travel_time"].values
    travel_time_guess_edges = travel_time_edges * slow_factor
    list_seconds = [0] + list(travel_time_guess_edges.cumsum())
    trajectory = [(gps[0], gps[1], time + start_time) for gps, time in zip(gps_points, list_seconds)]
    return trajectory, list_seconds[-1], road_segments_osmid


def map_imn_to_osm(imn, target_osm, n_trials=10, gdf_cumulative_p=None, activity_pools=None, random_seed=None):
    """
    Map IMN locations from source city to OSM nodes in target city.
    
    Uses activity-aware mapping to preserve both semantic meaning (activity types)
    and spatial structure (relative distances between locations).
    
    Args:
        imn: Individual Mobility Network with 'locations', 'home', 'work'
        target_osm: OSM graph of target city
        n_trials: Number of candidate samples per location
        gdf_cumulative_p: Population-weighted node sampling (fallback)
        activity_pools: Dict[activity_label -> List[osm_node_id]] for activity-aware sampling
        random_seed: Random seed for deterministic results
        
    Returns:
        (map_loc, rmse) where:
            map_loc: Dict[imn_location_id -> osm_node_id]
            rmse: Root mean square error of pairwise distance preservation (km)
    """
    from pandas import DataFrame
    
    # Set random seed for deterministic results
    # Use a combination of random_seed and a hash of IMN coordinates to ensure uniqueness
    if random_seed is not None:
        # Create a unique seed per user by combining random_seed with IMN home/work coordinates
        # This ensures different users get different mappings even with similar IMN structures
        home_coords = imn['locations'][imn['home']]['coordinates']
        work_coords = imn['locations'][imn['work']]['coordinates']
        coord_hash = hash((tuple(home_coords), tuple(work_coords))) % 1000000
        unique_seed = (random_seed * 1000000 + coord_hash) % (2**31)
        np.random.seed(unique_seed)
    
    nodes = list(target_osm.nodes())
    if gdf_cumulative_p is None:
        gdf_cumulative_p = DataFrame({"cumulative_p": [1.0], "intersecting_nodes": [nodes]})
    
    # Helper: get candidate nodes for an activity
    def get_activity_candidates(activity_label: str, n: int) -> List[int]:
        """Sample n candidate nodes for given activity type."""
        if activity_pools and activity_label in activity_pools:
            pool = activity_pools[activity_label]
            if pool:
                # Sample with replacement if needed
                return [pool[i] for i in np.random.choice(len(pool), size=min(n, len(pool)), replace=False)]
        # Fallback to population-weighted sampling
        return select_random_nodes(gdf_cumulative_p, n=n, all_nodes=nodes)
    
    # Helper: compute squared distance error between two location pairs
    def compute_distance_error(imn_loc1, imn_loc2, osm_node1, osm_node2) -> float:
        """Compute squared error (km²) between IMN distance and OSM distance."""
        dist_imn = haversine_distance(
            imn['locations'][imn_loc1]['coordinates'][0],
            imn['locations'][imn_loc1]['coordinates'][1],
            imn['locations'][imn_loc2]['coordinates'][0],
            imn['locations'][imn_loc2]['coordinates'][1],
        ) / 1000  # Convert to km
        
        dist_osm = haversine_distance(
            target_osm.nodes[osm_node1]['x'],
            target_osm.nodes[osm_node1]['y'],
            target_osm.nodes[osm_node2]['x'],
            target_osm.nodes[osm_node2]['y'],
        ) / 1000  # Convert to km
        
        return (dist_osm - dist_imn) ** 2
    
    # Step 1: Map home and work as anchor points
    home_id = imn['home']
    work_id = imn['work']
    
    # Get home-work distance in source city
    imn_hw_dist = haversine_distance(
        imn['locations'][home_id]['coordinates'][0],
        imn['locations'][home_id]['coordinates'][1],
        imn['locations'][work_id]['coordinates'][0],
        imn['locations'][work_id]['coordinates'][1],
    )

    # Sample home candidates - use more candidates and shuffle to ensure diversity
    n_candidates = max(n_trials * 3, 30)  # Sample at least 30 candidates
    home_candidates = get_activity_candidates('home', n_candidates)
    np.random.shuffle(home_candidates)

    # Hybrid sampling for work candidates:
    # - Normal users: small work sample (fast)
    # - Risky users (very small IMN HW distance): larger work sample (better coverage)
    imn_hw_km = imn_hw_dist / 1000.0
    RISKY_DIST_KM = 1.0
    if imn_hw_km < RISKY_DIST_KM:
        n_work_candidates = 2000
    else:
        n_work_candidates = max(n_trials * 3, 30)
    work_candidates = get_activity_candidates('work', n_work_candidates)
    np.random.shuffle(work_candidates)

    # Helper to collect candidate home-work pairs for a given constraint range
    def collect_pairs(min_factor: float, max_factor: float) -> List[Tuple[float, int, int]]:
        local_pairs: List[Tuple[float, int, int]] = []
        if imn_hw_dist <= 0:
            return local_pairs
        min_allowed = min_factor * imn_hw_dist
        max_allowed = max_factor * imn_hw_dist if np.isfinite(max_factor) else None

        if HAS_SKLEARN and len(work_candidates) > 0 and max_allowed is not None:
            work_coords_rad = np.array([
                [np.radians(target_osm.nodes[n]['y']), np.radians(target_osm.nodes[n]['x'])]
                for n in work_candidates
            ])
            tree = BallTree(work_coords_rad, metric='haversine')
            earth_radius_m = 6371000.0

            for home_node in home_candidates[:n_trials * 2]:
                home_lat = target_osm.nodes[home_node]['y']
                home_lon = target_osm.nodes[home_node]['x']
                home_coord_rad = np.array([[np.radians(home_lat), np.radians(home_lon)]])

                max_radius_rad = max_allowed / earth_radius_m
                indices, distances_rad = tree.query_radius(
                    home_coord_rad,
                    r=max_radius_rad,
                    return_distance=True,
                )

                for idx, dist_rad in zip(indices[0], distances_rad[0]):
                    work_node = work_candidates[idx]
                    if home_node == work_node:
                        continue
                    dist_m = dist_rad * earth_radius_m
                    if dist_m < min_allowed:
                        continue

                    base_error = abs(dist_m - imn_hw_dist)
                    if dist_m > imn_hw_dist:
                        error = base_error * 2.0
                    else:
                        error = base_error
                    local_pairs.append((error, home_node, work_node))
        else:
            # Fallback to simple loop if sklearn/BallTree not available
            fallback_work_candidates = get_activity_candidates('work', n_candidates)
            np.random.shuffle(fallback_work_candidates)
            for home_node in home_candidates[:n_trials * 2]:
                for work_node in fallback_work_candidates[:n_trials * 2]:
                    if home_node == work_node:
                        continue
                    osm_hw_dist = haversine_distance(
                        target_osm.nodes[home_node]['x'],
                        target_osm.nodes[home_node]['y'],
                        target_osm.nodes[work_node]['x'],
                        target_osm.nodes[work_node]['y'],
                    )
                    if max_allowed is not None:
                        if osm_hw_dist < min_allowed or osm_hw_dist > max_allowed:
                            continue
                    else:
                        if osm_hw_dist < min_allowed:
                            continue

                    base_error = abs(osm_hw_dist - imn_hw_dist)
                    if osm_hw_dist > imn_hw_dist:
                        error = base_error * 2.0
                    else:
                        error = base_error
                    local_pairs.append((error, home_node, work_node))

        return local_pairs

    # Relaxed constraint strategy: try progressively looser ranges before fallback
    pair_errors: List[Tuple[float, int, int]] = []
    constraint_levels = [
        (0.5, 1.5),   # original window
        (0.25, 2.0),  # looser window
        (0.0, np.inf) # no constraint (except home != work)
    ]
    for min_f, max_f in constraint_levels:
        pair_errors = collect_pairs(min_f, max_f)
        if pair_errors:
            break

    # Fallback if still no pairs (very unlikely with relaxed constraints)
    if not pair_errors:
        best_home = home_candidates[0] if home_candidates else nodes[0]
        # Choose closest work node in terms of distance error (no constraint)
        if activity_pools and 'work' in activity_pools and len(activity_pools['work']) > 0:
            candidate_pool = activity_pools['work']
        else:
            candidate_pool = nodes
        best_work = None
        best_hw_error = float('inf')
        for work_node in candidate_pool:
            if work_node == best_home:
                continue
            d_osm = haversine_distance(
                target_osm.nodes[best_home]['x'],
                target_osm.nodes[best_home]['y'],
                target_osm.nodes[work_node]['x'],
                target_osm.nodes[work_node]['y'],
            )
            base_error = abs(d_osm - imn_hw_dist)
            error = base_error * 2.0 if d_osm > imn_hw_dist else base_error
            if error < best_hw_error:
                best_hw_error = error
                best_work = work_node
        if best_work is None:
            best_work = candidate_pool[0] if candidate_pool else nodes[0]
    else:
        pair_errors.sort(key=lambda x: x[0])
        if random_seed is not None:
            pair_errors = [
                (err * (1.0 + np.random.random() * 0.01), h, w)
                for err, h, w in pair_errors
            ]
            pair_errors.sort(key=lambda x: x[0])

        top_k = max(20, min(len(pair_errors) // 5, 100))
        top_k = min(top_k, len(pair_errors))
        choice_idx = np.random.randint(top_k)
        best_hw_error, best_home, best_work = pair_errors[choice_idx]
    
    # Initialize mapping with home and work
    map_loc = {home_id: best_home, work_id: best_work}
    
    # Track cumulative squared error for RMSE calculation
    SE = (best_hw_error / 1000) ** 2  # Convert to km and square
    
    # Step 2: Map remaining locations incrementally
    # Sort by frequency (visit more important locations first)
    remaining_locs = [
        (loc_id, loc_data) 
        for loc_id, loc_data in imn['locations'].items() 
        if loc_id not in map_loc
    ]
    remaining_locs.sort(key=lambda x: x[1].get('frequency', 0), reverse=True)
    
    for loc_id, loc_data in remaining_locs:
        activity_label = loc_data.get('activity_label', 'unknown')
        
        # Get candidates for this activity type
        candidates = get_activity_candidates(activity_label, n_trials)
        
        # Find best candidate that preserves distances to already mapped locations
        best_error = float('inf')
        best_node = candidates[0] if candidates else nodes[np.random.randint(len(nodes))]
        
        for candidate_node in candidates:
            # Compute total squared error against all already-mapped locations
            total_error = 0.0
            for mapped_loc_id, mapped_osm_node in map_loc.items():
                total_error += compute_distance_error(loc_id, mapped_loc_id, candidate_node, mapped_osm_node)
            
            if total_error < best_error:
                best_error = total_error
                best_node = candidate_node
        
        map_loc[loc_id] = best_node
        SE += best_error
    
    # Step 3: Compute RMSE
    # RMSE = sqrt(sum of squared errors / number of location pairs)
    num_locations = len(map_loc)
    if num_locations > 1:
        num_pairs = num_locations * (num_locations - 1) / 2
        rmse = np.sqrt(SE / num_pairs)
    else:
        rmse = 0.0
    
    return map_loc, rmse


def find_in_cumulative(df, p_r):
    import numpy as np
    idx = np.searchsorted(df['cumulative_p'], p_r, side='left')
    if idx < len(df):
        return df.iloc[idx]['intersecting_nodes']
    return None


def select_random_nodes(gdf_cumulative_p, n=10, all_nodes: Optional[List[int]] = None):
    import numpy as np
    node_list: List[int] = []
    for pr in np.random.random(n):
        candidates = find_in_cumulative(gdf_cumulative_p, pr)
        if candidates is None or len(candidates) == 0:
            if all_nodes:
                node_list.append(all_nodes[np.random.randint(len(all_nodes))])
            else:
                continue
        else:
            node_list.append(candidates[np.random.randint(len(candidates))])
    return node_list


