from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from ..spatial.mapping import create_trajectory
from ..spatial.activity_pools import sample_from_activity_pool


def simulate_synthetic_trips(
    imn: Dict,
    synthetic_stays: List[Tuple[str, int, int]],
    G,
    gdf_cumulative_p,
    randomness: float = 0.5,
    fixed_home_node: Optional[int] = None,
    fixed_work_node: Optional[int] = None,
    precomputed_map_loc_rmse: Optional[Tuple[Dict[int, int], float]] = None,
    activity_pools: Optional[Dict[str, List[int]]] = None,
    user_id: Optional[int] = None,
) -> Tuple[List[Tuple[float, float, int]], Dict[Any, int], Dict[str, int], float, List[List[Tuple[float, float, int]]]]:
    """
    Simulate spatial trips for synthetic stays using precomputed IMN→OSM mapping.
    
    Spatializes a synthetic temporal timeline by assigning OSM nodes to each stay
    and connecting them via real road-network trajectories.
    
    Args:
        imn: Individual Mobility Network
        synthetic_stays: List of (activity_label, start_rel, end_rel) tuples
        G: OSM graph
        gdf_cumulative_p: Population cumulative probability (unused, kept for compatibility)
        randomness: Randomness level (used as random seed for reproducibility)
        fixed_home_node: Optional fixed home node
        fixed_work_node: Optional fixed work node
        precomputed_map_loc_rmse: Required precomputed IMN→OSM mapping (map_loc_imn, rmse)
        activity_pools: Optional activity-aware candidate node pools for new activities
        
    Returns:
        Tuple of (trajectory, osm_segments_usage, pseudo_map, rmse, legs_coords)
    """
    if not synthetic_stays:
        return None, None, None, None, None
    
    # Use precomputed mapping (required)
    if precomputed_map_loc_rmse is None:
        raise ValueError("precomputed_map_loc_rmse must be provided")
    
    map_loc_imn, rmse = precomputed_map_loc_rmse
    
    # Set random seed for reproducibility - use user_id to ensure uniqueness
    # This ensures each user gets different node selections even with same randomness level
    if randomness is not None:
        # Combine randomness level with user_id for unique seed per user
        # If user_id not provided, try to get from IMN, otherwise use hash of home/work coords
        if user_id is None:
            user_id = imn.get('uid', imn.get('user_id', 0))
            if user_id == 0:
                # Fallback: use hash of home/work coordinates for uniqueness
                try:
                    home_coords = tuple(imn['locations'][imn['home']]['coordinates'])
                    work_coords = tuple(imn['locations'][imn['work']]['coordinates'])
                    user_id = hash((home_coords, work_coords)) % 1000000
                except:
                    user_id = 0
        unique_seed = int(randomness * 10000) + user_id
        np.random.seed(unique_seed)
    
    # Get fixed home and work nodes
    home_node = fixed_home_node if fixed_home_node is not None else map_loc_imn.get(imn['home'])
    work_node = fixed_work_node if fixed_work_node is not None else map_loc_imn.get(imn['work'])
    all_nodes = list(G.nodes())
    
    # Assign OSM nodes to each stay
    stay_nodes: List[int] = []
    for idx, (act, st, et) in enumerate(synthetic_stays):
        act_label = str(act).lower() if act is not None else "unknown"
        
        # Priority 1: Home activity → use fixed_home_node
        if act_label == 'home' and home_node is not None:
            stay_nodes.append(home_node)
        # Priority 2: Work activity → use fixed_work_node
        elif act_label == 'work' and work_node is not None:
            stay_nodes.append(work_node)
        # Priority 3: Check if activity corresponds to an existing IMN location
        elif act in imn.get('locations', {}):
            # Activity is an IMN location ID → use the precomputed mapping
            mapped_node = map_loc_imn.get(act)
            if mapped_node is not None:
                stay_nodes.append(mapped_node)
            else:
                # Shouldn't happen if precomputed mapping is complete, but fallback
                stay_nodes.append(all_nodes[np.random.randint(len(all_nodes))])
        else:
            # Priority 4: New activity not in IMN → sample once from activity pools
            if activity_pools is not None:
                sampled = sample_from_activity_pool(act_label, activity_pools, n=1, all_nodes=all_nodes)
                if sampled:
                    stay_nodes.append(sampled[0])
                else:
                    stay_nodes.append(all_nodes[np.random.randint(len(all_nodes))])
            else:
                # No activity pools available → random fallback
                stay_nodes.append(all_nodes[np.random.randint(len(all_nodes))])
    
    # Create mutable copy of stays for local time adjustments
    stays_mut = [[act, int(st), int(et)] for (act, st, et) in synthetic_stays]
    
    # Build trajectories between consecutive stays
    trajectory: List[Tuple[float, float, int]] = []
    osm_segments_usage: Dict[Any, int] = {}
    legs_coords: List[List[Tuple[float, float, int]]] = []
    
    for i in range(len(stays_mut) - 1):
        act_from, st_from, et_from = stays_mut[i]
        act_to, st_to, et_to = stays_mut[i + 1]
        origin_node = stay_nodes[i]
        dest_node = stay_nodes[i + 1]
        
        # Compute available gap between stays
        allocated_gap = max(0, st_to - et_from)
        departure_time = et_from  # Trip starts when previous stay ends
        
        # Create trajectory on road network
        shortest_p, duration_p, osm_segments = create_trajectory(
            G, origin_node, dest_node, departure_time, use_cache=True
        )
        
        if shortest_p is None:
            continue
        
        # Handle travel time adjustments locally
        if duration_p > allocated_gap:
            # Travel time exceeds gap → shorten previous stay's end time
            excess = int(duration_p - allocated_gap)
            stays_mut[i][2] = max(st_from, et_from - excess)
            departure_time = stays_mut[i][2]  # New departure time (end of shortened stay)
            
            # Compute arrival time: arrival_time = departure_time + duration_p
            arrival_time = departure_time + duration_p
            stays_mut[i + 1][1] = arrival_time  # Update next stay's start to arrival time
            
            # Recompute trajectory with adjusted departure time
            shortest_p, duration_p, osm_segments = create_trajectory(
                G, origin_node, dest_node, departure_time, use_cache=True
            )
            
            if shortest_p is None:
                continue
        
        # Add trajectory leg
        trajectory.extend(shortest_p)
        legs_coords.append(shortest_p)
        
        # Track OSM segment usage
        for oss in osm_segments:
            osm_segments_usage[oss] = osm_segments_usage.get(oss, 0) + 1
    
    # Create pseudo_map: stay index → OSM node ID
    pseudo_map = {i: node for i, node in enumerate(stay_nodes)}
    
    return trajectory, osm_segments_usage, pseudo_map, rmse, legs_coords


