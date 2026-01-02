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
    original_stays: Optional[List] = None,
    global_used_locations: Optional[Dict[str, set]] = None,
    global_diversity_requirements: Optional[Dict[str, int]] = None,
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
    # Skip days with only one stay - no trips can be generated (need at least 2 stays for a trip)
    # Check early to avoid unnecessary node assignment
    if len(synthetic_stays) < 2:
        return None, {}, {}, 0.0, []
    
    all_nodes = list(G.nodes())
    
    # Use global diversity requirements (across all days) if provided, otherwise fall back to per-day
    # Global diversity ensures we use ALL mapped locations over the entire timeline
    if global_diversity_requirements is not None:
        diversity_requirements = global_diversity_requirements
    else:
        # Fallback: calculate from original stays for this day (per-day diversity)
        diversity_requirements: Dict[str, int] = {}
        if original_stays is not None:
            from collections import defaultdict
            original_locations_by_activity = defaultdict(set)
            for stay in original_stays:
                if hasattr(stay, 'location_id') and hasattr(stay, 'activity_label'):
                    act = str(stay.activity_label).lower()
                    original_locations_by_activity[act].add(stay.location_id)
            for act, loc_set in original_locations_by_activity.items():
                diversity_requirements[act] = len(loc_set)
    
    # Use global tracking if provided (across all days), otherwise use local (per-day)
    if global_used_locations is not None:
        used_locations_per_activity = global_used_locations
    else:
        used_locations_per_activity: Dict[str, set] = {}
    
    required_diversity_per_activity: Dict[str, int] = {}  # How many distinct locations we still need
    
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
            # Priority 4: Activity label (not location ID) → use frequency-weighted sampling from mapped locations
            # First, check if user has mapped locations with this activity type
            user_mapped_locations_for_activity = []
            for loc_id, loc_data in imn.get('locations', {}).items():
                loc_activity = str(loc_data.get('activity_label', 'unknown')).lower()
                if loc_activity == act_label and loc_id in map_loc_imn:
                    # This location is mapped and has the matching activity
                    frequency = loc_data.get('frequency', 0)
                    mapped_osm_node = map_loc_imn[loc_id]
                    user_mapped_locations_for_activity.append((mapped_osm_node, frequency))
            
            if user_mapped_locations_for_activity:
                # User has mapped locations for this activity → use diversity-aware frequency-weighted sampling
                nodes_list = [node for node, _ in user_mapped_locations_for_activity]
                frequencies = [freq for _, freq in user_mapped_locations_for_activity]
                
                # Initialize tracking for this activity if needed
                if act_label not in used_locations_per_activity:
                    used_locations_per_activity[act_label] = set()
                # Ensure diversity requirement is set (may have been set on a previous day)
                if act_label not in required_diversity_per_activity:
                    # Set diversity requirement: use at least as many distinct locations as original
                    required_diversity_per_activity[act_label] = diversity_requirements.get(act_label, 0)
                
                # Build candidate pools
                unused_nodes = [node for node in nodes_list if node not in used_locations_per_activity[act_label]]
                used_nodes = [node for node in nodes_list if node in used_locations_per_activity[act_label]]
                
                # Check diversity status
                current_distinct_count = len(used_locations_per_activity[act_label])
                required_diversity = required_diversity_per_activity[act_label]
                remaining_diversity = required_diversity - current_distinct_count
                
                # Option 1: Soft constraint with rejection sampling
                # Try natural frequency-weighted sampling, but enforce constraints
                max_attempts = 10  # Prevent infinite loops
                selected_node = None
                
                for attempt in range(max_attempts):
                    # Sample with frequency weighting from all locations
                    total_freq = sum(frequencies)
                    if total_freq > 0:
                        probabilities = [f / total_freq for f in frequencies]
                        candidate_node = np.random.choice(nodes_list, p=probabilities)
                    else:
                        candidate_node = np.random.choice(nodes_list)
                    
                    # Check if candidate violates diversity constraints
                    is_new_distinct = candidate_node not in used_locations_per_activity[act_label]
                    
                    if remaining_diversity > 0:
                        # Below minimum: must select unused location
                        if is_new_distinct:
                            selected_node = candidate_node
                            break
                        # Reject: resample (will try again)
                    elif remaining_diversity == 0:
                        # At maximum: must reuse already-used location
                        if not is_new_distinct:
                            selected_node = candidate_node
                            break
                        # Reject: resample from used locations only
                        if used_nodes:
                            used_indices = [i for i, node in enumerate(nodes_list) if node in used_nodes]
                            used_frequencies = [frequencies[i] for i in used_indices]
                            total_used_freq = sum(used_frequencies)
                            if total_used_freq > 0:
                                used_probs = [f / total_used_freq for f in used_frequencies]
                                selected_node = np.random.choice(used_nodes, p=used_probs)
                            else:
                                selected_node = np.random.choice(used_nodes)
                            break
                    else:
                        # Above maximum (shouldn't happen): force reuse
                        if not is_new_distinct:
                            selected_node = candidate_node
                            break
                        if used_nodes:
                            selected_node = np.random.choice(used_nodes)
                            break
                
                # Fallback if all attempts failed
                if selected_node is None:
                    if remaining_diversity > 0 and unused_nodes:
                        selected_node = np.random.choice(unused_nodes)
                    elif used_nodes:
                        selected_node = np.random.choice(used_nodes)
                    else:
                        selected_node = np.random.choice(nodes_list)
                
                # Track that we've used this location
                used_locations_per_activity[act_label].add(selected_node)
                stay_nodes.append(selected_node)
            else:
                # No mapped locations for this activity → fall back to city-wide activity pool
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


