from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from .timelines import find_anchor_stay_for_day
from ..spatial.mapping import create_trajectory, map_imn_to_osm, select_random_nodes
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
) -> Tuple[List[Tuple[float, float, int]], Dict[Any, int], Dict[str, int], float, List[List[Tuple[float, float, int]]]]:
    """
    Simulate spatial trips for synthetic stays using activity-aware node pools.
    
    Args:
        imn: Individual Mobility Network
        synthetic_stays: List of (activity_label, start_rel, end_rel) tuples
        G: OSM graph
        gdf_cumulative_p: Population cumulative probability (fallback if activity_pools not provided)
        randomness: Randomness level
        fixed_home_node: Optional fixed home node
        fixed_work_node: Optional fixed work node
        precomputed_map_loc_rmse: Optional precomputed IMN->OSM mapping
        activity_pools: Optional activity-aware candidate node pools
        
    Returns:
        Tuple of (trajectory, osm_segments_usage, pseudo_map, rmse, legs_coords)
    """
    if not synthetic_stays:
        return None, None, None, None, None
    if precomputed_map_loc_rmse is not None:
        map_loc_imn, rmse = precomputed_map_loc_rmse
    else:
        map_loc_imn, rmse = map_imn_to_osm(imn, G, gdf_cumulative_p=gdf_cumulative_p)
    home_node = fixed_home_node if fixed_home_node is not None else map_loc_imn.get(imn['home'])
    work_node = fixed_work_node if fixed_work_node is not None else map_loc_imn.get(imn['work'])
    all_nodes = list(G.nodes())
    stay_nodes: List[int] = []
    for idx, (act, st, et) in enumerate(synthetic_stays):
        act_label = str(act).lower() if act is not None else "unknown"
        if act_label == 'home' and home_node is not None:
            stay_nodes.append(home_node)
        elif act_label == 'work' and work_node is not None:
            stay_nodes.append(work_node)
        else:
            # Use activity-aware pools if available, otherwise fall back to population grid
            if activity_pools is not None:
                sampled = sample_from_activity_pool(act_label, activity_pools, n=1, all_nodes=all_nodes)
                if sampled:
                    stay_nodes.append(sampled[0])
                else:
                    stay_nodes.append(all_nodes[np.random.randint(len(all_nodes))])
            else:
                sampled = select_random_nodes(gdf_cumulative_p, n=1, all_nodes=all_nodes)
                if not sampled:
                    stay_nodes.append(all_nodes[np.random.randint(len(all_nodes))])
                else:
                    stay_nodes.append(sampled[0])
    first_label = str(synthetic_stays[0][0]).lower() if synthetic_stays and synthetic_stays[0][0] is not None else ""
    if len(stay_nodes) > 0 and first_label == 'home' and home_node is not None:
        stay_nodes[0] = home_node
    anchor_idx = None
    max_dur = -1
    for idx, (act, st, et) in enumerate(synthetic_stays):
        if act == 'home':
            continue
        dur = max(0, (et or 0) - (st or 0))
        if dur > max_dur:
            max_dur = dur
            anchor_idx = idx
    stays_mut = [ [act, int(st), int(e)] for (act, st, e) in synthetic_stays ]
    trajectory: List[Tuple[float, float, int]] = []
    osm_segments_usage: Dict[Any, int] = {}
    legs_coords: List[List[Tuple[float, float, int]]] = []
    for i in range(len(stays_mut) - 1):
        act_from, st_from, et_from = stays_mut[i]
        act_to, st_to, et_to = stays_mut[i + 1]
        origin_node = stay_nodes[i]
        dest_node = stay_nodes[i + 1]
        allocated_gap = max(0, st_to - et_from)
        trip_start_time = et_from
        shortest_p, duration_p, osm_segments = create_trajectory(G, origin_node, dest_node, trip_start_time, use_cache=True)
        if shortest_p is None:
            continue
        if duration_p > allocated_gap:
            delta = int(duration_p - allocated_gap)
            if i == anchor_idx:
                stays_mut[i + 1][1] += delta
            elif i + 1 == anchor_idx:
                stays_mut[i][2] = max(st_from, et_from - delta)
                trip_start_time = stays_mut[i][2]
            else:
                stays_mut[i][2] = max(st_from, et_from - delta)
                trip_start_time = stays_mut[i][2]
            shortest_p, duration_p, osm_segments = create_trajectory(G, origin_node, dest_node, trip_start_time, use_cache=True)
            if shortest_p is None:
                continue
        else:
            slack = int(allocated_gap - duration_p)
            trip_start_time = et_from + slack
            stays_mut[i][2] = trip_start_time
            shortest_p, duration_p, osm_segments = create_trajectory(G, origin_node, dest_node, trip_start_time, use_cache=True)
            if shortest_p is None:
                continue
        trajectory.extend(shortest_p)
        legs_coords.append(shortest_p)
        for oss in osm_segments:
            osm_segments_usage[oss] = osm_segments_usage.get(oss, 0) + 1
    pseudo_map = { i: node for i, node in enumerate(stay_nodes) }
    return trajectory, osm_segments_usage, pseudo_map, rmse, legs_coords


