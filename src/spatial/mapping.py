from typing import Dict, Tuple, Any, Optional, List
import numpy as np
import networkx as nx
import osmnx as ox

osm_paths_cache: Dict[Tuple[int, int], Optional[List[int]]] = {}


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


def map_imn_to_osm(imn, target_osm, n_trials=10, gdf_cumulative_p=None):
    from pandas import DataFrame
    nodes = list(target_osm.nodes())
    if gdf_cumulative_p is None:
        gdf_cumulative_p = DataFrame({"cumulative_p": [1.0], "intersecting_nodes": [nodes]})
    imn_dist = haversine_distance(
        imn['locations'][imn['home']]['coordinates'][0],
        imn['locations'][imn['home']]['coordinates'][1],
        imn['locations'][imn['work']]['coordinates'][0],
        imn['locations'][imn['work']]['coordinates'][1],
    )
    best_dist = 9e18
    for _ in range(n_trials):
        tmp_home_osm = select_random_nodes(gdf_cumulative_p, n=1, all_nodes=nodes)[0]
        tmp_work_osm = select_random_nodes(gdf_cumulative_p, n=1, all_nodes=nodes)[0]
        dist = haversine_distance(
            target_osm.nodes[tmp_home_osm]['x'],
            target_osm.nodes[tmp_home_osm]['y'],
            target_osm.nodes[tmp_work_osm]['x'],
            target_osm.nodes[tmp_work_osm]['y'],
        )
        if abs(dist - imn_dist) < abs(best_dist - imn_dist):
            best_home = tmp_home_osm
            best_work = tmp_work_osm
            best_dist = dist
    map_loc = {imn['home']: best_home, imn['work']: best_work}
    SE = (best_dist / 1000 - imn_dist / 1000) ** 2
    for loc in imn['locations'].keys():
        if loc in map_loc:
            continue
        best_dist_loc = 9e18
        for _ in range(n_trials):
            tmp_osm = select_random_nodes(gdf_cumulative_p, n=1, all_nodes=nodes)[0]
            dist_sum = 0
            for imn_l, osm_l in map_loc.items():
                dist_osm = haversine_distance(
                    target_osm.nodes[tmp_osm]['x'],
                    target_osm.nodes[tmp_osm]['y'],
                    target_osm.nodes[osm_l]['x'],
                    target_osm.nodes[osm_l]['y'],
                )
                dist_imn = haversine_distance(
                    imn['locations'][loc]['coordinates'][0],
                    imn['locations'][loc]['coordinates'][1],
                    imn['locations'][imn_l]['coordinates'][0],
                    imn['locations'][imn_l]['coordinates'][1],
                )
                dist_sum += (dist_osm/1000 - dist_imn/1000)**2
            if dist_sum < best_dist_loc:
                best_osm = tmp_osm
                best_dist_loc = dist_sum
        map_loc[loc] = best_osm
        SE += best_dist_loc
    if len(map_loc) > 1:
        rmse = np.sqrt(SE/(len(map_loc)*(len(map_loc)-1)/2))
    else:
        rmse = 0
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


