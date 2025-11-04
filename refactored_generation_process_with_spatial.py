#!/usr/bin/env python3
"""
Organized Generation Process for Individual Mobility Networks
Processes all users and generates synthetic timelines with organized output structure.
"""

import gzip
import json
import os
import random
import argparse
from dataclasses import dataclass
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pytz
import seaborn as sns
import pandas as pd

import imn_loading
import pickle
import requests
import networkx as nx
import osmnx as ox
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from shapely.geometry import Polygon
import folium

# Configuration
RANDOMNESS_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
TIMEZONE = pytz.timezone("Europe/Rome")

# Activity color mapping
ACTIVITY_COLORS = {
    "home": "skyblue",
    "work": "orange", 
    "eat": "green",
    "utility": "purple",
    "transit": "red",
    "unknown": "gray",
    "school": "yellow",
    "shop": "pink",
    "leisure": "lightgreen",
    "health": "lightcoral",
    "admin": "lightblue",
    "finance": "gold"
}

# POI to activity mapping
POI_TO_ACTIVITY = {
    "education": "school", 
    "food_and_drink": "eat", 
    "shopping": "shop",
    "entertainment_and_recreation": "leisure", 
    "transportation": "transit",
    "healthcare": "health", 
    "public_services": "admin", 
    "finance": "finance",
    "utilities": "utility", 
    "other": "unknown"
}


# ------------------------------
# Spatial constants/resources
# ------------------------------
PORTO_CENTER = (41.1494512, -8.6107884)
PORTO_RADIUS_M = 10000
SEDAC_TIFF_URL = "https://data.ghg.center/sedac-popdensity-yeargrid5yr-v4.11/gpw_v4_population_density_rev11_2020_30_sec_2020.tif"
DATA_DIR = "data"
PORTO_GRAPHML_PATH = os.path.join(DATA_DIR, "porto_drive.graphml")
SEDAC_TIFF_PATH = os.path.join(DATA_DIR, "gpw_v4_population_density_rev11_2020_30_sec_2020.tif")
POP_CUMULATIVE_P_PATH = os.path.join(DATA_DIR, "porto_population_cumulative.pkl")


class Stay:
    """Represents a stay at a location with activity and timing information."""
    
    def __init__(self, location_id: int, activity_label: str, start_time: int, end_time: int):
        self.location_id = location_id
        self.activity_label = activity_label
        self.start_time = start_time
        self.end_time = end_time
        self.duration = self.end_time - self.start_time if self.end_time is not None and self.start_time is not None else None

    def set_end_time(self, end_time: int):
        """Set the end time and recalculate duration."""
        self.end_time = end_time
        if self.start_time is not None:
            self.duration = self.end_time - self.start_time

    def to_dict(self) -> Dict[str, Any]:
        """Convert stay to dictionary representation."""
        return {
            'location_id': self.location_id,
            'activity_label': self.activity_label,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration
        }


# ------------------------------
# Spatial utilities (ported inline)
# ------------------------------

def haversine_distance(x1, y1, x2, y2):
    R = 6371000  # meters
    lat_rad1 = np.radians(y1)
    lon_rad1 = np.radians(x1)
    lat_rad2 = np.radians(y2)
    lon_rad2 = np.radians(x2)
    return 2 * R * np.arcsin(np.sqrt(np.sin((lat_rad2 - lat_rad1) / 2) ** 2 + np.cos(lat_rad1) * np.cos(lat_rad2) * (np.sin((lon_rad2 - lon_rad1) / 2) ** 2)))


def retrieve_osm(center, radius=20000):
    G = ox.graph.graph_from_point(center, dist=radius, network_type="drive", simplify=False)
    G = ox.routing.add_edge_speeds(G)
    G = ox.routing.add_edge_travel_times(G)
    return G


def create_trajectory(G, node_origin, node_destination, start_time, weight="travel_time", slow_factor=1, verbose=False, use_cache=True):
    global osm_paths_cache

    if verbose:
        print(f"Computing path: {node_origin} --> {node_destination}...")

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


def find_in_cumulative(df, p_r):
    idx = np.searchsorted(df['cumulative_p'], p_r, side='left')
    if idx < len(df):
        return df.iloc[idx]['intersecting_nodes']
    else:
        return None


def select_random_nodes(gdf_cumulative_p, n=10, all_nodes: Optional[List[int]] = None):
    node_list: List[int] = []
    for pr in np.random.random(n):
        candidates = find_in_cumulative(gdf_cumulative_p, pr)
        # Fallback if no intersecting nodes in the selected cell
        if candidates is None or len(candidates) == 0:
            print("   [warn] empty population cell -> fallback to random graph node")
            if all_nodes is not None and len(all_nodes) > 0:
                node_list.append(all_nodes[np.random.randint(len(all_nodes))])
            else:
                # As a last resort, skip; caller should handle empties
                continue
        else:
            node_list.append(candidates[np.random.randint(len(candidates))])
    return node_list


def map_imn_to_osm(imn, target_osm, home_osm=None, work_osm=None, n_trials=10, gdf_cumulative_p=None):
    best_dist = 999999999
    nodes = list(target_osm.nodes())
    if gdf_cumulative_p is None:
        gdf_cumulative_p = pd.DataFrame({"cumulative_p": [1.0], "intersecting_nodes": [nodes]})
    imn_dist = haversine_distance(
        imn['locations'][imn['home']]['coordinates'][0],
        imn['locations'][imn['home']]['coordinates'][1],
        imn['locations'][imn['work']]['coordinates'][0],
        imn['locations'][imn['work']]['coordinates'][1],
    )
    for _ in range(n_trials):
        tmp_home_osm = home_osm if home_osm is not None else select_random_nodes(gdf_cumulative_p, n=1, all_nodes=nodes)[0]
        tmp_work_osm = work_osm if work_osm is not None else select_random_nodes(gdf_cumulative_p, n=1, all_nodes=nodes)[0]
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
        best_dist_loc = 99999999999
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
                dist_sum += (dist_osm / 1000 - dist_imn / 1000) ** 2
            if dist_sum < best_dist_loc:
                best_osm = tmp_osm
                best_dist_loc = dist_sum
        map_loc[loc] = best_osm
        SE += best_dist_loc

    if len(map_loc) > 1:
        rmse = np.sqrt(SE / (len(map_loc) * (len(map_loc) - 1) / 2))
    else:
        rmse = 0
    print(f"Mapping {len(imn['locations'])} IMN locations → {len(map_loc)} OSM nodes, RMSE={rmse:.2f}")
    return map_loc, rmse


osm_paths_cache = {}


def init_osm_paths_cache(imn, G, map_loc, use_prefetch=False):
    global osm_paths_cache
    if use_prefetch:
        for _ in range(len(osm_paths_cache) - 500000):
            osm_paths_cache.popitem()
        map_loc_set = [map_loc[loc] for loc in imn['locations'].keys()]
        for loc in imn['locations'].keys():
            if imn['locations'][loc]['frequency'] > 10:
                node_origin = map_loc[loc]
                routes = nx.single_source_dijkstra_path(G, source=node_origin, cutoff=None, weight="travel_time")
                for dest, path in routes.items():
                    if dest in map_loc_set:
                        osm_paths_cache[(node_origin, dest)] = path
    else:
        osm_paths_cache = {}


def simulate_trips(
    imn,
    target_osm,
    start_sym=None,
    end_sym=None,
    home_osm=None,
    work_osm=None,
    n_trials=10,
    stay_time_error=0.2,
    gdf_cumulative_p=None,
    use_cache=True,
    use_prefetch=False,
):
    map_loc, rmse = map_imn_to_osm(
        imn,
        target_osm,
        home_osm=home_osm,
        work_osm=work_osm,
        n_trials=n_trials,
        gdf_cumulative_p=gdf_cumulative_p,
    )

    if start_sym is None:
        start_sym = min([r[2] for r in imn['trips']]) - 1
    if end_sym is None:
        end_sym = max([r[2] for r in imn['trips']]) + 1

    sym_time = None
    prev_end = None
    trajectory = []
    osm_segments_usage = {}
    if use_cache:
        init_osm_paths_cache(imn, target_osm, map_loc, use_prefetch=use_prefetch)
    for l_from, l_to, st, end in imn['trips']:
        if l_from == l_to:
            continue
        if (st < start_sym) or (st > end_sym):
            continue
        if sym_time is None:
            sym_time = st
            prev_end = st
        sym_stay = (st - prev_end) + random.randint(-int((st - prev_end) * stay_time_error), int((st - prev_end) * stay_time_error))
        sym_time += sym_stay
        prev_end = end

        shortest_p, duration_p, osm_segments = create_trajectory(target_osm, map_loc[l_from], map_loc[l_to], sym_time, use_cache=use_cache)
        if shortest_p is None:
            return None, None, None, None

        trajectory.extend(shortest_p)
        sym_time += duration_p
        for oss in osm_segments:
            osm_segments_usage[oss] = osm_segments_usage.get(oss, 0) + 1

    return trajectory, osm_segments_usage, map_loc, rmse


# ------------------------------
# Population utilities (ported inline)
# ------------------------------

def fetch_population_data_from_tiff(tiff_file, bbox):
    min_lon, min_lat, max_lon, max_lat = bbox
    with rasterio.open(tiff_file) as src:
        bbox_geom = {
            "type": "Polygon",
            "coordinates": [[
                (min_lon, min_lat), (max_lon, min_lat), (max_lon, max_lat),
                (min_lon, max_lat), (min_lon, min_lat)
            ]],
        }
        out_image, out_transform = mask(src, [bbox_geom], crop=True)
        out_image = out_image[0]
        population_data = []
        rows, cols = np.where(out_image > 0)
        for row, col in zip(rows, cols):
            population_density = out_image[row, col]
            x, y = out_transform * (col, row)
            x2, y2 = out_transform * (col + 1, row + 1)
            cell_polygon = Polygon([(x, y), (x2, y), (x2, y2), (x, y2), (x, y)])
            population_data.append({"population_density": float(population_density), "geometry": cell_polygon})
        population_gdf = gpd.GeoDataFrame(population_data, geometry="geometry")
        population_gdf.set_crs(src.crs, inplace=True)
    return population_gdf


def add_intersecting_nodes(gdf, graph):
    nodes_gdf = ox.graph_to_gdfs(graph, nodes=True, edges=False)
    if gdf.crs != nodes_gdf.crs:
        nodes_gdf = nodes_gdf.to_crs(gdf.crs)
    intersecting_node_ids = []
    for geom in gdf.geometry:
        intersecting_nodes = nodes_gdf[nodes_gdf.intersects(geom)]
        intersecting_node_ids.append(intersecting_nodes.index.tolist())
    gdf = gdf.copy()
    gdf['intersecting_nodes'] = intersecting_node_ids
    return gdf


def generate_cumulative_map(population_tiff_file, G):
    bbox = ox.graph_to_gdfs(G, nodes=True, edges=False).total_bounds
    population_gdf = fetch_population_data_from_tiff(population_tiff_file, bbox)
    gdf_extended = add_intersecting_nodes(population_gdf, G)
    gdf_extended['p'] = gdf_extended['population_density'] / gdf_extended.population_density.sum()
    gdf_extended['cumulative_p'] = gdf_extended['p'].cumsum()
    gdf_cumulative_p = gdf_extended[['cumulative_p', 'intersecting_nodes']]
    return gdf_cumulative_p


# ** Prepare OSM graph and population raster (used for node sampling)
def ensure_spatial_resources() -> Tuple[nx.MultiDiGraph, pd.DataFrame]:
    os.makedirs(DATA_DIR, exist_ok=True)

    # OSM graph: load from disk or download once
    if os.path.exists(PORTO_GRAPHML_PATH):
        G = ox.load_graphml(PORTO_GRAPHML_PATH)
    else:
        print(" - Downloading Porto OSM graph...", end="", flush=True)
        G = retrieve_osm(PORTO_CENTER, PORTO_RADIUS_M)
        ox.save_graphml(G, PORTO_GRAPHML_PATH)
        print(" saved")

    # TIFF: download if missing
    if not os.path.exists(SEDAC_TIFF_PATH):
        print(" - Downloading SEDAC population TIFF...", end="", flush=True)
        resp = requests.get(SEDAC_TIFF_URL)
        resp.raise_for_status()
        with open(SEDAC_TIFF_PATH, 'wb') as f:
            f.write(resp.content)
        print(" downloaded")

    # Cumulative population map: cache to pickle
    if os.path.exists(POP_CUMULATIVE_P_PATH):
        with open(POP_CUMULATIVE_P_PATH, 'rb') as f:
            gdf_cumulative_p = pickle.load(f)
    else:
        print(" - Building population cumulative map...", end="", flush=True)
        gdf_cumulative_p = generate_cumulative_map(SEDAC_TIFF_PATH, G)
        with open(POP_CUMULATIVE_P_PATH, 'wb') as f:
            pickle.dump(gdf_cumulative_p, f)
        print(" saved")

    # Debug population cell-node linkage stats
    try:
        print("Population cell intersecting_nodes counts:")
        print(gdf_cumulative_p['intersecting_nodes'].apply(len).describe())
    except Exception as _:
        pass

    return G, gdf_cumulative_p


# ------------------------------
# Visualization: interactive map of trajectory and endpoints
# ------------------------------

def _compute_location_stats_from_stays(stays_by_day: Dict[datetime.date, List[Stay]]) -> Dict[int, Dict[str, Any]]:
    stats: Dict[int, Dict[str, Any]] = {}
    for _, day_stays in stays_by_day.items():
        for s in day_stays:
            if s.location_id not in stats:
                stats[s.location_id] = {"visits": 0, "total_duration": 0}
            stats[s.location_id]["visits"] += 1
            if s.duration is not None:
                stats[s.location_id]["total_duration"] += int(s.duration)
    return stats


def generate_interactive_map(user_id: int,
                             trajectory: List[Tuple[float, float, int]],
                             map_loc: Dict[int, int],
                             G: nx.MultiDiGraph,
                             enriched_imn: Dict,
                             stays_by_day: Dict[datetime.date, List[Stay]],
                             out_html_path: str,
                             draw_polyline: bool = True,
                             activity_override: Optional[Dict[int, str]] = None) -> None:
    if not trajectory and not map_loc:
        return

    # Center map on the midpoint of trajectory if provided, otherwise on mapped nodes
    if trajectory:
        mid_idx = len(trajectory) // 2
        center_lat, center_lon = trajectory[mid_idx][0], trajectory[mid_idx][1]
    else:
        lats = []
        lons = []
        for _, osm_node in map_loc.items():
            node_data = G.nodes[osm_node]
            lats.append(node_data['y'])
            lons.append(node_data['x'])
        if not lats or not lons:
            return
        center_lat = float(np.mean(lats))
        center_lon = float(np.mean(lons))

    # Create map with semi-transparent OSM tiles
    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=None)
    folium.TileLayer('OpenStreetMap', opacity=0.6, control=False).add_to(m)

    # Draw trajectory polyline (optional)
    if draw_polyline and trajectory:
        polyline_coords = [(lat, lon) for lat, lon, _ in trajectory]
        folium.PolyLine(polyline_coords, color="#1f77b4", weight=3, opacity=0.8, tooltip="Trajectory").add_to(m)

    # Compute per-location stats from stays
    loc_stats = _compute_location_stats_from_stays(stays_by_day)

    # Add markers for each IMN location (mapped to OSM node)
    for loc_id, osm_node in map_loc.items():
        node_data = G.nodes[osm_node]
        node_lat = node_data['y']
        node_lon = node_data['x']
        # Resolve activity label: override for synthetic/pseudo IDs if provided
        if activity_override is not None and loc_id in activity_override:
            activity = activity_override[loc_id]
            freq = None
        else:
            loc_info = enriched_imn['locations'].get(loc_id, {})
            activity = loc_info.get('activity_label', 'unknown')
            freq = loc_info.get('frequency', None)
        stat = loc_stats.get(loc_id, {"visits": 0, "total_duration": 0})
        total_hours = round(stat["total_duration"] / 3600.0, 2) if stat["total_duration"] else 0.0
        popup_html = f"""
        <div style='font-size:12px;'>
            <b>User:</b> {user_id}<br/>
            <b>Location ID:</b> {loc_id}<br/>
            <b>Activity:</b> {activity}<br/>
            <b>Visits (days stays):</b> {stat['visits']}<br/>
            <b>Total stay duration (h):</b> {total_hours}<br/>
            <b>IMN frequency:</b> {freq if freq is not None else '-'}<br/>
            <b>OSM node:</b> {osm_node}
        </div>
        """
        # Color indicator using ACTIVITY_COLORS as a small circle, plus default Marker on top
        marker_color = ACTIVITY_COLORS.get(activity, "black")
        folium.CircleMarker(
            [node_lat, node_lon],
            radius=6,
            color=marker_color,
            fill=True,
            fill_color=marker_color,
            fill_opacity=0.9,
            weight=2,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{activity}"
        ).add_to(m)

    # Save map
    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    m.save(out_html_path)


def create_split_map_html(left_title: str, left_src: str, right_title: str, right_src: str, out_html_path: str) -> None:
    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\" />
  <title>Split Map View</title>
  <style>
    body {{ margin: 0; padding: 0; font-family: Arial, sans-serif; }}
    .container {{ display: flex; height: 100vh; width: 100vw; }}
    .pane {{ flex: 1; display: flex; flex-direction: column; min-width: 0; }}
    .header {{ padding: 8px 12px; background: #f5f5f5; border-bottom: 1px solid #ddd; font-weight: bold; }}
    .frame {{ flex: 1; border: 0; width: 100%; }}
  </style>
  </head>
  <body>
    <div class=\"container\">
      <div class=\"pane\">
        <div class=\"header\">{left_title}</div>
        <iframe class=\"frame\" src=\"{left_src}\"></iframe>
      </div>
      <div class=\"pane\"> 
        <div class=\"header\">{right_title}</div>
        <iframe class=\"frame\" src=\"{right_src}\"></iframe>
      </div>
    </div>
  </body>
</html>
"""
    with open(out_html_path, 'w') as f:
        f.write(html)


# ------------------------------
# Synthetic stays → spatial trips simulation
# ------------------------------

# ** Map original IMN home/work locations to fixed OSM nodes
def simulate_synthetic_trips(
    imn: Dict,
    synthetic_stays: List[Tuple[str, int, int]],
    G: nx.MultiDiGraph,
    gdf_cumulative_p: pd.DataFrame,
    randomness: float = 0.5,
    fixed_home_node: Optional[int] = None,
    fixed_work_node: Optional[int] = None,
    precomputed_map_loc_rmse: Optional[Tuple[Dict[int, int], float]] = None,
) -> Tuple[List[Tuple[float, float, int]], Dict[Any, int], Dict[str, int], float, List[List[Tuple[float, float, int]]]]:
    """Simulate spatial trips for one day based on synthetic stays.

    - synthetic_stays: list of (activity_label, start_rel, end_rel) in seconds from midnight
    Returns (trajectory, osm_segments_usage, activity_to_node, rmse)
    """
    if not synthetic_stays:
        return None, None, None, None, None

    # ** Map original IMN home/work locations to fixed OSM nodes
    # Establish fixed home/work mapping once
    if precomputed_map_loc_rmse is not None:
        map_loc_imn, rmse = precomputed_map_loc_rmse
        print("    [synthetic] using precomputed IMN→OSM mapping for home/work")
    else:
        print("    [synthetic] mapping IMN home/work to OSM...")
        map_loc_imn, rmse = map_imn_to_osm(imn, G, gdf_cumulative_p=gdf_cumulative_p)

    # Choose home/work nodes: prefer fixed if provided
    home_node = fixed_home_node if fixed_home_node is not None else map_loc_imn.get(imn['home'])
    work_node = fixed_work_node if fixed_work_node is not None else map_loc_imn.get(imn['work'])
    all_nodes = list(G.nodes())

    # ** Assign each stay to an OSM node (home/work fixed, others via sampling/POI)
    # Per-stay mapping: assign one OSM node per stay instance
    stay_nodes: List[int] = []
    unique_acts = [ act for (act, _, _) in synthetic_stays ]
    print(f"    [synthetic] activities in day (ordered): {unique_acts}")
    for idx, (act, st, et) in enumerate(synthetic_stays):
        # ** Normalize activity labels to ensure consistent mapping rules
        # Debug type and normalize label
        print(f"    [synthetic][debug] activity label type: {act} ({type(act)})")
        act_label = str(act).lower() if act is not None else "unknown"
        # ** Assign each stay to an OSM node (home/work fixed, others via sampling/POI)
        if act_label == 'home' and home_node is not None:
            stay_nodes.append(home_node)
        elif act_label == 'work' and work_node is not None:
            stay_nodes.append(work_node)
        else:
            sampled = select_random_nodes(gdf_cumulative_p, n=1, all_nodes=all_nodes)
            if not sampled:
                print("    [synthetic][warn] no sampled node; using random graph node")
                stay_nodes.append(all_nodes[np.random.randint(len(all_nodes))])
            else:
                stay_nodes.append(sampled[0])

    # First location override: only enforce home if the first activity is home
    first_label = str(synthetic_stays[0][0]).lower() if synthetic_stays and synthetic_stays[0][0] is not None else ""
    if len(stay_nodes) > 0 and first_label == 'home' and home_node is not None:
        stay_nodes[0] = home_node
    print(f"    [synthetic] per-stay nodes: {stay_nodes}")
    try:
        distinct_nodes = len(set(stay_nodes))
        coords = [(G.nodes[n]['y'], G.nodes[n]['x']) for n in stay_nodes]
        print(f"    [synthetic] distinct nodes: {distinct_nodes}")
    except Exception:
        pass

    # Identify anchor (longest non-home stay)
    anchor_idx = None
    max_dur = -1
    for idx, (act, st, et) in enumerate(synthetic_stays):
        if act == 'home':
            continue
        dur = max(0, (et or 0) - (st or 0))
        if dur > max_dur:
            max_dur = dur
            anchor_idx = idx

    # Make mutable copy to adjust times
    stays_mut = [ [act, int(st), int(et)] for (act, st, et) in synthetic_stays ]

    # Build trips and adjust for travel durations
    trajectory: List[Tuple[float, float, int]] = []
    osm_segments_usage: Dict[Any, int] = {}
    legs_coords: List[List[Tuple[float, float, int]]] = []

    print(f"    [synthetic] synthetic stays: {len(stays_mut)} → expected trips: {len(stays_mut)-1}")
    leg_success = 0
    leg_fail = 0
    for i in range(len(stays_mut) - 1):
        act_from, st_from, et_from = stays_mut[i]
        act_to, st_to, et_to = stays_mut[i + 1]

        origin_node = stay_nodes[i]
        dest_node = stay_nodes[i + 1]

        # ** Compute shortest path in OSM graph between consecutive stay nodes
        # Allocated gap (trip window)
        allocated_gap = max(0, st_to - et_from)

        # If trip is shorter than gap, start later to arrive exactly at next start
        trip_start_time = et_from
        shortest_p, duration_p, osm_segments = create_trajectory(G, origin_node, dest_node, trip_start_time, use_cache=True)
        if shortest_p is None:
            y1, x1 = G.nodes[origin_node]['y'], G.nodes[origin_node]['x']
            y2, x2 = G.nodes[dest_node]['y'], G.nodes[dest_node]['x']
            print(f"    [synthetic][warn] no route {origin_node}->{dest_node} ({y1:.5f},{x1:.5f} -> {y2:.5f},{x2:.5f}); skipping")
            leg_fail += 1
            continue

        # ** Adjust stay start/end times if travel duration > or < allocated gap
        if duration_p > allocated_gap:
            # Need more time than allocated
            delta = int(duration_p - allocated_gap)
            if i == anchor_idx:
                # Previous is anchor → cannot shift it earlier; shift next start later
                stays_mut[i + 1][1] += delta  # shift next start
            elif i + 1 == anchor_idx:
                # Next is anchor → shift previous end earlier
                stays_mut[i][2] = max(st_from, et_from - delta)
                trip_start_time = stays_mut[i][2]
            else:
                # Shift previous end earlier in general case
                stays_mut[i][2] = max(st_from, et_from - delta)
                trip_start_time = stays_mut[i][2]
            # Recompute route at adjusted start
            shortest_p, duration_p, osm_segments = create_trajectory(G, origin_node, dest_node, trip_start_time, use_cache=True)
            if shortest_p is None:
                print(f"    [synthetic][warn] no route after adjust {origin_node}->{dest_node}; skipping")
                leg_fail += 1
                continue
        else:
            # duration shorter than allocated gap: delay start so arrival matches next start
            slack = int(allocated_gap - duration_p)
            trip_start_time = et_from + slack
            # extend previous stay to the delayed start
            stays_mut[i][2] = trip_start_time
            # recompute route at delayed start to preserve timing
            shortest_p, duration_p, osm_segments = create_trajectory(G, origin_node, dest_node, trip_start_time, use_cache=True)
            if shortest_p is None:
                print(f"    [synthetic][warn] no route with slack {origin_node}->{dest_node}; skipping")
                leg_fail += 1
                continue

        # Append trajectory points (whole trajectory and per-leg)
        trajectory.extend(shortest_p)
        legs_coords.append(shortest_p)
        # Update usage
        for oss in osm_segments:
            osm_segments_usage[oss] = osm_segments_usage.get(oss, 0) + 1
        leg_success += 1

    print(f"    [synthetic] legs success={leg_success}, fail={leg_fail}")
    print(f"    [synthetic] built {len(trajectory)} points across {len(osm_segments_usage)} segments")
    # For mapping on the Porto map: convert per-stay mapping to a label-index map
    pseudo_map = { i: node for i, node in enumerate(stay_nodes) }
    return trajectory, osm_segments_usage, pseudo_map, rmse, legs_coords


# ------------------------------
# Multi-day Porto map with per-day layers and combined layer
# ------------------------------

def generate_interactive_porto_map_multi(
    user_id: int,
    per_day_data: Dict[datetime.date, Dict[str, Any]],
    G: nx.MultiDiGraph,
    out_html_path: str
) -> None:
    # per_day_data[day] = { 'trajectory': List[(lat,lon,t)], 'pseudo_map_loc': Dict[idx->node], 'synthetic_stays': List[(act,s,e)] }
    if not per_day_data:
        return

    # Determine center from first available trajectory
    first_day = next(iter(per_day_data.keys()))
    first_traj = per_day_data[first_day]['trajectory']
    if first_traj:
        mid_idx = len(first_traj) // 2
        center_lat, center_lon = first_traj[mid_idx][0], first_traj[mid_idx][1]
    else:
        # fallback: any node from first day
        any_node = next(iter(per_day_data[first_day]['pseudo_map_loc'].values()))
        center_lat, center_lon = G.nodes[any_node]['y'], G.nodes[any_node]['x']

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=None)
    folium.TileLayer('OpenStreetMap', opacity=0.6, control=False).add_to(m)

    # Static layer for unique home/work markers across all days
    static_hw = folium.FeatureGroup(name="Home/Work", show=True)
    added_hw_nodes = set()

    # Per-day layers
    for day, content in per_day_data.items():
        day_label = str(day)
        fg = folium.FeatureGroup(name=f"Day {day_label}", show=True)

        # Progressive blue per leg: from 1.0 down to ~0.45
        legs_coords: List[List[Tuple[float, float, int]]] = content.get('legs_coords', [])
        if legs_coords:
            n_legs = len(legs_coords)
            if n_legs > 0:
                for i, leg in enumerate(legs_coords):
                    coords = [(lat, lon) for lat, lon, _ in leg]
                    # opacity from 1.0 to 0.45 across legs
                    if n_legs == 1:
                        opacity = 1.0
                    else:
                        opacity = max(0.45, 1.0 - (i * (1.0 - 0.45) / (n_legs - 1)))
                    folium.PolyLine(coords, color="#1f77b4", weight=3, opacity=opacity, tooltip=f"Trip {i+1} - {day_label}").add_to(fg)
        else:
            traj = content['trajectory']
            if traj:
                coords = [(lat, lon) for lat, lon, _ in traj]
                folium.PolyLine(coords, color="#1f77b4", weight=3, opacity=0.9, tooltip=f"Trajectory {day_label}").add_to(fg)

        pseudo_map_loc = content['pseudo_map_loc']
        synthetic_stays = content['synthetic_stays']
        for idx, node in pseudo_map_loc.items():
            if node not in G.nodes:
                continue
            node_lat = G.nodes[node]['y']
            node_lon = G.nodes[node]['x']
            act = str(synthetic_stays[idx][0]).lower() if idx < len(synthetic_stays) else 'unknown'
            marker_color = ACTIVITY_COLORS.get(act, "black")
            popup_html = f"""
            <div style='font-size:12px;'>
                <b>User:</b> {user_id}<br/>
                <b>Day:</b> {day_label}<br/>
                <b>Stay index:</b> {idx}<br/>
                <b>Activity:</b> {act}<br/>
                <b>OSM node:</b> {node}
            </div>
            """
            # Add home/work once in static layer; others per-day layer
            if act in ("home", "work"):
                if node not in added_hw_nodes:
                    folium.CircleMarker(
                        [node_lat, node_lon],
                        radius=7,
                        color=marker_color,
                        fill=True,
                        fill_color=marker_color,
                        fill_opacity=0.95,
                        weight=2,
                        popup=folium.Popup(popup_html, max_width=300),
                        tooltip=f"{act}"
                    ).add_to(static_hw)
                    added_hw_nodes.add(node)
            else:
                folium.CircleMarker(
                    [node_lat, node_lon],
                    radius=6,
                    color=marker_color,
                    fill=True,
                    fill_color=marker_color,
                    fill_opacity=0.9,
                    weight=2,
                    popup=folium.Popup(popup_html, max_width=300),
                    tooltip=f"{act}"
                ).add_to(fg)

        fg.add_to(m)

    static_hw.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    m.save(out_html_path)

# ------------------------------
# IO / CONFIG
# ------------------------------

@dataclass
class PathsConfig:
    full_imn_path: str = 'data/milano_2007_imns.json.gz'
    test_imn_path: str = 'data/test_milano_imns.json.gz'
    poi_path: str = 'data/test_milano_imns_pois.json.gz'
    results_dir: str = './results'
    prob_subdir: str = 'user_probability_reports4'
    vis_subdir: str = 'user_timeline_visualizations4'

    def prob_dir(self) -> str:
        return os.path.join(self.results_dir, self.prob_subdir)

    def vis_dir(self) -> str:
        return os.path.join(self.results_dir, self.vis_subdir)


def ensure_output_structure(paths: PathsConfig) -> None:
    os.makedirs(paths.results_dir, exist_ok=True)
    os.makedirs(paths.prob_dir(), exist_ok=True)
    os.makedirs(paths.vis_dir(), exist_ok=True)


def parse_args(argv: Optional[List[str]] = None) -> PathsConfig:
    parser = argparse.ArgumentParser(description='Generate synthetic timelines from IMNs.')
    parser.add_argument('--full-imn', default='data/milano_2007_imns.json.gz', help='Path to full IMNs (for user IDs).')
    parser.add_argument('--test-imn', default='data/test_milano_imns.json.gz', help='Path to subset/test IMNs.')
    parser.add_argument('--poi', default='data/test_milano_imns_pois.json.gz', help='Path to POI enrichment data.')
    parser.add_argument('--results-dir', default='./results', help='Base directory for results.')
    parser.add_argument('--prob-subdir', default='user_probability_reports4', help='Probability reports subdirectory.')
    parser.add_argument('--vis-subdir', default='user_timeline_visualizations4', help='Timeline visualizations subdirectory.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (optional).')
    args = parser.parse_args(argv)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    return PathsConfig(
        full_imn_path=args.full_imn,
        test_imn_path=args.test_imn,
        poi_path=args.poi,
        results_dir=args.results_dir,
        prob_subdir=args.prob_subdir,
        vis_subdir=args.vis_subdir,
    )


def read_poi_data(filepath: str) -> Dict[int, Dict]:
    """Read POI data from gzipped JSON file."""
    poi_data = {}
    with gzip.open(filepath, 'rt', encoding='utf-8') as f:
        for line in f:
            row = json.loads(line)
            poi_data[row['uid']] = row
    return poi_data


def enrich_imn_with_poi(imn: Dict, poi_info: Dict) -> Dict:
    """Enrich IMN with POI activity labels."""
    poi_classes = poi_info["poi_classes"]
    enriched = {}
    
    for loc_id, loc in imn["locations"].items():
        vec = poi_info["poi_freq"].get(loc_id, [0.0] * len(poi_classes))
        top_idx = int(np.argmax(vec))
        label = POI_TO_ACTIVITY.get(poi_classes[top_idx], "unknown")
        
        # Override with home/work if applicable
        if loc_id == imn.get("home"): 
            label = "home"
        if loc_id == imn.get("work"): 
            label = "work"
            
        enriched[loc_id] = {**loc, "activity_label": label}
    
    imn["locations"] = enriched
    return imn


def extract_stays_from_trips(trips: List[Tuple], locations: Dict) -> List[Stay]:
    """Convert trips into stays by considering the destination of each trip as a stay.
    Gaps are later handled by day-stretching; we do not create a 'trip' activity."""
    stays = []
    
    # First pass: create stays with start times
    for from_id, to_id, st, et in trips:
        activity_label = locations[to_id].get('activity_label', 'unknown')
        stays.append(Stay(to_id, activity_label, et, None))
    
    # Second pass: set end times only; gaps will be handled by stretching
    all_activities = []
    for i in range(len(stays)):
        current_stay = stays[i]
        
        # Set end time based on next trip's start time
        if i < len(stays) - 1:
            next_trip_start = trips[i + 1][2]  # start time of the next trip
            current_stay.set_end_time(next_trip_start)
        else:
            # Handle the last stay
            if current_stay.start_time is not None:
                current_stay.set_end_time(current_stay.start_time + 3600)  # Default 1 hour duration
        
        # Add the stay activity
        all_activities.append(current_stay)
    
    return all_activities


def extract_stays_by_day(stays: List[Stay], tz) -> Dict[datetime.date, List[Stay]]:
    """Group stays by day, handling cross-day stays."""
    stays_by_day = defaultdict(list)
    
    for stay in stays:
        if stay.start_time is None or stay.end_time is None:
            continue
            
        start_dt = datetime.fromtimestamp(stay.start_time, tz)
        end_dt = datetime.fromtimestamp(stay.end_time, tz)
        
        # If stay spans multiple days, split it
        current_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = current_dt + timedelta(days=1)
        
        while current_dt < end_dt:
            day_start = max(start_dt, current_dt)
            day_end = min(end_dt, end_of_day)
            
            # Create stay for this day
            day_stay = Stay(
                stay.location_id,
                stay.activity_label,
                int(day_start.timestamp()),
                int(day_end.timestamp())
            )
            
            stays_by_day[current_dt.date()].append(day_stay)
            
            # Move to next day
            current_dt = end_of_day
            end_of_day = current_dt + timedelta(days=1)
    
    return stays_by_day


def _stretch_day_stays_to_full_coverage(day_stays: List[Stay], tz) -> List[Stay]:
    """Stretch a single day's stays so they exactly cover the whole day without gaps.

    Rules:
    - Snap first stay start to midnight
    - Between consecutive stays, set previous end to next start (remove gaps/overlaps)
    - Snap last stay end to next midnight
    """
    if not day_stays:
        return []

    # Determine day bounds from first stay
    first_dt = datetime.fromtimestamp(day_stays[0].start_time, tz)
    midnight_dt = first_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    day_start = int(midnight_dt.timestamp())
    day_end = int((midnight_dt + timedelta(days=1)).timestamp())

    # Sort by start time
    sorted_stays = sorted(day_stays, key=lambda s: (s.start_time or 0, s.end_time or 0))

    # Clone to avoid mutating original objects unintentionally
    stretched: List[Stay] = []
    for s in sorted_stays:
        stretched.append(Stay(s.location_id, s.activity_label, s.start_time, s.end_time))

    # Snap first start to midnight
    if stretched[0].start_time is None:
        stretched[0].start_time = day_start
    else:
        stretched[0].start_time = min(max(stretched[0].start_time, day_start), day_end)

    # Ensure continuity between stays (fill gaps, trim overlaps)
    for i in range(len(stretched) - 1):
        current = stretched[i]
        nxt = stretched[i + 1]

        # Normalize None times
        if current.end_time is None:
            current.end_time = current.start_time
        if nxt.start_time is None:
            nxt.start_time = current.end_time

        # Force continuity: current ends exactly at next start
        nxt.start_time = max(min(nxt.start_time, day_end), day_start)
        current.end_time = max(min(nxt.start_time, day_end), day_start)

        # Update duration
        current.duration = max(0, current.end_time - current.start_time)

    # Snap last end to day_end
    last = stretched[-1]
    if last.end_time is None:
        last.end_time = day_end
    else:
        last.end_time = min(max(last.end_time, day_start), day_end)
    last.end_time = day_end
    last.duration = max(0, last.end_time - last.start_time)

    # Ensure first start is day_start after adjustments
    stretched[0].start_time = day_start
    stretched[0].duration = max(0, (stretched[0].end_time or day_start) - stretched[0].start_time)

    return stretched


def stretch_all_days(stays_by_day: Dict[datetime.date, List[Stay]], tz) -> Dict[datetime.date, List[Stay]]:
    """Apply stretching to every day's stays to eliminate gaps and cover full day."""
    stretched_by_day: Dict[datetime.date, List[Stay]] = {}
    for day, day_stays in stays_by_day.items():
        stretched_by_day[day] = _stretch_day_stays_to_full_coverage(day_stays, tz)
    return stretched_by_day


def build_stay_distributions(stays_by_day: Dict[datetime.date, List[Stay]]) -> Tuple[Dict, Dict, Any]:
    """Build distributions for stay durations and activity types across all days."""
    duration_dist = defaultdict(list)
    activity_transitions = defaultdict(list)
    trip_durations = []
    
    # Collect data from all days
    for day, day_stays in stays_by_day.items():
        for i in range(len(day_stays) - 1):
            current_stay = day_stays[i]
            next_stay = day_stays[i + 1]
            
            # Record duration for this activity type
            if current_stay.duration is not None:
                duration_dist[current_stay.activity_label].append(current_stay.duration)
            
            # Record activity transition
            activity_transitions[current_stay.activity_label].append(next_stay.activity_label)
            
            # Record trip duration (gap between stays)
            if current_stay.end_time is not None and next_stay.start_time is not None:
                trip_duration = next_stay.start_time - current_stay.end_time
                if trip_duration > 0:
                    trip_durations.append(trip_duration)
    
    # Convert lists to probability distributions
    duration_probs = {}
    for activity, durations in duration_dist.items():
        if len(durations) > 0:
            hist, bins = np.histogram(durations, bins=20, density=False)
            duration_probs[activity] = (hist, bins)
    
    transition_probs = {}
    for from_activity, to_activities in activity_transitions.items():
        if len(to_activities) > 0:
            unique_activities, counts = np.unique(to_activities, return_counts=True)
            probs = counts / counts.sum()
            transition_probs[from_activity] = dict(zip(unique_activities, probs))
    
    # Trip duration distribution
    trip_duration_probs = None
    if len(trip_durations) > 0:
        hist, bins = np.histogram(trip_durations, bins=20, density=False)
        trip_duration_probs = (hist, bins)
    
    return duration_probs, transition_probs, trip_duration_probs


def user_probs_report(duration_probs: Dict, transition_probs: Dict, trip_duration_probs: Any, 
                     user_id: int, out_folder: str) -> None:
    """Generate and save user probability report with visualizations."""
    os.makedirs(out_folder, exist_ok=True)

    # Expand durations into samples for plotting
    expanded = {}
    for act, (hist, bins) in duration_probs.items():
        mids = (bins[:-1] + bins[1:]) / 2
        samples = np.repeat(mids, hist)
        expanded[act] = samples

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Durations boxplot
    if expanded:
        sns.boxplot(data=pd.DataFrame(dict([(k, pd.Series(v)) for k, v in expanded.items()])),
                    ax=axes[0])
        axes[0].set_title("Stay Durations per Activity")
        axes[0].set_ylabel("Duration (s)")
        axes[0].tick_params(axis='x', rotation=45)

    # Transition heatmap
    df = pd.DataFrame(transition_probs).fillna(0)
    sns.heatmap(df, annot=True, cmap="Blues", cbar_kws={'label': 'Probability'}, ax=axes[1])
    axes[1].set_title("Transition Probability Matrix")
    axes[1].set_xlabel("From Activity")
    axes[1].set_ylabel("To Activity")

    # Trip duration density
    if trip_duration_probs is not None:
        hist, bins = trip_duration_probs
        mids = (bins[:-1] + bins[1:]) / 2
        samples = np.repeat(mids, hist)
        if len(samples) > 1:
            sns.kdeplot(samples, fill=True, ax=axes[2])
        else:
            axes[2].hist(samples, bins=10, edgecolor="k")
        axes[2].set_title("Trip Duration Distribution")
        axes[2].set_xlabel("Duration (s)")
        axes[2].set_ylabel("Density")

    plt.tight_layout()
    fig_path = os.path.join(out_folder, f"user_probs_report_{user_id}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # Save JSON report
    json_report = {
        "duration_probs": {k: {"hist": hist.tolist(), "bins": bins.tolist()} 
                          for k, (hist, bins) in duration_probs.items()},
        "transition_probs": {k: {kk: float(vv) for kk, vv in d.items()} 
                           for k, d in transition_probs.items()},
        "trip_duration_probs": {
            "hist": trip_duration_probs[0].tolist(),
            "bins": trip_duration_probs[1].tolist()
        } if trip_duration_probs is not None else None
    }

    json_path = os.path.join(out_folder, f"user_probs_report_{user_id}.json")
    with open(json_path, "w") as f:
        json.dump(json_report, f, indent=2)

    print(f"  ✓ Probability report saved: {os.path.basename(fig_path)}, {os.path.basename(json_path)}")


def sample_from_hist(hist: np.ndarray, bins: np.ndarray) -> float:
    """Sample a value from histogram (hist, bins)."""
    if hist.sum() == 0:
        return np.mean(bins)
    probs = hist / hist.sum()
    bin_idx = np.random.choice(len(hist), p=probs)
    return np.random.uniform(bins[bin_idx], bins[bin_idx + 1])


def find_anchor_stay_for_day(stays: List[Stay]) -> Stay:
    """Find the longest non-home stay for a given day."""
    non_home_stays = [s for s in stays if s.activity_label != 'home']
    if not non_home_stays:
        return None
    return max(non_home_stays, key=lambda s: s.duration)


def generate_synthetic_day(original_stays: List[Stay], duration_probs: Dict, 
                          transition_probs: Dict, randomness: float = 0.5, 
                          day_length: int = 24 * 3600, tz=None) -> List[Tuple[str, int, int]]:
    """Generate synthetic day timeline based on original stays and learned distributions."""
    if not original_stays:
        return []
    
    # Convert to relative seconds
    first_dt = datetime.fromtimestamp(original_stays[0].start_time, tz)
    day_start_dt = first_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    day_start = int(day_start_dt.timestamp())

    rel_stays = []
    for s in original_stays:
        rel_start = int(s.start_time - day_start)
        rel_end = int(s.end_time - day_start)
        rel_stays.append((s, rel_start, rel_end))

    # Find anchor (in relative time)
    anchor_stay = find_anchor_stay_for_day(original_stays)
    if anchor_stay is None:
        return []  # fallback if no non-home stay

    anchor_rel_start = int(anchor_stay.start_time - day_start)
    anchor_rel_end = int(anchor_stay.end_time - day_start)
    anchor_orig_dur = max(1, anchor_rel_end - anchor_rel_start)

    # Perturb anchor start and duration within +/- 0.25 * randomness * original_duration
    max_shift = int(0.25 * randomness * anchor_orig_dur)
    if max_shift > 0:
        delta_start = int(np.random.uniform(-max_shift, max_shift))
        delta_dur = int(np.random.uniform(-max_shift, max_shift))
    else:
        delta_start = 0
        delta_dur = 0

    pert_start = anchor_rel_start + delta_start
    pert_start = max(0, min(pert_start, day_length))
    pert_dur = max(1, anchor_orig_dur + delta_dur)
    # If anchor is the first activity of the day, do not allow any leading gap
    is_anchor_first = False
    if rel_stays and rel_stays[0][0] is anchor_stay:
        is_anchor_first = True
        pert_start = 0

    pert_end = pert_start + pert_dur
    if pert_end > day_length:
        pert_end = day_length
        pert_dur = max(1, pert_end - pert_start)

    synthetic_stays = []
    current_time = 0
    prev_activity = "home"  # always force first stay to begin at home

    # Generate before anchor (respect perturbed anchor start)
    for (s, rel_start, rel_end) in rel_stays:
        if s is anchor_stay:
            break

        # Choose activity
        if current_time == 0:
            act = "home"  # enforce home at midnight
        elif random.random() < (1 - randomness):
            act = s.activity_label
        else:
            if prev_activity in transition_probs:
                to_probs = transition_probs[prev_activity]
                act = random.choices(list(to_probs.keys()), weights=to_probs.values())[0]
            else:
                act = s.activity_label

        # Choose duration
        orig_dur = rel_end - rel_start
        if act in duration_probs:
            hist, bins = duration_probs[act]
            sampled_dur = sample_from_hist(hist, bins)
        else:
            # Fallback for activities not in distributions
            sampled_dur = orig_dur
        dur = int((1 - randomness) * orig_dur + randomness * sampled_dur)

        end_time = min(current_time + dur, pert_start)
        if end_time > current_time:  # avoid zero/negative durations
            synthetic_stays.append((act, current_time, end_time))
            prev_activity = act
        current_time = end_time
        if current_time >= pert_start:
            break

    # Insert anchor with perturbation applied
    synthetic_stays.append((anchor_stay.activity_label, pert_start, pert_end))
    current_time = pert_end
    prev_activity = anchor_stay.activity_label

    # Generate after anchor
    passed_anchor = False
    for (s, rel_start, rel_end) in rel_stays:
        if s is anchor_stay:
            passed_anchor = True
            continue
        if not passed_anchor:
            continue

        # Choose activity
        if random.random() < (1 - randomness):
            act = s.activity_label
        else:
            if prev_activity in transition_probs:
                to_probs = transition_probs[prev_activity]
                act = random.choices(list(to_probs.keys()), weights=to_probs.values())[0]
            else:
                act = s.activity_label

        # Choose duration
        orig_dur = rel_end - rel_start
        if act in duration_probs:
            hist, bins = duration_probs[act]
            sampled_dur = sample_from_hist(hist, bins)
        else:
            # Fallback for activities not in distributions
            sampled_dur = orig_dur
        dur = int((1 - randomness) * orig_dur + randomness * sampled_dur)

        end_time = min(current_time + dur, day_length)
        if end_time > current_time:
            synthetic_stays.append((act, current_time, end_time))
            prev_activity = act
        current_time = end_time
        if current_time >= day_length:
            break

    # Anchor-aware stretching: do not change anchor; extend pre-anchor last stay to
    # anchor start if needed; extend final non-anchor stay to day end.
    def _stretch_anchor_aware(
        stays_rel: List[Tuple[str, int, int]],
        anchor_tuple: Tuple[str, int, int],
        total_len: int
    ) -> List[Tuple[str, int, int]]:
        if not stays_rel:
            return []
        # Sort by start time
        ordered = sorted(stays_rel, key=lambda x: x[1])

        # Identify anchor index if present exactly
        anchor_idx = None
        for i, t in enumerate(ordered):
            if t == anchor_tuple:
                anchor_idx = i
                break

        # Ensure the first stay starts at 0 if it's not the anchor
        out: List[Tuple[str, int, int]] = []
        act0, s0, e0 = ordered[0]
        if anchor_idx == 0:
            # Keep anchor unchanged; do not force start to 0
            out.append((act0, s0, e0))
        else:
            s0 = 0
            e0 = max(0, min(e0, total_len))
            if e0 < s0:
                e0 = s0
            out.append((act0, s0, e0))

        # Iterate and enforce continuity, preserving anchor start
        for i in range(1, len(ordered)):
            act, st, et = ordered[i]
            st = max(0, min(st, total_len))
            et = max(0, min(et, total_len))
            prev_act, prev_st, prev_et = out[-1]

            if anchor_idx is not None and i == anchor_idx:
                # Do not move anchor start; stretch previous end to anchor start if gap
                st_fixed = st
                prev_et = min(max(st_fixed, 0), total_len)
                out[-1] = (prev_act, prev_st, prev_et)
                if et < st_fixed:
                    et = st_fixed
                out.append((act, st_fixed, et))
            else:
                # Force current start to previous end
                st = prev_et
                if et < st:
                    et = st
                out.append((act, st, et))

        # Stretch last non-anchor stay to day end
        last_idx = len(out) - 1
        if anchor_idx is not None and last_idx == anchor_idx:
            # Do not stretch if anchor is the last; leave as is
            pass
        else:
            last_act, last_st, _ = out[-1]
            out[-1] = (last_act, last_st, total_len)

        return out

    anchor_tuple = (anchor_stay.activity_label, pert_start, pert_end)
    synthetic_stays = _stretch_anchor_aware(synthetic_stays, anchor_tuple, day_length)
    return synthetic_stays


def prepare_day_data(stays_by_day: Dict[datetime.date, List[Stay]], 
                    user_duration_probs: Dict, user_transition_probs: Dict, 
                    randomness_levels: List[float], tz) -> Dict:
    """Prepare day data structure for visualization."""
    day_data = {}
    sorted_days = sorted(stays_by_day.keys())

    for day_date in sorted_days:
        day = stays_by_day[day_date]

        # Midnight timestamp
        first_dt = datetime.fromtimestamp(day[0].start_time, tz)
        midnight_dt = first_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        day_start = int(midnight_dt.timestamp())

        # Original stays → relative seconds
        original_stays = [
            (s.activity_label, s.start_time - day_start, s.end_time - day_start)
            for s in day
        ]

        # Synthetic stays for each randomness
        synthetic_dict = {}
        for r in randomness_levels:
            synthetic = generate_synthetic_day(
                day, user_duration_probs, user_transition_probs, randomness=r, tz=tz
            )
            synthetic_dict[r] = synthetic

        # Anchor stay in relative seconds
        anchor = find_anchor_stay_for_day(day)
        anchor_tuple = None
        if anchor:
            anchor_tuple = (
                anchor.activity_label,
                anchor.start_time - day_start,
                anchor.end_time - day_start,
            )

        day_data[day_date] = {
            "original": original_stays,
            "synthetic": synthetic_dict,
            "anchor": anchor_tuple
        }

    return day_data


def plot_stays(stays: List[Tuple[str, int, int]], y_offset: float, ax, anchor: Tuple = None):
    """Plot stays as bars at a given y_offset. Highlight anchor with bold stroke."""
    for act, st, et in stays:
        is_anchor = anchor and (act, st, et) == anchor
        ax.barh(
            y_offset,
            et - st,
            left=st,
            height=0.25,  # same height for all
            color=ACTIVITY_COLORS.get(act, "black"),
            edgecolor="black" if is_anchor else None,
            linewidth=2 if is_anchor else 0.5,
            alpha=0.9
        )


def visualize_day_data(day_data: Dict, user_id: int = 0) -> plt.Figure:
    """Plot all days in one figure with original and synthetic timelines."""
    fig, axes = plt.subplots(len(day_data), 1, figsize=(16, 3 * len(day_data)), sharex=True)
    if len(day_data) == 1:
        axes = [axes]

    for ax, (day_date, content) in zip(axes, sorted(day_data.items())):
        original = content["original"]
        synthetics = content["synthetic"]
        anchor = content.get("anchor")

        # Plot original
        y_offset = 0.5
        plot_stays(original, y_offset, ax, anchor=anchor)
        labels = ["Original"]

        # Plot synthetics
        for r, stays in sorted(synthetics.items()):
            y_offset += 0.4
            plot_stays(stays, y_offset, ax)
            labels.append(f"Rand={r}")

        # Y axis
        ax.set_yticks([0.5 + i * 0.4 for i in range(len(labels))])
        ax.set_yticklabels(labels, fontsize=8)

        # X axis: 24h time-of-day
        ax.set_xlim(0, 24 * 3600)
        ax.set_xticks([i * 3600 for i in range(0, 25, 2)])
        ax.set_xticklabels([f"{i:02d}:00" for i in range(0, 25, 2)], rotation=0)
        ax.grid(True, axis="x", linestyle=":", alpha=0.3)

        ax.set_title(f"User {user_id} - {day_date}", fontsize=12)

    # Legend
    legend_handles = [mpatches.Patch(color=c, label=l) for l, c in ACTIVITY_COLORS.items()]
    fig.legend(handles=legend_handles, title="Activity Types",
               loc="upper right", fontsize=9, title_fontsize=10)

    fig.suptitle(f"User {user_id} - Original and Synthetic Timelines", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.85, 0.98])
    return fig


def process_single_user(user_id: int, imn: Dict, poi_info: Dict, 
                       randomness_levels: List[float], paths: PathsConfig, tz,
                       G: nx.MultiDiGraph = None, gdf_cumulative_p: pd.DataFrame = None) -> None:
    """Process a single user and generate all outputs."""
    print(f"Processing user {user_id}...")
    
    # Enrich IMN with POI data
    enriched = enrich_imn_with_poi(imn, poi_info)
    
    # Extract stays from trips
    stays = extract_stays_from_trips(enriched['trips'], enriched['locations'])
    
    # Group stays by day (keep original, non-stretched for distributions/visualization)
    stays_by_day = extract_stays_by_day(stays, tz)
    
    if not stays_by_day:
        print(f"  ⚠ No valid stays found for user {user_id}")
        return
    
    # Build user-specific distributions
    user_duration_probs, user_transition_probs, user_trip_duration_probs = build_stay_distributions(stays_by_day)
    
    # Generate probability report
    user_probs_report(user_duration_probs, user_transition_probs, user_trip_duration_probs, 
                     user_id, paths.prob_dir())
    
    # Prepare day data for visualization
    day_data = prepare_day_data(stays_by_day, user_duration_probs, user_transition_probs, 
                               randomness_levels, tz)
    # DEBUG: how many synthetic days and stays per day
    try:
        print(f"[DEBUG] User {user_id} synthetic timeline days: {len(day_data)}")
        dbg_r = 0.5 if 0.5 in randomness_levels else (randomness_levels[0] if randomness_levels else None)
        for d, ddata in day_data.items():
            if dbg_r is not None and dbg_r in ddata['synthetic']:
                print(f"   - Day {d}: {len(ddata['synthetic'][dbg_r])} stays @ randomness={dbg_r}")
            else:
                # fallback: print first available randomness length
                if ddata['synthetic']:
                    any_r = next(iter(ddata['synthetic'].keys()))
                    print(f"   - Day {d}: {len(ddata['synthetic'][any_r])} stays @ randomness={any_r}")
    except Exception:
        pass
    
    # Generate timeline visualization
    fig = visualize_day_data(day_data, user_id=user_id)
    timeline_path = os.path.join(paths.vis_dir(), f"user_{user_id}_timelines.png")
    os.makedirs(os.path.dirname(timeline_path), exist_ok=True)
    fig.savefig(timeline_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Timeline visualization saved: {os.path.basename(timeline_path)}")

    # Track generated Porto map path for optional split view
    porto_map_path: Optional[str] = None

    # If spatial resources are provided, run spatial simulation in Porto and save trajectory + map
    if G is not None and gdf_cumulative_p is not None:
        print("  ↳ Running spatial simulation in Porto (synthetic stays, all days)...")
        chosen_r = RANDOMNESS_LEVELS[2] if len(RANDOMNESS_LEVELS) > 2 else RANDOMNESS_LEVELS[0]
        # Compute home/work mapping once for the user and reuse across days
        map_loc_imn_user, rmse_user = map_imn_to_osm(enriched, G, gdf_cumulative_p=gdf_cumulative_p)
        fixed_home = map_loc_imn_user.get(enriched['home'])
        fixed_work = map_loc_imn_user.get(enriched['work'])
        per_day_outputs: Dict[datetime.date, Dict[str, Any]] = {}
        combined_traj: List[Tuple[float, float, int]] = []
        any_success = False
        for some_day, ddata in day_data.items():
            synthetic_for_r = ddata["synthetic"].get(chosen_r, [])
            try:
                print(f"[DEBUG] Spatial mapping is using day {some_day}, randomness={chosen_r}")
                print(f"[DEBUG] Day {some_day} activities: {[act for (act,_,_) in synthetic_for_r]}")
            except Exception:
                pass
            traj, osm_usage, pseudo_map_loc, rmse, legs_coords = simulate_synthetic_trips(
                enriched,
                synthetic_for_r,
                G,
                gdf_cumulative_p,
                randomness=chosen_r,
                fixed_home_node=fixed_home,
                fixed_work_node=fixed_work,
                precomputed_map_loc_rmse=(map_loc_imn_user, rmse_user),
            )
            if traj is None:
                print(f"  ⚠ Spatial simulation failed for user {user_id} on day {some_day}")
                continue
            any_success = True
            combined_traj.extend(traj)
            per_day_outputs[some_day] = {
                'trajectory': traj,
                'pseudo_map_loc': pseudo_map_loc,
                'synthetic_stays': synthetic_for_r,
                'legs_coords': legs_coords,
            }
            try:
                print(f"[DEBUG] Finished spatial mapping for user {user_id} day {some_day}: {len(synthetic_for_r)} stays, {len(traj)} points")
            except Exception:
                pass

        if not any_success:
            print("  ⚠ Spatial simulation failed for all days for this user")
        else:
            # Save combined CSV with trajectory_id for each day
            traj_path = os.path.join(paths.results_dir, "synthetic_trajectories", f"user_{user_id}_porto_trajectory.csv")
            os.makedirs(os.path.dirname(traj_path), exist_ok=True)
            
            # Build DataFrame with trajectory_id (one per day)
            # Convert relative times to Unix timestamps
            traj_records = []
            trajectory_counter = 0
            for some_day in sorted(per_day_outputs.keys()):
                # Get midnight timestamp for this day
                day_midnight = datetime.combine(some_day, datetime.min.time())
                day_midnight_ts = int(day_midnight.replace(tzinfo=TIMEZONE).timestamp())
                
                day_traj = per_day_outputs[some_day]['trajectory']
                for lat, lon, relative_time in day_traj:
                    # Convert relative time (seconds from midnight) to Unix timestamp
                    unix_time = day_midnight_ts + int(relative_time)
                    traj_records.append({
                        'trajectory_id': trajectory_counter,
                        'lat': lat,
                        'lon': lon,
                        'time': unix_time,
                        'day_date': str(some_day)
                    })
                trajectory_counter += 1
            
            df_traj = pd.DataFrame(traj_records)
            df_traj.to_csv(traj_path, index=False)
            print(f"  ✓ Spatial trajectory saved ({trajectory_counter} days/trajectories): {os.path.basename(traj_path)}")

            # Generate multi-day interactive map with per-day layers and combined layer
            map_path = os.path.join(paths.results_dir, "synthetic_trajectories", f"user_{user_id}_porto_map.html")
            try:
                generate_interactive_porto_map_multi(user_id, per_day_outputs, G, map_path)
                print(f"  ✓ Interactive multi-day map saved: {os.path.basename(map_path)}")
                porto_map_path = map_path
            except Exception as e:
                print(f"  ⚠ Failed to create interactive multi-day map: {e}")

    # Also create an interactive map in the original city using straight-line segments between IMN locations
    try:
        print("  ↳ Creating original-city map...")
        # Build straight-line trajectory from IMN trips (lat/lon/time)
        straight_traj = []
        for l_from, l_to, st, et in enriched['trips']:
            from_lat, from_lon = enriched['locations'][l_from]['coordinates'][1], enriched['locations'][l_from]['coordinates'][0]
            to_lat, to_lon = enriched['locations'][l_to]['coordinates'][1], enriched['locations'][l_to]['coordinates'][0]
            straight_traj.append((from_lat, from_lon, st))
            straight_traj.append((to_lat, to_lon, et))

        stays_by_day_local = stays_by_day  # already computed earlier from enriched IMN
        orig_map_path = os.path.join(paths.results_dir, "synthetic_trajectories", f"user_{user_id}_original_city_map.html")

        # Use a dummy SmallGraph to reuse generate_interactive_map: create node entries for each IMN location
        class SmallGraph:
            def __init__(self):
                self.nodes = {}
        G_small = SmallGraph()
        map_loc_orig = {}
        for loc_id, loc in enriched['locations'].items():
            fake_node_id = loc_id  # reuse loc_id as node id
            map_loc_orig[loc_id] = fake_node_id
            G_small.nodes[fake_node_id] = { 'x': loc['coordinates'][0], 'y': loc['coordinates'][1] }

        # Do not draw trajectory on original map (only markers by activity color)
        generate_interactive_map(user_id, straight_traj, map_loc_orig, G_small, enriched, stays_by_day_local, orig_map_path, draw_polyline=False)
        print(f"  ✓ Original-city map saved: {os.path.basename(orig_map_path)}")

        # Build split view HTML combining the two maps (only if Porto map was created)
        if porto_map_path is not None:
            split_path = os.path.join(paths.results_dir, "synthetic_trajectories", f"user_{user_id}_split_map.html")
            try:
                left_rel = os.path.basename(orig_map_path)
                right_rel = os.path.basename(porto_map_path)
                create_split_map_html("Original IMN (source city)", left_rel, "Simulated Trajectory (Porto)", right_rel, split_path)
                print(f"  ✓ Split map saved: {os.path.basename(split_path)}")
            except Exception as e:
                print(f"  ⚠ Failed to create split map: {e}")
        else:
            print("  ⚠ Split map skipped (Porto map not available)")
    except Exception as e:
        print(f"  ⚠ Failed to create original-city map: {e}")


def load_datasets(paths: PathsConfig) -> Tuple[Dict, Dict]:
    print("Loading data...")
    cache_path = os.path.join(paths.results_dir, "datasets_cache.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                cached = pickle.load(f)
            imns = cached["imns"]
            poi_data = cached["poi"]
            print(f"✓ Loaded datasets from cache ({cache_path})")
            return imns, poi_data
        except Exception as e:
            print(f"⚠ Failed to load cache, re-reading datasets: {e}")

    # Load full dataset first to get user IDs
    full_imns = imn_loading.read_imn(paths.full_imn_path)
    filtered_user_ids = list(full_imns.keys())

    # Load test dataset (subset)
    imns = imn_loading.read_imn(paths.test_imn_path)
    # Keep only users that exist in both datasets
    imns = {k: imns[k] for k in filtered_user_ids if k in imns}

    poi_data = read_poi_data(paths.poi_path)
    print(f"✓ Loaded {len(imns)} IMNs and POI data for {len(poi_data)} users")

    # Persist datasets to pickle for faster reloads in iterative runs
    try:
        os.makedirs(paths.results_dir, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump({"imns": imns, "poi": poi_data, "full_user_ids": filtered_user_ids}, f)
    except Exception as e:
        print(f"⚠ Could not cache datasets: {e}")

    return imns, poi_data


def run_pipeline(paths: PathsConfig, randomness_levels: List[float], tz) -> None:
    print("Starting Individual Mobility Network Generation Process")
    print("=" * 60)

    ensure_output_structure(paths)

    print(f"Results will be saved to: {paths.results_dir}")
    print(f"  - User probability reports: {paths.results_dir}/{paths.prob_subdir}/")
    print(f"  - Timeline visualizations: {paths.results_dir}/{paths.vis_subdir}/")
    print()

    try:
        imns, poi_data = load_datasets(paths)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Ensure spatial resources (download/cache once) and keep in memory
    try:
        print("Preparing spatial resources (Porto OSM + population TIFF)...")
        G, gdf_cumulative_p = ensure_spatial_resources()
        print("✓ Spatial resources ready")
    except Exception as e:
        print(f"⚠ Spatial resources setup failed: {e}")
        G, gdf_cumulative_p = None, None

    print(f"\nProcessing {len(imns)} users...")
    print("-" * 40)
    # Process first 20 users that have POI data
    selected_user_ids = [uid for uid in imns.keys() if uid in poi_data][:20]
    print(f"Processing first {len(selected_user_ids)} users...")
    for idx, uid in enumerate(selected_user_ids, 1):
        print(f"[{idx}/{len(selected_user_ids)}] ")
        try:
            process_single_user(
                uid,
                imns[uid],
                poi_data[uid],
                randomness_levels,
                paths,
                tz,
                G=G,
                gdf_cumulative_p=gdf_cumulative_p,
            )
        except Exception as e:
            print(f"❌ Error processing user {uid}: {e}")

    print("\n" + "=" * 60)
    print("Processing complete!")
    print(f"Results saved to: {paths.results_dir}")


def main(argv: Optional[List[str]] = None):
    paths = parse_args(argv)
    run_pipeline(paths, RANDOMNESS_LEVELS, TIMEZONE)


if __name__ == "__main__":
    main()
