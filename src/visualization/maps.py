from typing import Dict, Any, List, Tuple
import folium
import json
import os


def generate_interactive_porto_map_multi(user_id: int, per_day_data: Dict, G, out_html_path: str) -> None:
    if not per_day_data:
        return
    first_day = next(iter(per_day_data.keys()))
    first_traj = per_day_data[first_day]['trajectory']
    if first_traj:
        mid_idx = len(first_traj) // 2
        center_lat, center_lon = first_traj[mid_idx][0], first_traj[mid_idx][1]
    else:
        any_node = next(iter(per_day_data[first_day]['pseudo_map_loc'].values()))
        center_lat, center_lon = G.nodes[any_node]['y'], G.nodes[any_node]['x']

    m = folium.Map(location=[center_lat, center_lon], zoom_start=13, tiles=None)
    folium.TileLayer('OpenStreetMap', opacity=0.6, control=False).add_to(m)

    static_hw = folium.FeatureGroup(name="Home/Work", show=True)
    added_hw_nodes = set()

    for day, content in per_day_data.items():
        day_label = str(day)
        fg = folium.FeatureGroup(name=f"Day {day_label}", show=False)  # Unchecked by default
        legs_coords: List[List[Tuple[float, float, int]]] = content.get('legs_coords', [])
        if legs_coords:
            n_legs = len(legs_coords)
            if n_legs > 0:
                for i, leg in enumerate(legs_coords):
                    coords = [(lat, lon) for lat, lon, _ in leg]
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
            marker_color = {
                'home': 'darkblue', 'work': 'orange', 'eat': 'green', 'utility': 'purple', 'transit': 'red', 'unknown': 'gray',
                'school': 'yellow', 'shop': 'pink', 'leisure': 'lightgreen', 'health': 'lightcoral', 'admin': 'lightblue', 'finance': 'gold'
            }.get(act, 'black')
            popup_html = f"""
            <div style='font-size:12px;'>
                <b>User:</b> {user_id}<br/>
                <b>Day:</b> {day_label}<br/>
                <b>Stay index:</b> {idx}<br/>
                <b>Activity:</b> {act}<br/>
                <b>OSM node:</b> {node}
            </div>
            """
            if act in ("home", "work"):
                if node not in added_hw_nodes:
                    folium.CircleMarker([node_lat, node_lon], radius=8, color='white', fill=True, fill_color=marker_color,
                                        fill_opacity=0.8, weight=2, opacity=1, popup=folium.Popup(popup_html, max_width=300), tooltip=f"{act}").add_to(static_hw)
                    added_hw_nodes.add(node)
            else:
                folium.CircleMarker([node_lat, node_lon], radius=6, color='white', fill=True, fill_color=marker_color,
                                    fill_opacity=0.8, weight=2, opacity=1, popup=folium.Popup(popup_html, max_width=300), tooltip=f"{act}").add_to(fg)

        fg.add_to(m)

    static_hw.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add distance measurement tool
    from folium.plugins import MeasureControl
    MeasureControl(primary_length_unit='kilometers', secondary_length_unit='meters').add_to(m)
    
    import os
    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    m.save(out_html_path)



def load_original_trajectories(user_id: int, trajectories_dir: str = "data/trajectories") -> Dict:
    """Load original GPS trajectories from JSON file."""
    traj_file = os.path.join(trajectories_dir, f"user_{user_id}_trajectories.json")
    if not os.path.exists(traj_file):
        return None
    
    with open(traj_file, 'r') as f:
        data = json.load(f)
    return data.get('trajectories', {})


def organize_trajectories_by_day(trajectories: Dict, tz) -> Dict:
    """Organize trajectories by day based on start time."""
    from datetime import datetime
    from collections import defaultdict
    
    trajectories_by_day = defaultdict(list)
    
    for traj_id, traj_data in trajectories.items():
        start_time = traj_data.get('_start_time', 0)
        if start_time:
            start_dt = datetime.fromtimestamp(start_time, tz)
            day_date = start_dt.date()
            trajectories_by_day[day_date].append((traj_id, traj_data))
    
    return dict(trajectories_by_day)


def generate_interactive_original_city_map(user_id: int, enriched_imn: Dict, stays_by_day: Dict, out_html_path: str, 
                                          trajectories_dir: str = "data/trajectories", tz=None) -> None:
    """Create an interactive map for the original city using actual GPS trajectories.
    Shows original trajectories organized by day with layer controls, matching Porto map structure.
    """
    if not enriched_imn or not enriched_imn.get('locations'):
        return
    
    # Load original trajectories from JSON
    original_trajectories = load_original_trajectories(user_id, trajectories_dir)
    
    # Organize trajectories by day if available
    trajectories_by_day = {}
    if original_trajectories and tz:
        trajectories_by_day = organize_trajectories_by_day(original_trajectories, tz)
    
    # Determine center by averaging coordinates
    try:
        lats = [loc['coordinates'][1] for loc in enriched_imn['locations'].values()]
        lons = [loc['coordinates'][0] for loc in enriched_imn['locations'].values()]
        center_lat = float(sum(lats) / len(lats))
        center_lon = float(sum(lons) / len(lons))
    except Exception:
        first_loc = next(iter(enriched_imn['locations'].values()))
        center_lat = first_loc['coordinates'][1]
        center_lon = first_loc['coordinates'][0]

    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=None)
    folium.TileLayer('OpenStreetMap', opacity=0.5, control=False).add_to(m)

    # Activity color palette
    def resolve_color(act: str) -> str:
        palette = {
            'home': 'darkblue', 'work': 'orange', 'eat': 'green', 'utility': 'purple', 'transit': 'red', 'unknown': 'gray',
            'school': 'yellow', 'shop': 'pink', 'leisure': 'lightgreen', 'health': 'lightcoral', 'admin': 'lightblue', 'finance': 'gold'
        }
        return palette.get(str(act).lower() if act is not None else 'unknown', 'black')

    # Static layer for home/work locations
    static_hw = folium.FeatureGroup(name="Home/Work", show=True)
    added_hw_nodes = set()
    
    home_id = enriched_imn.get('home')
    work_id = enriched_imn.get('work')
    
    # Add day-level layers (same structure as Porto map)
    for day_date in sorted(stays_by_day.keys()):
        day_label = str(day_date)
        fg = folium.FeatureGroup(name=f"Day {day_label}", show=False)  # Unchecked by default
        
        day_stays = stays_by_day[day_date]
        
        # Draw trajectories using actual GPS data if available for this day
        if day_date in trajectories_by_day:
            # Use actual GPS trajectories
            for traj_id, traj_data in trajectories_by_day[day_date]:
                gps_points = traj_data.get('object', [])
                if not gps_points:
                    continue
                
                # Draw trajectory as polyline
                coords = [(point[1], point[0]) for point in gps_points]  # (lat, lon)
                
                folium.PolyLine(
                    coords, 
                    color="#1f77b4",  # Same blue as Porto map
                    weight=2, 
                    opacity=0.6, 
                    tooltip=f"Trajectory {traj_id} - {day_label}"
                ).add_to(fg)
        else:
            # Fallback: draw simple lines between consecutive stays
            for i in range(len(day_stays) - 1):
                stay_from = day_stays[i]
                stay_to = day_stays[i + 1]
                
                loc_from = enriched_imn['locations'].get(stay_from.location_id)
                loc_to = enriched_imn['locations'].get(stay_to.location_id)
                
                if loc_from and loc_to:
                    coords = [
                        (loc_from['coordinates'][1], loc_from['coordinates'][0]),
                        (loc_to['coordinates'][1], loc_to['coordinates'][0])
                    ]
                    folium.PolyLine(
                        coords, 
                        color="#1f77b4", 
                        weight=2, 
                        opacity=0.6, 
                        tooltip=f"Trip {i+1} - {day_label}"
                    ).add_to(fg)
        
        # Add stay markers for this day
        for idx, stay in enumerate(day_stays):
            loc = enriched_imn['locations'].get(stay.location_id)
            if not loc:
                continue
                
            lat, lon = loc['coordinates'][1], loc['coordinates'][0]
            act = loc.get('activity_label', 'unknown')
            marker_color = resolve_color(act)
            
            popup_html = f"""
            <div style='font-size:12px;'>
                <b>User:</b> {user_id}<br/>
                <b>Day:</b> {day_label}<br/>
                <b>Stay index:</b> {idx}<br/>
                <b>Location ID:</b> {stay.location_id}<br/>
                <b>Activity:</b> {act}<br/>
                <b>Duration:</b> {stay.duration}s
            </div>
            """
            
            if act in ("home", "work"):
                if stay.location_id not in added_hw_nodes:
                    folium.CircleMarker(
                        [lat, lon], 
                        radius=8, 
                        color='white', 
                        fill=True, 
                        fill_color=marker_color,
                        fill_opacity=0.8, 
                        weight=2, 
                        opacity=1, 
                        popup=folium.Popup(popup_html, max_width=300), 
                        tooltip=f"{act}"
                    ).add_to(static_hw)
                    added_hw_nodes.add(stay.location_id)
            else:
                folium.CircleMarker(
                    [lat, lon], 
                    radius=6, 
                    color='white', 
                    fill=True, 
                    fill_color=marker_color,
                    fill_opacity=0.8, 
                    weight=2, 
                    opacity=1, 
                    popup=folium.Popup(popup_html, max_width=300), 
                    tooltip=f"{act}"
                ).add_to(fg)
        
        fg.add_to(m)
    
    static_hw.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Add distance measurement tool
    from folium.plugins import MeasureControl
    MeasureControl(primary_length_unit='kilometers', secondary_length_unit='meters').add_to(m)

    os.makedirs(os.path.dirname(out_html_path), exist_ok=True)
    m.save(out_html_path)


def create_split_map_html(left_title: str, left_src: str, right_title: str, right_src: str, out_html_path: str) -> None:
    """Compose a simple split view HTML embedding two maps side-by-side."""
    import os
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

