from typing import Dict, List, Tuple, Optional
import numpy as np
from ..spatial.mapping import haversine_distance
import os
import json


class Stay:
    def __init__(self, location_id: int, activity_label: str, start_time: int, end_time: int):
        self.location_id = location_id
        self.activity_label = activity_label
        self.start_time = start_time
        self.end_time = end_time
        self.duration = self.end_time - self.start_time if self.end_time is not None and self.start_time is not None else None

    def set_end_time(self, end_time: int):
        self.end_time = end_time
        if self.start_time is not None:
            self.duration = self.end_time - self.start_time


def compute_imn_scale(imn: Dict) -> float:
    """
    Compute IMN spatial scale: mean distance from home to all other locations.
    This represents the overall spatial scale of the user's mobility.
    
    Args:
        imn: Individual Mobility Network (enriched)
        
    Returns:
        Mean distance from home to all other locations in km, or 0.0 if no other locations
    """
    home_id = imn.get('home')
    if home_id is None:
        return 0.0
    
    home_coords = imn['locations'][home_id]['coordinates']
    total_dist = 0.0
    num_other_locs = 0
    
    for loc_id, loc_data in imn.get('locations', {}).items():
        if loc_id != home_id:
            dist_from_home = haversine_distance(
                home_coords[0], home_coords[1],
                loc_data['coordinates'][0], loc_data['coordinates'][1]
            ) / 1000.0  # Convert to km
            total_dist += dist_from_home
            num_other_locs += 1
    
    if num_other_locs > 0:
        return total_dist / num_other_locs
    else:
        return 0.0


def compute_radius_of_gyration(trajectory_points: List, home_coords: Tuple[float, float]) -> float:
    """
    Compute radius of gyration for a trajectory to detect if it's a real trip or GPS noise.
    
    Radius of gyration measures the spatial spread of points around the center of mass.
    A small radius indicates points clustered around a location (GPS noise), while
    a large radius indicates real movement.
    
    Args:
        trajectory_points: List of [lon, lat, time] points
        home_coords: Home coordinates [lon, lat]
        
    Returns:
        Radius of gyration in km, or 0.0 if insufficient points
    """
    if len(trajectory_points) < 3:
        return 0.0
    
    # Extract coordinates
    coords = []
    for point in trajectory_points:
        if len(point) >= 2:
            lon, lat = point[0], point[1]
            coords.append((lat, lon))
    
    if len(coords) < 3:
        return 0.0
    
    # Compute center of mass
    center_lat = sum(c[0] for c in coords) / len(coords)
    center_lon = sum(c[1] for c in coords) / len(coords)
    
    # Compute radius of gyration: sqrt(mean(squared distances from center))
    squared_distances = []
    for lat, lon in coords:
        dist = haversine_distance(center_lat, center_lon, lat, lon) / 1000.0  # km
        squared_distances.append(dist ** 2)
    
    if not squared_distances:
        return 0.0
    
    mean_squared_dist = sum(squared_distances) / len(squared_distances)
    radius_of_gyration = np.sqrt(mean_squared_dist)
    
    return radius_of_gyration


def find_intermediate_location_for_home_to_home(
    imn: Dict,
    user_id: Optional[int],
    imn_scale: float,
    trip_start_time: int,
    trip_end_time: int,
    trajectories_dir: str = "data/trajectories"
) -> Optional[str]:
    """
    Find an intermediate location to split a home-to-home trip.
    
    Uses Option 2 (trajectory-based) first with strict threshold, then falls back to Option 1 (distance-based).
    Only proceeds if trajectory quality check (radius of gyration) indicates a real trip.
    
    Args:
        imn: Individual Mobility Network (enriched)
        user_id: User ID for loading trajectory data
        imn_scale: IMN spatial scale (mean distance from home to all locations) in km
        trip_start_time: Start time of the home-to-home trip (Unix timestamp)
        trip_end_time: End time of the home-to-home trip (Unix timestamp)
        trajectories_dir: Directory containing trajectory JSON files
        
    Returns:
        Location ID to use as intermediate stop, or None if no suitable location found
    """
    if imn_scale <= 0:
        return None
    
    home_id = imn.get('home')
    if home_id is None:
        return None
    
    home_coords = imn['locations'][home_id]['coordinates']
    user_locations = imn.get('locations', {})
    
    # Option 2: Try trajectory-based approach (STRICT - only if within 0.5 × imn_scale)
    if user_id is not None:
        traj_file = os.path.join(trajectories_dir, f"user_{user_id}_trajectories.json")
        if os.path.exists(traj_file):
            try:
                with open(traj_file, 'r') as f:
                    traj_data = json.load(f)
                
                trajectories = traj_data.get('trajectories', {})
                
                # Find trajectory that matches this time window (with tolerance)
                best_trajectory = None
                best_traj_points = None
                min_time_diff = float('inf')
                
                for traj_id, traj_obj in trajectories.items():
                    if 'object' not in traj_obj:
                        continue
                    
                    points = traj_obj['object']  # [[lon, lat, time], ...]
                    if len(points) < 3:  # Need at least 3 points for radius of gyration
                        continue
                    
                    # Check if trajectory overlaps with trip time window (with 1 hour tolerance)
                    traj_start = points[0][2] if len(points[0]) > 2 else None
                    traj_end = points[-1][2] if len(points[-1]) > 2 else None
                    
                    if traj_start is None or traj_end is None:
                        continue
                    
                    # Check overlap
                    if not (traj_end < trip_start_time - 3600 or traj_start > trip_end_time + 3600):
                        # This trajectory might be relevant
                        time_diff = abs((traj_start + traj_end) / 2 - (trip_start_time + trip_end_time) / 2)
                        if time_diff < min_time_diff:
                            min_time_diff = time_diff
                            best_trajectory = traj_obj
                            best_traj_points = points
                
                # If we found a matching trajectory, check quality and find intermediate
                if best_traj_points is not None:
                    # Check radius of gyration to filter GPS noise
                    # (This is a double-check; main quality check happens in split_home_to_home_trips_in_original_stays)
                    radius_gyration = compute_radius_of_gyration(best_traj_points, home_coords)
                    
                    # Require minimum radius of gyration (1 km) to be considered a real trip
                    # This filters out GPS noise where points cluster around a location
                    min_radius_km = 1.0  # 1 km
                    if radius_gyration < min_radius_km:
                        # GPS noise - skip this trip
                        return None
                    
                    # Find farthest point from home in this trajectory
                    farthest_point = None
                    farthest_dist = 0.0
                    
                    for point in best_traj_points:
                        if len(point) < 2:
                            continue
                        lon, lat = point[0], point[1]
                        dist = haversine_distance(
                            home_coords[0], home_coords[1],
                            lat, lon
                        ) / 1000.0  # Convert to km
                        
                        if dist > farthest_dist:
                            farthest_dist = dist
                            farthest_point = (lat, lon)
                    
                    # If we found a farthest point, find closest user location
                    if farthest_point is not None:
                        farthest_lat, farthest_lon = farthest_point
                        closest_location = None
                        closest_dist = float('inf')
                        
                        for loc_id, loc_data in user_locations.items():
                            if loc_id == home_id:
                                continue
                            
                            loc_coords = loc_data['coordinates']
                            dist = haversine_distance(
                                farthest_lat, farthest_lon,
                                loc_coords[0], loc_coords[1]
                            ) / 1000.0  # Convert to km
                            
                            if dist < closest_dist:
                                closest_dist = dist
                                closest_location = loc_id
                        
                        # STRICT: Only use if closest location is within 0.5 × imn_scale
                        if closest_location is not None and closest_dist <= 0.5 * imn_scale:
                            return closest_location
            except (json.JSONDecodeError, KeyError, IOError):
                # If trajectory loading fails, fall through to Option 1
                pass
    
    # Option 1: Fallback to distance-based approach (with controlled range)
    # Find locations where distance from home is between 0.3 × imn_scale and 0.7 × imn_scale
    candidates = []
    min_dist = 0.3 * imn_scale  # Stricter lower bound
    max_dist = 0.7 * imn_scale  # Stricter upper bound
    
    for loc_id, loc_data in user_locations.items():
        if loc_id == home_id:
            continue
        
        loc_coords = loc_data['coordinates']
        dist_from_home = haversine_distance(
            home_coords[0], home_coords[1],
            loc_coords[0], loc_coords[1]
        ) / 1000.0  # Convert to km
        
        if min_dist <= dist_from_home <= max_dist:
            candidates.append(loc_id)
    
    # Randomly select from candidates if any exist
    if candidates:
        return np.random.choice(candidates)
    
    # No suitable location found - skip splitting (don't force it)
    return None


def split_home_to_home_trips_in_original_stays(
    stays_by_day: Dict,
    imn: Dict,
    user_id: Optional[int],
    trajectories_dir: str = "data/trajectories"
) -> Dict:
    """
    Detect and split home-to-home trips in original stays by inserting intermediate stops.
    This happens BEFORE temporal generation, so temporal generation sees "home → intermediate → home"
    instead of "home → home".
    
    The key insight: When we have consecutive home stays with no gap, they represent a home-to-home trip.
    We need to detect when a stay's duration represents a trip (not just being at home).
    
    Args:
        stays_by_day: Dictionary mapping dates to lists of Stay objects
        imn: Individual Mobility Network (enriched)
        user_id: User ID for loading trajectory data
        trajectories_dir: Directory containing trajectory JSON files
        
    Returns:
        Modified stays_by_day with home-to-home trips split (if intermediate location found)
    """
    # Compute IMN spatial scale
    imn_scale = compute_imn_scale(imn)
    if imn_scale <= 0:
        return stays_by_day  # Can't split without valid scale
    
    home_id = imn.get('home')
    if home_id is None:
        return stays_by_day
    
    modified_stays_by_day = {}
    
    for day_date, day_stays in stays_by_day.items():
        if len(day_stays) < 2:
            modified_stays_by_day[day_date] = day_stays
            continue
        
        modified_stays = []
        i = 0
        
        while i < len(day_stays):
            current_stay = day_stays[i]
            
            # Check if this is a "home" stay
            if str(current_stay.activity_label).lower() == 'home' and i < len(day_stays) - 1:
                next_stay = day_stays[i + 1]
                
                # Check if next stay is also "home" (home-to-home trip)
                if str(next_stay.activity_label).lower() == 'home':
                    # QUALITY CHECK: Only process home-to-home trips with sufficient spatial spread
                    # Check radius of gyration to filter GPS noise (ROG < 1km = noise)
                    trip_has_quality = False
                    home_coords = imn['locations'][home_id]['coordinates']
                    
                    if user_id is not None:
                        traj_file = os.path.join(trajectories_dir, f"user_{user_id}_trajectories.json")
                        if os.path.exists(traj_file):
                            try:
                                with open(traj_file, 'r') as f:
                                    traj_data = json.load(f)
                                
                                trajectories = traj_data.get('trajectories', {})
                                
                                # Find trajectory matching this home-to-home trip time window
                                trip_start = current_stay.start_time
                                trip_end = current_stay.end_time
                                
                                best_traj_points = None
                                min_time_diff = float('inf')
                                
                                for traj_id, traj_obj in trajectories.items():
                                    if 'object' not in traj_obj:
                                        continue
                                    
                                    points = traj_obj['object']
                                    if len(points) < 3:  # Need at least 3 points for ROG
                                        continue
                                    
                                    traj_start = points[0][2] if len(points[0]) > 2 else None
                                    traj_end = points[-1][2] if len(points[-1]) > 2 else None
                                    
                                    if traj_start is None or traj_end is None:
                                        continue
                                    
                                    # Check overlap with trip time window (1 hour tolerance)
                                    if not (traj_end < trip_start - 3600 or traj_start > trip_end + 3600):
                                        time_diff = abs((traj_start + traj_end) / 2 - (trip_start + trip_end) / 2)
                                        if time_diff < min_time_diff:
                                            min_time_diff = time_diff
                                            best_traj_points = points
                                
                                # Compute radius of gyration for this home-to-home trip
                                if best_traj_points is not None:
                                    radius_gyration = compute_radius_of_gyration(best_traj_points, home_coords)
                                    # Require ROG >= 1km to be considered a real trip (not GPS noise)
                                    if radius_gyration >= 1.0:  # 1 km
                                        trip_has_quality = True
                            except (json.JSONDecodeError, KeyError, IOError):
                                # If trajectory loading fails, skip quality check (conservative: don't split)
                                pass
                    
                    # Only attempt to split if trip has sufficient spatial quality
                    if trip_has_quality:
                        stay_duration = current_stay.end_time - current_stay.start_time if current_stay.end_time and current_stay.start_time else 0
                        
                        # Try to find intermediate location
                        intermediate_loc = find_intermediate_location_for_home_to_home(
                            imn, user_id, imn_scale, 
                            current_stay.start_time, current_stay.end_time,
                            trajectories_dir
                        )
                        
                        if intermediate_loc is not None:
                            # Split the trip: home → intermediate → home
                            # Insert intermediate stay at midpoint of the stay duration
                            # Use current_stay's end_time as the split point (where trip would end)
                            midpoint_time = current_stay.start_time + (stay_duration // 2)
                            intermediate_duration = min(300, stay_duration // 4)  # Stay for 5 min or 1/4 of trip
                            
                            # Get activity label for intermediate location
                            intermediate_activity = imn['locations'].get(intermediate_loc, {}).get('activity_label', 'unknown')
                            
                            # Split current_stay into two parts: before and after intermediate
                            # Part 1: home stay from start to midpoint
                            stay_part1 = Stay(
                                location_id=current_stay.location_id,
                                activity_label=current_stay.activity_label,
                                start_time=current_stay.start_time,
                                end_time=midpoint_time
                            )
                            
                            # Intermediate stay
                            intermediate_stay = Stay(
                                location_id=intermediate_loc,
                                activity_label=intermediate_activity,
                                start_time=midpoint_time,
                                end_time=midpoint_time + intermediate_duration
                            )
                            
                            # Part 2: home stay from end of intermediate to original end
                            stay_part2 = Stay(
                                location_id=current_stay.location_id,
                                activity_label=current_stay.activity_label,
                                start_time=midpoint_time + intermediate_duration,
                                end_time=current_stay.end_time
                            )
                            
                            modified_stays.append(stay_part1)  # First part of home stay
                            modified_stays.append(intermediate_stay)  # Intermediate stay
                            modified_stays.append(stay_part2)  # Second part of home stay
                            # Skip next_stay since we've already handled the transition
                            i += 1  # Only skip current_stay, next_stay will be processed normally
                            continue
            
            # No splitting needed, keep original stay
            modified_stays.append(current_stay)
            i += 1
        
        modified_stays_by_day[day_date] = modified_stays
    
    return modified_stays_by_day


def read_stays_from_trips(trips: List[Tuple], locations: Dict) -> List[Stay]:
    """
    Extract stays from trips.
    
    Creates stays at both the origin of the first trip and destinations of all trips.
    This ensures we capture the initial location (e.g., home) before the first trip.
    """
    stays = []
    
    if not trips:
        return stays
    
    # Create stay at origin of first trip (from trip start time)
    first_from_id, first_to_id, first_st, first_et = trips[0]
    first_from_label = locations[first_from_id].get('activity_label', 'unknown')
    first_origin_stay = Stay(first_from_id, first_from_label, first_st, first_et)
    stays.append(first_origin_stay)
    
    # Create stays at destinations of all trips
    for from_id, to_id, st, et in trips:
        activity_label = locations[to_id].get('activity_label', 'unknown')
        stays.append(Stay(to_id, activity_label, et, None))
    
    # Set end times for all stays
    for i in range(len(stays)):
        current_stay = stays[i]
        if i < len(stays) - 1:
            # End time is start of next stay
            next_stay_start = stays[i + 1].start_time
            if next_stay_start is not None:
                current_stay.set_end_time(next_stay_start)
        else:
            # Last stay: set end time to start + 1 hour (default)
            if current_stay.start_time is not None:
                current_stay.set_end_time(current_stay.start_time + 3600)
    
    return stays


def extract_stays_by_day(stays: List[Stay], tz) -> Dict:
    from collections import defaultdict
    from datetime import datetime, timedelta
    stays_by_day = defaultdict(list)
    for stay in stays:
        if stay.start_time is None or stay.end_time is None:
            continue
        start_dt = datetime.fromtimestamp(stay.start_time, tz)
        end_dt = datetime.fromtimestamp(stay.end_time, tz)
        current_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = current_dt + timedelta(days=1)
        while current_dt < end_dt:
            day_start = max(start_dt, current_dt)
            day_end = min(end_dt, end_of_day)
            day_stay = Stay(stay.location_id, stay.activity_label, int(day_start.timestamp()), int(day_end.timestamp()))
            stays_by_day[current_dt.date()].append(day_stay)
            current_dt = end_of_day
            end_of_day = current_dt + timedelta(days=1)
    return stays_by_day
