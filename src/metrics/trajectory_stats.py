"""
Compute trajectory-level statistics: duration, length, temporal patterns.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timezone


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate Haversine distance between two points in meters.
    
    Args:
        lat1, lon1: First point coordinates
        lat2, lon2: Second point coordinates
        
    Returns:
        Distance in meters
    """
    R = 6371000  # Earth radius in meters
    
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2) ** 2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c


def compute_trip_duration(df: pd.DataFrame) -> pd.Series:
    """
    Compute duration for each trajectory (end_time - start_time).
    
    Args:
        df: DataFrame with columns [trajectory_id, lat, lon, time]
        
    Returns:
        Series with trajectory_id as index and duration in seconds
    """
    grouped = df.groupby('trajectory_id')['time']
    durations = grouped.max() - grouped.min()
    return durations


def compute_trip_length(df: pd.DataFrame) -> pd.Series:
    """
    Compute Haversine distance between origin and destination for each trajectory.
    
    Args:
        df: DataFrame with columns [trajectory_id, lat, lon, time]
        
    Returns:
        Series with trajectory_id as index and length in meters
    """
    grouped = df.groupby('trajectory_id')
    
    def get_od_distance(group):
        if len(group) < 2:
            return 0.0
        first = group.iloc[0]
        last = group.iloc[-1]
        return haversine_distance(first['lat'], first['lon'], 
                                 last['lat'], last['lon'])
    
    lengths = grouped.apply(get_od_distance, include_groups=False)
    return lengths


def compute_trip_path_length(df: pd.DataFrame) -> pd.Series:
    """
    Compute total path length (sum of all segments) for each trajectory.
    
    Args:
        df: DataFrame with columns [trajectory_id, lat, lon, time]
        
    Returns:
        Series with trajectory_id as index and total path length in meters
    """
    grouped = df.groupby('trajectory_id')
    
    def get_path_length(group):
        if len(group) < 2:
            return 0.0
        coords = group[['lat', 'lon']].values
        total_dist = 0.0
        for i in range(len(coords) - 1):
            total_dist += haversine_distance(coords[i][0], coords[i][1],
                                            coords[i+1][0], coords[i+1][1])
        return total_dist
    
    path_lengths = grouped.apply(get_path_length, include_groups=False)
    return path_lengths


# City to timezone mapping
CITY_TIMEZONES = {
    'milan': 'Europe/Rome',
    'porto': 'Europe/Lisbon',
    'rome': 'Europe/Rome',
    'lisbon': 'Europe/Lisbon',
}


def get_city_timezone(city_name: Optional[str]) -> Optional[str]:
    """
    Get timezone for a city name.
    
    Args:
        city_name: City name (e.g., 'milan', 'porto')
        
    Returns:
        Timezone string (e.g., 'Europe/Rome') or None if not found
    """
    if city_name is None:
        return None
    city_lower = city_name.lower()
    return CITY_TIMEZONES.get(city_lower)


def extract_temporal_features(df: pd.DataFrame, city_name: Optional[str] = None) -> pd.DataFrame:
    """
    Extract temporal features: start time, hour of day, day of week.
    
    Handles both Unix timestamps and relative times (seconds from midnight).
    If timestamps are relative (< 100000), converts using day_date column.
    
    Args:
        df: DataFrame with columns [trajectory_id, lat, lon, time]
                Optional: day_date column for relative time conversion
        city_name: City name for timezone conversion (e.g., 'milan', 'porto').
                   If None, uses UTC. This ensures hour-of-day reflects local time.
        
    Returns:
        DataFrame with trajectory-level temporal features
    """
    grouped = df.groupby('trajectory_id')
    
    start_times = grouped['time'].min()
    
    # Check if timestamps are relative (seconds from midnight) or Unix timestamps
    # If max time < 100000 (< ~27 hours), it's definitely relative time
    is_relative_time = start_times.max() < 100000
    
    if is_relative_time and 'day_date' in df.columns:
        # Convert relative times to Unix timestamps using day_date
        # IMPORTANT: Use UTC explicitly to match original trajectory timestamps
        try:
            # Convert each trajectory's relative time to Unix timestamp
            converted_times = []
            for traj_id in start_times.index:
                traj_data = df[df['trajectory_id'] == traj_id]
                day_str = traj_data['day_date'].iloc[0]
                day_date = pd.to_datetime(day_str).date()
                # Create midnight datetime in UTC (not local timezone!)
                day_midnight_utc = datetime.combine(day_date, datetime.min.time(), tzinfo=timezone.utc)
                day_midnight_ts = int(day_midnight_utc.timestamp())
                
                relative_time = start_times[traj_id]
                unix_time = day_midnight_ts + int(relative_time)
                converted_times.append(unix_time)
            
            start_times = pd.Series(converted_times, index=start_times.index)
        except Exception as e:
            # If conversion fails, use a fixed reference (April 3, 2007) in UTC
            print(f"  ⚠ Warning: Could not parse day_date, using fixed reference date. Error: {e}")
            # April 3, 2007 00:00:00 UTC
            reference_ts = 1175558400  # Fixed UTC timestamp for April 3, 2007 midnight UTC
            start_times = start_times + reference_ts
    
    # Convert timestamps to datetime (Unix timestamps are in UTC)
    dt_series = pd.to_datetime(start_times, unit='s', utc=True)
    
    # Convert to local timezone for hour extraction
    # This ensures "hour of day" reflects local time, not UTC
    tz_name = get_city_timezone(city_name)
    if tz_name:
        try:
            import pytz
            local_tz = pytz.timezone(tz_name)
            dt_series_local = dt_series.dt.tz_convert(local_tz)
        except (ImportError, Exception):
            # Fallback if pytz not available or timezone invalid - use UTC hours
            dt_series_local = dt_series
    else:
        # No city specified - use UTC hours
        dt_series_local = dt_series
    
    temporal_features = pd.DataFrame({
        'trajectory_id': start_times.index,
        'start_time': start_times.values,
        'start_hour': dt_series_local.dt.hour.values,
        'start_day_of_week': dt_series_local.dt.dayofweek.values,
    })
    
    return temporal_features


def compute_trajectory_statistics(df: pd.DataFrame, city_name: Optional[str] = None) -> pd.DataFrame:
    """
    Compute comprehensive trajectory statistics.
    
    Args:
        df: DataFrame with columns [trajectory_id, lat, lon, time]
        city_name: City name for timezone conversion (e.g., 'milan', 'porto').
                   If None, uses UTC. This ensures hour-of-day reflects local time.
        
    Returns:
        DataFrame with all trajectory-level statistics
    """
    durations = compute_trip_duration(df)
    od_lengths = compute_trip_length(df)
    path_lengths = compute_trip_path_length(df)
    temporal_features = extract_temporal_features(df, city_name=city_name)
    
    # Extract origin and destination coordinates
    grouped = df.groupby('trajectory_id')
    origins = grouped[['lat', 'lon']].first().rename(columns={'lat': 'origin_lat', 'lon': 'origin_lon'})
    destinations = grouped[['lat', 'lon']].last().rename(columns={'lat': 'dest_lat', 'lon': 'dest_lon'})
    
    # Combine all statistics
    stats = pd.DataFrame({
        'duration_seconds': durations,
        'od_distance_meters': od_lengths,
        'path_length_meters': path_lengths,
    })
    
    stats = stats.join(origins).join(destinations).reset_index()
    stats = stats.merge(temporal_features, on='trajectory_id')
    
    return stats

