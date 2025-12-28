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
    # Sort by time to ensure first/last are chronologically correct
    df_sorted = df.sort_values(['trajectory_id', 'time']).reset_index(drop=True)
    grouped = df_sorted.groupby('trajectory_id')
    
    def get_od_distance(group):
        if len(group) < 2:
            return 0.0
        # After sorting, first is earliest, last is latest
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
    # Sort by time to ensure points are in chronological order
    df_sorted = df.sort_values(['trajectory_id', 'time']).reset_index(drop=True)
    grouped = df_sorted.groupby('trajectory_id')
    
    def get_path_length(group):
        if len(group) < 2:
            return 0.0
        # After sorting, coords are in chronological order
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
        # Use the city timezone when available to avoid shifting local midnight.
        try:
            tz_name = get_city_timezone(city_name)
            local_tz = None
            if tz_name:
                try:
                    import pytz
                    local_tz = pytz.timezone(tz_name)
                except Exception:
                    local_tz = None

            converted_times = []
            for traj_id in start_times.index:
                traj_data = df[df['trajectory_id'] == traj_id]
                day_str = traj_data['day_date'].iloc[0]
                day_date = pd.to_datetime(day_str).date()

                if local_tz is not None:
                    day_midnight_local = local_tz.localize(datetime.combine(day_date, datetime.min.time()))
                    day_midnight_ts = int(day_midnight_local.timestamp())
                else:
                    # Fallback to UTC if timezone unavailable
                    day_midnight_ts = int(datetime.combine(day_date, datetime.min.time(), tzinfo=timezone.utc).timestamp())
                
                relative_time = start_times[traj_id]
                unix_time = day_midnight_ts + int(relative_time)
                converted_times.append(unix_time)
            
            start_times = pd.Series(converted_times, index=start_times.index)
            # After conversion, start_times are now Unix timestamps, not relative
            is_relative_time = False
        except Exception as e:
            # If conversion fails, use a fixed reference (April 3, 2007) in UTC
            print(f"  ⚠ Warning: Could not parse day_date, using fixed reference date. Error: {e}")
            # April 3, 2007 00:00:00 UTC
            reference_ts = 1175558400  # Fixed UTC timestamp for April 3, 2007 midnight UTC
            start_times = start_times + reference_ts
            # After conversion, start_times are now Unix timestamps, not relative
            is_relative_time = False
    
    # If times are relative (<100000), compute start_hour directly from relative seconds.
    # Relative times are already in seconds since day start, so just convert to hours modulo 24.
    if is_relative_time:
        # Simple and correct: relative seconds / 3600, wrapped to 0-23
        start_hours = (start_times.values / 3600.0) % 24
        start_hours = start_hours.astype(int)
        
        # For day of week, use day_date if available
        if 'day_date' in df.columns:
            start_dows = []
            for traj_id in start_times.index:
                traj_data = df[df['trajectory_id'] == traj_id]
                day_str = traj_data['day_date'].iloc[0]
                try:
                    day_date = pd.to_datetime(day_str)
                    start_dows.append(day_date.weekday())
                except:
                    start_dows.append(np.nan)
        else:
            start_dows = [np.nan] * len(start_hours)

        temporal_features = pd.DataFrame({
            'trajectory_id': start_times.index,
            'start_time': start_times.values,  # relative seconds (not converted to Unix)
            'start_hour': start_hours,
            'start_day_of_week': start_dows,
        })
    else:
        # Absolute timestamps: use timezone-aware conversion
        dt_series = pd.to_datetime(start_times, unit='s', utc=True)
        tz_name = get_city_timezone(city_name)
        if tz_name:
            try:
                import pytz
                local_tz = pytz.timezone(tz_name)
                dt_series_local = dt_series.dt.tz_convert(local_tz)
            except (ImportError, Exception):
                dt_series_local = dt_series
        else:
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

    # Defensive casting: ensure all core metrics are 1D numeric Series.
    # On large experiments, upstream data may accidentally contain non‑scalar
    # objects (e.g. small arrays) which would otherwise cause pandas to treat
    # them as 2D and raise errors when building the DataFrame.
    cleaned_metrics = {}
    for name, series in [
        ("duration_seconds", durations),
        ("od_distance_meters", od_lengths),
        ("path_length_meters", path_lengths),
    ]:
        if isinstance(series, pd.Series):
            # Happy path: already a Series indexed by trajectory_id
            cleaned = pd.to_numeric(series, errors="coerce")
        else:
            # Fall back: try to treat input as a 1D array; if it's not 1D,
            # drop it and log a warning.
            arr = np.asarray(series)
            if arr.ndim != 1:
                print(
                    f"  ⚠ Metric '{name}' had unexpected shape {arr.shape}; "
                    f"replacing with empty Series."
                )
                cleaned = pd.Series(dtype=float)
            else:
                cleaned = pd.Series(pd.to_numeric(arr, errors="coerce"))
        cleaned_metrics[name] = cleaned

    durations = cleaned_metrics["duration_seconds"]
    od_lengths = cleaned_metrics["od_distance_meters"]
    path_lengths = cleaned_metrics["path_length_meters"]
    temporal_features = extract_temporal_features(df, city_name=city_name)
    
    # Extract origin and destination coordinates
    # Sort by time to ensure first/last are chronologically correct
    df_sorted = df.sort_values(['trajectory_id', 'time']).reset_index(drop=True)
    grouped = df_sorted.groupby('trajectory_id')
    origins = grouped[['lat', 'lon']].first().rename(columns={'lat': 'origin_lat', 'lon': 'origin_lon'})
    destinations = grouped[['lat', 'lon']].last().rename(columns={'lat': 'dest_lat', 'lon': 'dest_lon'})
    
    # Build a canonical trajectory_id index to avoid relying on implicit index
    # names, which can be lost when Series are empty or coerced.
    traj_ids = (
        df['trajectory_id']
        .drop_duplicates()
        .sort_values()
        .values
    )
    stats = pd.DataFrame({'trajectory_id': traj_ids}).set_index('trajectory_id')
    
    # Align metrics on this index
    stats['duration_seconds'] = durations.reindex(stats.index)
    stats['od_distance_meters'] = od_lengths.reindex(stats.index)
    stats['path_length_meters'] = path_lengths.reindex(stats.index)
    
    # Join spatial info
    stats = stats.join(origins, how='left').join(destinations, how='left')
    
    # Bring trajectory_id back as a column and merge temporal features
    stats = stats.reset_index()
    stats = stats.merge(temporal_features, on='trajectory_id', how='left')
    
    return stats

