"""
Compute spatial statistics: OD matrix, origin density, spatial distributions.
"""

import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional
from collections import defaultdict


def assign_to_grid(lat: float, lon: float, grid_size_m: float = 500.0, 
                   ref_lat: float = None, ref_lon: float = None) -> Tuple[int, int]:
    """
    Assign a lat/lon coordinate to a grid cell using a reference origin.
    
    Args:
        lat: Latitude
        lon: Longitude
        grid_size_m: Grid cell size in meters (default 500m)
        ref_lat: Reference latitude (grid origin). If None, uses absolute coordinates (legacy mode).
        ref_lon: Reference longitude (grid origin). If None, uses absolute coordinates (legacy mode).
        
    Returns:
        Tuple of (grid_x, grid_y) indices relative to reference origin
    """
    # Approximate conversion: 1 degree ≈ 111km at equator
    meters_per_degree_lat = 111000.0
    
    if ref_lat is None or ref_lon is None:
        # Legacy mode: use absolute coordinates (for backward compatibility, but incorrect)
        # Use individual point's latitude for longitude scaling (inconsistent, but legacy behavior)
        meters_per_degree_lon = 111000.0 * np.cos(np.radians(lat))
        grid_x = int(lon * meters_per_degree_lon / grid_size_m)
        grid_y = int(lat * meters_per_degree_lat / grid_size_m)
    else:
        # Correct mode: use relative coordinates from reference origin
        # Use reference latitude for consistent longitude scaling across all points
        meters_per_degree_lon = 111000.0 * np.cos(np.radians(ref_lat))
        lat_offset = lat - ref_lat
        lon_offset = lon - ref_lon
        # Use floor() instead of int() to handle negative offsets correctly
        # int() truncates toward zero, floor() always rounds down
        grid_x = math.floor(lon_offset * meters_per_degree_lon / grid_size_m)
        grid_y = math.floor(lat_offset * meters_per_degree_lat / grid_size_m)
    
    return (grid_x, grid_y)


def grid_to_latlon(grid_x: int, grid_y: int, grid_size_m: float = 500.0,
                   ref_lat: float = None, ref_lon: float = None) -> Tuple[float, float]:
    """
    Convert grid cell indices back to lat/lon coordinates.
    
    Args:
        grid_x: Grid X index
        grid_y: Grid Y index
        grid_size_m: Grid cell size in meters
        ref_lat: Reference latitude (grid origin). If None, assumes grid starts at (0, 0).
        ref_lon: Reference longitude (grid origin). If None, assumes grid starts at (0, 0).
        
    Returns:
        Tuple of (lat, lon) for the bottom-left corner of the grid cell
    """
    meters_per_degree_lat = 111000.0
    
    if ref_lat is None or ref_lon is None:
        # Legacy mode: assume grid starts at (0, 0)
        # Use 0 (equator) as approximation for longitude scaling
        meters_per_degree_lon = 111000.0 * np.cos(np.radians(0.0))
        lat = grid_y * grid_size_m / meters_per_degree_lat
        lon = grid_x * grid_size_m / meters_per_degree_lon
    else:
        # Correct mode: convert relative grid indices to absolute coordinates
        # Use reference latitude for consistent longitude scaling
        meters_per_degree_lon = 111000.0 * np.cos(np.radians(ref_lat))
        lat_offset = grid_y * grid_size_m / meters_per_degree_lat
        lon_offset = grid_x * grid_size_m / meters_per_degree_lon
        lat = ref_lat + lat_offset
        lon = ref_lon + lon_offset
    
    return (lat, lon)


def compute_global_reference_origin(trajectory_stats: pd.DataFrame) -> Tuple[float, float]:
    """
    Compute a global reference origin from all coordinates in trajectory statistics.
    This ensures consistent grid alignment across all spatial metrics.
    
    Args:
        trajectory_stats: DataFrame with origin_lat, origin_lon, dest_lat, dest_lon
        
    Returns:
        Tuple of (ref_lat, ref_lon) - the minimum latitude and longitude across all coordinates
    """
    all_lats = pd.concat([trajectory_stats['origin_lat'], trajectory_stats['dest_lat']])
    all_lons = pd.concat([trajectory_stats['origin_lon'], trajectory_stats['dest_lon']])
    ref_lat = all_lats.min()
    ref_lon = all_lons.min()
    return (ref_lat, ref_lon)


def compute_od_matrix(trajectory_stats: pd.DataFrame, grid_size_m: float = 500.0, 
                      ref_lat: Optional[float] = None, ref_lon: Optional[float] = None) -> pd.DataFrame:
    """
    Compute origin-destination matrix aggregated by grid cells.
    
    Args:
        trajectory_stats: DataFrame with origin_lat, origin_lon, dest_lat, dest_lon
        grid_size_m: Grid cell size in meters
        ref_lat: Reference latitude (grid origin). If None, computed from all coordinates.
        ref_lon: Reference longitude (grid origin). If None, computed from all coordinates.
        
    Returns:
        DataFrame with columns [origin_cell_x, origin_cell_y, dest_cell_x, dest_cell_y, 
                                trip_count, origin_lat, origin_lon, dest_lat, dest_lon,
                                ref_lat, ref_lon] (ref_lat/ref_lon stored for visualization)
    """
    # Use provided reference origin, or compute from all coordinates
    if ref_lat is None or ref_lon is None:
        ref_lat, ref_lon = compute_global_reference_origin(trajectory_stats)
    
    od_counts = defaultdict(int)
    od_coords = {}  # Store representative coordinates for each OD pair
    
    for _, row in trajectory_stats.iterrows():
        origin_cell = assign_to_grid(row['origin_lat'], row['origin_lon'], grid_size_m, ref_lat, ref_lon)
        dest_cell = assign_to_grid(row['dest_lat'], row['dest_lon'], grid_size_m, ref_lat, ref_lon)
        od_key = (origin_cell, dest_cell)
        od_counts[od_key] += 1
        
        # Store first occurrence coordinates as representative
        if od_key not in od_coords:
            od_coords[od_key] = {
                'origin_lat': row['origin_lat'],
                'origin_lon': row['origin_lon'],
                'dest_lat': row['dest_lat'],
                'dest_lon': row['dest_lon']
            }
    
    # Convert to DataFrame
    od_data = []
    for (origin, dest), count in od_counts.items():
        coords = od_coords[(origin, dest)]
        od_data.append({
            'origin_cell_x': origin[0],
            'origin_cell_y': origin[1],
            'dest_cell_x': dest[0],
            'dest_cell_y': dest[1],
            'trip_count': count,
            'origin_lat': coords['origin_lat'],
            'origin_lon': coords['origin_lon'],
            'dest_lat': coords['dest_lat'],
            'dest_lon': coords['dest_lon'],
            'ref_lat': ref_lat,
            'ref_lon': ref_lon
        })
    
    od_df = pd.DataFrame(od_data)
    return od_df


def compute_origin_density(trajectory_stats: pd.DataFrame, grid_size_m: float = 500.0,
                           ref_lat: Optional[float] = None, ref_lon: Optional[float] = None) -> pd.DataFrame:
    """
    Compute spatial density of trip origins.
    
    Args:
        trajectory_stats: DataFrame with origin_lat, origin_lon
        grid_size_m: Grid cell size in meters
        ref_lat: Reference latitude (grid origin). If None, computed from all coordinates.
        ref_lon: Reference longitude (grid origin). If None, computed from all coordinates.
        
    Returns:
        DataFrame with columns [cell_x, cell_y, origin_count, ref_lat, ref_lon]
        (ref_lat/ref_lon stored for visualization)
    """
    # Use provided reference origin, or compute from all coordinates for consistency
    if ref_lat is None or ref_lon is None:
        ref_lat, ref_lon = compute_global_reference_origin(trajectory_stats)
    
    cell_counts = defaultdict(int)
    
    for _, row in trajectory_stats.iterrows():
        cell = assign_to_grid(row['origin_lat'], row['origin_lon'], grid_size_m, ref_lat, ref_lon)
        cell_counts[cell] += 1
    
    # Convert to DataFrame
    density_data = []
    for cell, count in cell_counts.items():
        density_data.append({
            'cell_x': cell[0],
            'cell_y': cell[1],
            'origin_count': count,
            'ref_lat': ref_lat,
            'ref_lon': ref_lon
        })
    
    density_df = pd.DataFrame(density_data).sort_values('origin_count', ascending=False)
    return density_df


def compute_destination_density(trajectory_stats: pd.DataFrame, grid_size_m: float = 500.0,
                               ref_lat: Optional[float] = None, ref_lon: Optional[float] = None) -> pd.DataFrame:
    """
    Compute spatial density of trip destinations.
    
    Args:
        trajectory_stats: DataFrame with dest_lat, dest_lon
        grid_size_m: Grid cell size in meters
        ref_lat: Reference latitude (grid origin). If None, computed from all coordinates.
        ref_lon: Reference longitude (grid origin). If None, computed from all coordinates.
        
    Returns:
        DataFrame with columns [cell_x, cell_y, dest_count, ref_lat, ref_lon]
        (ref_lat/ref_lon stored for visualization)
    """
    # Use provided reference origin, or compute from all coordinates for consistency
    if ref_lat is None or ref_lon is None:
        ref_lat, ref_lon = compute_global_reference_origin(trajectory_stats)
    
    cell_counts = defaultdict(int)
    
    for _, row in trajectory_stats.iterrows():
        cell = assign_to_grid(row['dest_lat'], row['dest_lon'], grid_size_m, ref_lat, ref_lon)
        cell_counts[cell] += 1
    
    # Convert to DataFrame
    density_data = []
    for cell, count in cell_counts.items():
        density_data.append({
            'cell_x': cell[0],
            'cell_y': cell[1],
            'dest_count': count,
            'ref_lat': ref_lat,
            'ref_lon': ref_lon
        })
    
    density_df = pd.DataFrame(density_data).sort_values('dest_count', ascending=False)
    return density_df


def compute_spatial_coverage(trajectory_stats: pd.DataFrame) -> Dict[str, float]:
    """
    Compute spatial coverage statistics: bounding box, area covered.
    
    Args:
        trajectory_stats: DataFrame with origin_lat, origin_lon, dest_lat, dest_lon
        
    Returns:
        Dictionary with spatial coverage metrics
    """
    all_lats = pd.concat([trajectory_stats['origin_lat'], trajectory_stats['dest_lat']])
    all_lons = pd.concat([trajectory_stats['origin_lon'], trajectory_stats['dest_lon']])
    
    bbox = {
        'min_lat': all_lats.min(),
        'max_lat': all_lats.max(),
        'min_lon': all_lons.min(),
        'max_lon': all_lons.max(),
    }
    
    # Approximate area in km²
    lat_range_km = (bbox['max_lat'] - bbox['min_lat']) * 111.0
    lon_range_km = (bbox['max_lon'] - bbox['min_lon']) * 111.0 * np.cos(np.radians(all_lats.mean()))
    area_km2 = lat_range_km * lon_range_km
    
    return {
        **bbox,
        'lat_range_km': lat_range_km,
        'lon_range_km': lon_range_km,
        'area_km2': area_km2,
    }

