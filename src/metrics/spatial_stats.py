"""
Compute spatial statistics: OD matrix, origin density, spatial distributions.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict
from collections import defaultdict


def assign_to_grid(lat: float, lon: float, grid_size_m: float = 500.0) -> Tuple[int, int]:
    """
    Assign a lat/lon coordinate to a grid cell.
    
    Args:
        lat: Latitude
        lon: Longitude
        grid_size_m: Grid cell size in meters (default 500m)
        
    Returns:
        Tuple of (grid_x, grid_y) indices
    """
    # Approximate conversion: 1 degree ≈ 111km at equator
    # For more accuracy, this should be adjusted by latitude
    meters_per_degree_lat = 111000.0
    meters_per_degree_lon = 111000.0 * np.cos(np.radians(lat))
    
    grid_x = int(lon * meters_per_degree_lon / grid_size_m)
    grid_y = int(lat * meters_per_degree_lat / grid_size_m)
    
    return (grid_x, grid_y)


def compute_od_matrix(trajectory_stats: pd.DataFrame, grid_size_m: float = 500.0) -> pd.DataFrame:
    """
    Compute origin-destination matrix aggregated by grid cells.
    
    Args:
        trajectory_stats: DataFrame with origin_lat, origin_lon, dest_lat, dest_lon
        grid_size_m: Grid cell size in meters
        
    Returns:
        DataFrame with columns [origin_cell, dest_cell, trip_count]
    """
    od_counts = defaultdict(int)
    
    for _, row in trajectory_stats.iterrows():
        origin_cell = assign_to_grid(row['origin_lat'], row['origin_lon'], grid_size_m)
        dest_cell = assign_to_grid(row['dest_lat'], row['dest_lon'], grid_size_m)
        od_counts[(origin_cell, dest_cell)] += 1
    
    # Convert to DataFrame
    od_data = []
    for (origin, dest), count in od_counts.items():
        od_data.append({
            'origin_cell_x': origin[0],
            'origin_cell_y': origin[1],
            'dest_cell_x': dest[0],
            'dest_cell_y': dest[1],
            'trip_count': count
        })
    
    od_df = pd.DataFrame(od_data)
    return od_df


def compute_origin_density(trajectory_stats: pd.DataFrame, grid_size_m: float = 500.0) -> pd.DataFrame:
    """
    Compute spatial density of trip origins.
    
    Args:
        trajectory_stats: DataFrame with origin_lat, origin_lon
        grid_size_m: Grid cell size in meters
        
    Returns:
        DataFrame with columns [cell_x, cell_y, origin_count]
    """
    cell_counts = defaultdict(int)
    
    for _, row in trajectory_stats.iterrows():
        cell = assign_to_grid(row['origin_lat'], row['origin_lon'], grid_size_m)
        cell_counts[cell] += 1
    
    # Convert to DataFrame
    density_data = []
    for cell, count in cell_counts.items():
        density_data.append({
            'cell_x': cell[0],
            'cell_y': cell[1],
            'origin_count': count
        })
    
    density_df = pd.DataFrame(density_data).sort_values('origin_count', ascending=False)
    return density_df


def compute_destination_density(trajectory_stats: pd.DataFrame, grid_size_m: float = 500.0) -> pd.DataFrame:
    """
    Compute spatial density of trip destinations.
    
    Args:
        trajectory_stats: DataFrame with dest_lat, dest_lon
        grid_size_m: Grid cell size in meters
        
    Returns:
        DataFrame with columns [cell_x, cell_y, dest_count]
    """
    cell_counts = defaultdict(int)
    
    for _, row in trajectory_stats.iterrows():
        cell = assign_to_grid(row['dest_lat'], row['dest_lon'], grid_size_m)
        cell_counts[cell] += 1
    
    # Convert to DataFrame
    density_data = []
    for cell, count in cell_counts.items():
        density_data.append({
            'cell_x': cell[0],
            'cell_y': cell[1],
            'dest_count': count
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

