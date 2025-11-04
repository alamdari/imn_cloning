from typing import Tuple
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon
import osmnx as ox


def fetch_population_data_from_tiff(tiff_file: str, bbox: Tuple[float, float, float, float]) -> gpd.GeoDataFrame:
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
            population_density = float(out_image[row, col])
            x, y = out_transform * (col, row)
            x2, y2 = out_transform * (col + 1, row + 1)
            cell_polygon = Polygon([(x, y), (x2, y), (x2, y2), (x, y2), (x, y)])
            population_data.append({"population_density": population_density, "geometry": cell_polygon})
        population_gdf = gpd.GeoDataFrame(population_data, geometry="geometry")
        population_gdf.set_crs(src.crs, inplace=True)
    return population_gdf


def add_intersecting_nodes(gdf: gpd.GeoDataFrame, graph) -> gpd.GeoDataFrame:
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


def generate_cumulative_map(population_tiff_file: str, G) -> gpd.GeoDataFrame:
    bbox = ox.graph_to_gdfs(G, nodes=True, edges=False).total_bounds
    population_gdf = fetch_population_data_from_tiff(population_tiff_file, bbox)
    gdf_extended = add_intersecting_nodes(population_gdf, G)
    gdf_extended['p'] = gdf_extended['population_density'] / gdf_extended.population_density.sum()
    gdf_extended['cumulative_p'] = gdf_extended['p'].cumsum()
    return gdf_extended[['cumulative_p', 'intersecting_nodes']]


