import os
import pickle
import requests
import osmnx as ox
from typing import Tuple
import pandas as pd

# Spatial constants/resources
PORTO_CENTER = (41.1494512, -8.6107884)
PORTO_RADIUS_M = 10000
SEDAC_TIFF_URL = "https://data.ghg.center/sedac-popdensity-yeargrid5yr-v4.11/gpw_v4_population_density_rev11_2020_30_sec_2020.tif"


# ** Prepare OSM graph and population raster (used for node sampling)
def ensure_spatial_resources(
    data_dir: str,
    generate_cumulative_map_fn,
) -> Tuple[object, pd.DataFrame]:
    os.makedirs(data_dir, exist_ok=True)
    graphml_path = os.path.join(data_dir, "porto_drive.graphml")
    tiff_path = os.path.join(data_dir, "gpw_v4_population_density_rev11_2020_30_sec_2020.tif")
    pop_pickle = os.path.join(data_dir, "porto_population_cumulative.pkl")

    # OSM graph
    if os.path.exists(graphml_path):
        G = ox.load_graphml(graphml_path)
    else:
        G = ox.graph.graph_from_point(PORTO_CENTER, dist=PORTO_RADIUS_M, network_type="drive", simplify=False)
        G = ox.routing.add_edge_speeds(G)
        G = ox.routing.add_edge_travel_times(G)
        ox.save_graphml(G, graphml_path)

    # TIFF
    if not os.path.exists(tiff_path):
        resp = requests.get(SEDAC_TIFF_URL)
        resp.raise_for_status()
        with open(tiff_path, 'wb') as f:
            f.write(resp.content)

    # Cumulative population map
    if os.path.exists(pop_pickle):
        with open(pop_pickle, 'rb') as f:
            gdf_cumulative_p = pickle.load(f)
    else:
        gdf_cumulative_p = generate_cumulative_map_fn(tiff_path, G)
        with open(pop_pickle, 'wb') as f:
            pickle.dump(gdf_cumulative_p, f)

    return G, gdf_cumulative_p


