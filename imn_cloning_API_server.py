import pandas as pd
import numpy as np
import osmnx as ox
import os

import imn_generation
import imn_loading
import mobility_generation
import random
import population

from flask import Flask, request, jsonify
import requests
app = Flask(__name__)
imns = None
map_porto = None
gdf_cumulative_p = None

def init():
    global imns
    global map_porto
    
    if imns != None:
        return 0
    # Load IMNs from file
    print("Server initialization:")
    print(" - Loading IMNs...", end="", flush=True)
    imns = imn_loading.read_imn('data/milano_2007_imns.json.gz')
    print(f" loaded {len(imns)} IMNs")

    # Retrieve a map within a bounding-box of 10km around Porto (Portugal) city center
    print(" - Loading Porto map...", end="", flush=True)
    porto_center = (41.1494512, -8.6107884)
    map_porto = mobility_generation.retrieve_osm(porto_center, 10000)
    print(f" loaded, {len(map_porto.nodes())} nodes, {len(map_porto.edges())} edges")
    
    # Build a cumulative probability over a grid map based on population
    # estimated from SEDAC data, contained in "data/gpw_v4_population_density_rev11_2020_30_sec_2020.tif"
    # Souce: https://dljsq618eotzp.cloudfront.net/browseui/index.html#sedac-popdensity-yeargrid5yr-v4.11/
    
    print(" - Checking population map...", end="", flush=True)
    tiff_file = "data/gpw_v4_population_density_rev11_2020_30_sec_2020.tif"
    if not os.path.exists(tiff_file):
        print(" not found, downloading...", end="", flush=True)
        url = "https://data.ghg.center/sedac-popdensity-yeargrid5yr-v4.11/gpw_v4_population_density_rev11_2020_30_sec_2020.tif"
        response = requests.get(url)
        with open(tiff_file, 'wb') as f:
            f.write(response.content)
        print(" downloaded")
    else:
        print(" found")

    print(" - Loading population map...", end="", flush=True)
    gdf_cumulative_p = population.generate_cumulative_map(tiff_file, map_porto)
    print(f" loaded {len(gdf_cumulative_p)} cells")
    
    print("Initialization terminated.")
    return 1


def imn_cloning_to_porto(n_users=10):
    global osm_paths_cache
    osm_paths_cache = {}
    # Apply IMN cloning to generate n_users users (and their trajectories) in Porto

    # Select random IMNs
    selected_uids = list(imns.keys())
    random.shuffle(selected_uids)
    selected_uids = selected_uids[:n_users]

    # Set main parameters
    period_start = None  # Take all data period
    period_end = None  # Take all data period
    user_max_trials = 10

    # Set work location = INESC TEC recharge facilities
    inesctec_coords = (41.17940489334383, -8.595515050610357) # INESC TEC's headquarters
    inesctec_loc, dist = ox.nearest_nodes(map_porto, inesctec_coords[1], inesctec_coords[0], return_dist=True)

    # Output
    outf = []

    print("Process started!")

    for i, uid in enumerate(selected_uids):
        for trial in range(user_max_trials):
            print(f" - Cloning {uid} ({i+1} / {n_users})", end="", flush=True)
            trip_schedule, osm_counts, lmap, rmse = mobility_generation.simulate_trips(
                imns[uid],  # source IMN
                map_porto,  # target OSM road network
                work_osm=inesctec_loc,  # manually set work location
                home_osm=None,  # home location will be random
                n_trials=10, # trials to find best node for each location
                start_sym=period_start, # time window start (w.r.t. source data)
                end_sym=period_end,  # time window end
                gdf_cumulative_p=gdf_cumulative_p,  # weights for location sampling
                use_cache=True,  # use cache for shortest paths 
                use_prefetch=True)  # prefetch shortest paths from high-freq locations
            if trip_schedule != None:
                # Add users' trajectories to output
                outf.extend([ (uid,mylat,mylon,mytime) for mylat, mylon, mytime in trip_schedule ])
                print(f" -- {len(lmap)} locs, RMSE = {rmse:.2f} km")
                break
            else:
                print("   ----- Cloning failed! Retry...")
    return outf




@app.route('/imn_cloning', methods=['POST'])
def imn_cloning_main():
    try:
        data = request.get_json()
        n_users = data['n_users']
        
        if not isinstance(n_users, int) or (n_users < 1):
            return jsonify({'error': 'Please provide valid n_users'}), 400
        
        result = imn_cloning_to_porto(n_users)
        return jsonify({'result': result}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init()
    app.run(debug=False, host='0.0.0.0', port=5001)




