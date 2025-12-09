#!/usr/bin/env python3
"""
Add POI (Point of Interest) data to IMN files.

Usage:
    python3 add_poi_to_imns.py input_imns.json.gz output_pois.json.gz
    python3 add_poi_to_imns.py input_imns.json.gz output_pois.json.gz --buffer 200
"""
import numpy as np
import osmnx as ox
import networkx as nx
from shapely.geometry import box
from rtree import index
import gzip
import json
import argparse
import sys
import pandas as pd
from typing import Dict, List
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Add POI data to IMN files")
    parser.add_argument("input_filename", type=str, help="Path to input IMN json.gz file")
    parser.add_argument("output_filename", type=str, help="Path to output POI json.gz file")
    parser.add_argument("--buffer", type=int, default=200, help="POI buffer distance in meters (default: 200)")
    
    args = parser.parse_args()
    
    input_filename = args.input_filename
    output_filename = args.output_filename
    buffer_poi_meters = args.buffer

    # Open and read the gzipped JSON file
    data = []
    print(f"Loading IMNs from {input_filename}...")
    with gzip.open(input_filename, 'rt', encoding='utf-8') as f:
        lcount = 0
        for line in f:
            data.append(json.loads(line))
            lcount += 1
            if lcount % 10 == 0:
                print(f"  Reading IMN n.{lcount}")
    print(f"✓ Loaded {len(data)} IMNs\n")

    # Calculate bounding box
    print("Calculating bounding box...")
    loc_bbox = [999, 999, -999, -999]
    for mydata in data:
        xlist = [xy[0] for _, xy in mydata['location_prototype'].items()]
        ylist = [xy[1] for _, xy in mydata['location_prototype'].items()]
        loc_bbox[0] = min(loc_bbox[0], min(xlist))
        loc_bbox[1] = min(loc_bbox[1], min(ylist))
        loc_bbox[2] = max(loc_bbox[2], max(xlist))
        loc_bbox[3] = max(loc_bbox[3], max(ylist))
    loc_bbox = [loc_bbox[0]-0.001, loc_bbox[1]-0.001, loc_bbox[2]+0.001, loc_bbox[3]+0.001]
    print(f"  BBox of data: {loc_bbox}\n")

    # Compute meters_to_degree conversion factor
    xa, ya, xb, yb = loc_bbox
    meters_per_cent_degree = []
    for pt in [(xa,ya), (xa,yb), (xb,ya), (xb,yb)]:
        for delta in [(-0.001, 0), (0.001, 0), (0, -0.001), (0, 0.001)]:
            pt2 = (pt[0]+delta[0], pt[1]+delta[1])
            meters_per_cent_degree.append(ox.distance.great_circle(pt[1], pt[0], pt2[1], pt2[0]))
    meters_per_cent_degree = np.min(meters_per_cent_degree)
    meters_to_degree = 1/(meters_per_cent_degree/0.001)

    # Retrieve all POI features from OSMnx
    # We query ALL amenities/POIs (not just ACTIVITY_TAGS) so we can classify unmatched ones as "other"
    print("Retrieving POI features from OSMnx (all amenities + ACTIVITY_TAGS tags)...")
    
    # First, query ALL amenities (to capture everything, including those not in ACTIVITY_TAGS)
    print("  Querying all amenities...")
    tags_all_amenities = {'amenity': True}
    gdf_all_amenities = ox.features.features_from_bbox(bbox=loc_bbox, tags=tags_all_amenities)
    print(f"  ✓ Retrieved {len(gdf_all_amenities)} amenities")
    
    # Also query other tag types that might have POIs (shop, leisure, etc.)
    print("  Querying other POI tag types...")
    tags_other = {
        'shop': True,
        'leisure': True,
        'public_transport': True,
        'railway': True,
        'highway': ['bus_stop', 'bus_station'],
        'office': True,
        'man_made': True,
        'power': True,
        'aeroway': True,
    }
    gdf_other = ox.features.features_from_bbox(bbox=loc_bbox, tags=tags_other)
    print(f"  ✓ Retrieved {len(gdf_other)} other POI features")
    
    # Combine all features
    import geopandas as gpd
    if len(gdf_all_amenities) > 0 and len(gdf_other) > 0:
        gdf_amenities = pd.concat([gdf_all_amenities, gdf_other], ignore_index=True)
        # Remove duplicates (some features might have multiple tags)
        gdf_amenities = gdf_amenities.drop_duplicates(subset=['geometry'], keep='first')
    elif len(gdf_all_amenities) > 0:
        gdf_amenities = gdf_all_amenities
    elif len(gdf_other) > 0:
        gdf_amenities = gdf_other
    else:
        gdf_amenities = gpd.GeoDataFrame()
    
    if not isinstance(gdf_amenities, gpd.GeoDataFrame):
        gdf_amenities = gpd.GeoDataFrame(gdf_amenities)
    
    print(f"  ✓ Total unique POI features: {len(gdf_amenities)}\n")
    
    # Import ACTIVITY_TAGS structure (excluding home/work)
    # Same structure as src/spatial/activity_pools.py
    ACTIVITY_TAGS = {
        'transit': [
            ('amenity', [
                'parking', 'taxi', 'fuel', 'parking_entrance', 'parking_exit', 'parking_access',
                'car_rental', 'bicycle_rental', 'bus_station', 'bicycle_parking', 'motorcycle_parking',
                'ferry_terminal', 'kick-scooter_parking', 'motorcycle_rental', 'taxi_rank'
            ]),
            ('public_transport', ['station', 'stop_position', 'platform']),
            ('railway', ['station', 'halt', 'stop', 'tram_stop']),
            ('highway', ['bus_stop', 'bus_station']),
            ('aeroway', ['aerodrome', 'terminal']),
        ],
        'health': [
            ('amenity', [
                'pharmacy', 'clinic', 'hospital', 'dentist', 'doctors', 'veterinary', 'nursing_home',
                'health_post'
            ]),
        ],
        'admin': [
            ('amenity', [
                'post_office', 'police', 'fire_station', 'townhall', 'courthouse', 'waste_disposal',
                'social_facility', 'shelter', 'community_centre', 'public_bookcase', 'events_venue',
                'prison', 'archive', 'mortuary', 'public_building', 'crematorium', 'payment_centre',
                'meeting_room', 'reception_desk', 'group_home', 'dormitory'
            ]),
            ('office', ['government', 'administrative']),
        ],
        'finance': [
            ('amenity', ['bank', 'atm', 'bureau_de_change', 'money_transfer']),
        ],
        'eat': [
            ('amenity', [
                'restaurant', 'cafe', 'fast_food', 'pub', 'bar', 'ice_cream', 'biergarten', 'food_court',
                'canteen'
            ]),
        ],
        'utility': [
            ('amenity', [
                'telephone', 'recycling', 'drinking_water', 'toilets', 'vending_machine',
                'charging_station', 'compressed_air', 'sanitary_dump_station', 'water_point',
                'bicycle_repair_station', 'vacuum_cleaner', 'parcel_locker'
            ]),
            ('man_made', ['water_tower', 'wastewater_plant', 'works']),
            ('power', ['substation', 'plant']),
        ],
        'school': [
            ('amenity', [
                'library', 'school', 'kindergarten', 'college', 'university', 'music_school',
                'driving_school', 'childcare', 'research_institute', 'language_school'
            ]),
        ],
        'leisure': [
            ('amenity', [
                'cinema', 'theatre', 'nightclub', 'studio', 'events_venue', 'gambling',
                'public_bath', 'watering_place', 'auditorium'
            ]),
            ('leisure', ['park', 'playground', 'pitch', 'sports_centre', 'stadium', 'swimming_pool']),
        ],
        'shop': [
            ('amenity', ['marketplace', 'mall']),
            ('shop', [
                'supermarket', 'convenience', 'mall', 'department_store', 'bakery', 'beverages',
                'alcohol', 'butcher', 'greengrocer', 'kiosk', 'clothes', 'shoes', 'electronics',
                'hardware', 'doityourself', 'furniture', 'books', 'sports', 'outdoor', 'variety_store',
                'stationery', 'beauty', 'chemist', 'jewelry', 'gift'
            ]),
        ],
    }
    

    # Build tag_value_to_activity mapping from ACTIVITY_TAGS (same as activity_pools.py)
    tag_value_to_activity: Dict[str, Dict[str, str]] = defaultdict(dict)
    for activity, kv_list in ACTIVITY_TAGS.items():
        for key, values in kv_list:
            for v in values:
                tag_value_to_activity[key][v] = activity
    
    # Use activity labels directly (same as activity_pools.py) - no conversion needed
    activity_labels_no_other = sorted(set(ACTIVITY_TAGS.keys()))
    activity_to_class_id = { activity: id+1 for id, activity in enumerate(activity_labels_no_other) }
    activity_to_class_id['other'] = 0
    class_id_to_activity = { id: activity for activity, id in activity_to_class_id.items() }
    # Build poi_classes list - ensure all IDs from 0 to max are present
    max_class_id = max(class_id_to_activity.keys()) if class_id_to_activity else 0
    poi_classes = [class_id_to_activity.get(i, 'other') for i in range(max_class_id + 1)]

    # Load POI features into R-tree index
    print("Loading POI features in R-tree index...")
    idx = index.Index()
    index_n = 0
    classified_count = defaultdict(int)
    
    for _pos, feature in gdf_amenities.iterrows():
        # Map feature to activity using same logic as activity_pools.py
        activity_label = 'other'  # Default to 'other' if no match
        
        # Check each tag key in tag_value_to_activity
        for key in tag_value_to_activity.keys():
            if key in feature and feature.get(key):
                val = feature.get(key)
                # Handle lists / multiple values
                if isinstance(val, (list, tuple, set)):
                    for v in val:
                        if v in tag_value_to_activity[key]:
                            activity_label = tag_value_to_activity[key][v]
                            break
                else:
                    if val in tag_value_to_activity[key]:
                        activity_label = tag_value_to_activity[key][val]
                if activity_label != 'other':
                    break
        
        classified_count[activity_label] += 1
        class_id = activity_to_class_id.get(activity_label, 0)
        idx.insert(index_n, feature['geometry'].bounds, obj=class_id)
        index_n += 1
    
    print(f"  ✓ Indexed {index_n} POI features")
    print(f"  Classification breakdown:")
    for activity in sorted(classified_count.keys()):
        count = classified_count[activity]
        pct = 100 * count / index_n if index_n > 0 else 0
        print(f"    {activity}: {count} ({pct:.1f}%)")
    print()

    # Compute POI frequencies
    buffer_poi = buffer_poi_meters*meters_to_degree
    print(f"Computing POI frequencies (buffer={buffer_poi_meters}m)...")
    # poi_classes already defined above
    poi_dict_list = []
    for mydata in data:
        poi_dict = { 
            'uid': mydata['uid'], 
            'poi_classes': poi_classes,
            'poi_freq': { } 
        }
        for loc_id, xy in mydata['location_prototype'].items():
            x, y = xy
            x0, y0, x1, y1 = x-buffer_poi, y-buffer_poi, x+buffer_poi, y+buffer_poi 
            poi_vector = np.zeros(max_class_id + 1)
            for item in list(idx.intersection((x0, y0, x1, y1), objects=True)):
                class_id = item.object
                if class_id < 0 or class_id > max_class_id:
                    continue  # Skip invalid class IDs
                bbox = item.bbox
                if ox.distance.great_circle(y, x, bbox[1], bbox[0]) <= buffer_poi_meters:
                    poi_vector[class_id] += 1
            poi_dict['poi_freq'][loc_id] = poi_vector.tolist()
        poi_dict_list.append(poi_dict)
        if len(poi_dict_list) % 10 == 0:
            print(f"  ({len(poi_dict_list)}/{len(data)}) uid: {mydata['uid']}, {len(poi_dict['poi_freq'])} locations processed")

    # Save POI frequencies
    print(f"\nSaving POI frequencies to {output_filename}...")
    with gzip.GzipFile(output_filename, 'w') as fout:
        for poi_dict in poi_dict_list:
            json_str = '%s\n' % json.dumps(poi_dict)
            json_bytes = json_str.encode('utf-8')
            fout.write(json_bytes)
    print(f"✓ Saved {len(poi_dict_list)} POI records")
    print("\nDone!")

if __name__ == '__main__':
    main()
