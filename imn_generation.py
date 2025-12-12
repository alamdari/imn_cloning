import pandas as pd
import numpy as np
import json
import gzip
import datetime
import networkx as nx
from networkx.readwrite import json_graph
import argparse
import sys
import os
sys.path.append('libs')

from trajectory import Trajectory
import trajectory_segmenter as trajectory_segmenter
from individual_mobility_network import build_imn

def key2str(k):
    if isinstance(k, tuple):
        return str(k)
    elif isinstance(k, datetime.time):
        return str(k)
    elif isinstance(k, np.int64):
        return str(k)
    elif isinstance(k, np.float64):
        return str(k)
    return k


def clear_tuples4json(o):
    if isinstance(o, dict):
        return {key2str(k): clear_tuples4json(o[k]) for k in o}
    return o


def agenda_converter(o):
    if isinstance(o, datetime.datetime):
        return o.__str__()
    elif isinstance(o, datetime.timedelta):
        return o.__str__()
    elif isinstance(o, Trajectory):
        return o.to_json()
    elif isinstance(o, nx.DiGraph):
        # return json_graph.node_link_data(o, {'link': 'edges', 'source': 'from', 'target': 'to'})
        return json_graph.node_link_data(o, 
                                         source="from",
                                         target="to",
                                         edges="edges")
    else:
        return o.__str__()

def main_from_code(points_df, output_filename, temporal_thr=1200, spatial_thr=50):
    # min number of trajectories after segmentation for a user to be considered as
    # a user with enough mobility
    min_trajs_no = 10 

    expected_cols = ['id','longitude','latitude','timestamp']
    for c in expected_cols:
        if c not in points_df.columns:
            print(f"ERROR: input dataframe does not contain {c} (should have {expected_cols})")
    
    for uid, user_data in points_df.groupby('id'):
        points = user_data.sort_values(by=['timestamp'])[
            ['longitude','latitude','timestamp']].values.tolist()
        trajs_dict = trajectory_segmenter.segment_trajectories(
            points, int(uid), temporal_thr=temporal_thr, spatial_thr=spatial_thr)

        imh = {'uid': int(uid), 'trajectories': trajs_dict}

        if len(trajs_dict) <= min_trajs_no:
            continue

        imn = build_imn(imh)
        if imn is not None:
            imn["uid"] = uid
            json_str = '%s\n' % json.dumps(clear_tuples4json(imn), default=agenda_converter)
            json_bytes = json_str.encode('utf-8')
            with gzip.GzipFile(output_filename, 'a') as fout:
                fout.write(json_bytes)

            print(f"IMN created for {uid} with {len(trajs_dict)} trajectories")
            
def main():
    parser = argparse.ArgumentParser(description="Generate Individual Mobility Network (IMN)")
    parser.add_argument("input_filename", type=str, help="Path to the input CSV file")
    parser.add_argument("output_filename", type=str, help="Path to the output json.gz file")
    
    # Segmentation parameters
    parser.add_argument("--temporal_thr", type=int, default=20, help="Temporal threshold in minutes")
    parser.add_argument("--spatial_thr", type=int, default=50, help="Spatial threshold in meters")
    
    # Optional trajectory saving
    parser.add_argument("--save_trajectories", type=str, default=None, 
                       help="Optional: Directory to save segmented trajectories (one file per user)")
    
    args = parser.parse_args()

    input_filename = args.input_filename
    output_filename = args.output_filename
    save_trajectories = args.save_trajectories
    temporal_thr = args.temporal_thr * 60
    spatial_thr = args.spatial_thr
    
    # min number of trajectories after segmentation for a user to be considered as
    # a user with enough mobility
    min_trajs_no = 2 

    points_df = pd.read_csv(input_filename)
    print(points_df.head())
    print(f"\nProcessing {len(points_df.groupby('id'))} users...")
    if save_trajectories:
        os.makedirs(save_trajectories, exist_ok=True)
        print(f"  → Will save segmented trajectories to: {save_trajectories}/")
    print()

    user_count = 0
    for uid, user_data in points_df.groupby('id'):
        user_count += 1
        points = user_data.sort_values(by=['timestamp'])[
            ['longitude','latitude','timestamp']].values.tolist()
        trajs_dict = trajectory_segmenter.segment_trajectories(
            points, int(uid), temporal_thr=temporal_thr, spatial_thr=spatial_thr)

        # Save trajectories if requested (one file per user)
        if save_trajectories and len(trajs_dict) > min_trajs_no:
            traj_record = {
                'uid': int(uid),
                'num_trajectories': len(trajs_dict),
                'trajectories': trajs_dict
            }
            traj_filename = os.path.join(save_trajectories, f"user_{uid}_trajectories.json")
            with open(traj_filename, 'w') as fout:
                json.dump(clear_tuples4json(traj_record), fout, default=agenda_converter)

        imh = {'uid': int(uid), 'trajectories': trajs_dict}

        if len(trajs_dict) <= min_trajs_no:
            if user_count % 10 == 0:
                print(f"[{user_count}] User {uid}: {len(trajs_dict)} trajectories (skipped - too few)")
            continue

        imn = build_imn(imh)
        if imn is not None:
            imn["uid"] = uid
            json_str = '%s\n' % json.dumps(clear_tuples4json(imn), default=agenda_converter)
            json_bytes = json_str.encode('utf-8')
            with gzip.GzipFile(output_filename, 'a') as fout:
                fout.write(json_bytes)

            print(f"[{user_count}] User {uid}: {len(trajs_dict)} trajectories → IMN created")
        else:
            print(f"[{user_count}] User {uid}: {len(trajs_dict)} trajectories → IMN failed")

if __name__ == '__main__':
    main()
