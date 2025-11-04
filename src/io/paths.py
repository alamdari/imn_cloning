import os
import argparse
from dataclasses import dataclass


@dataclass
class PathsConfig:
    imn_path: str = 'data/milano_2007_full_imns.json.gz'
    poi_path: str = 'data/milano_2007_full_imns_pois.json.gz'
    results_dir: str = './results'
    prob_subdir: str = 'user_probability_reports'
    vis_subdir: str = 'user_timeline_visualizations'
    num_users: int = 100  # Number of users to randomly sample for processing

    def prob_dir(self) -> str:
        return os.path.join(self.results_dir, self.prob_subdir)

    def vis_dir(self) -> str:
        return os.path.join(self.results_dir, self.vis_subdir)


def ensure_output_structure(paths: PathsConfig) -> None:
    os.makedirs(paths.results_dir, exist_ok=True)
    os.makedirs(paths.prob_dir(), exist_ok=True)
    os.makedirs(paths.vis_dir(), exist_ok=True)


def parse_args(argv=None) -> PathsConfig:
    parser = argparse.ArgumentParser(description='Generate synthetic timelines from IMNs.')
    parser.add_argument('--imn', default='data/milano_2007_full_imns.json.gz', help='Path to IMN file.')
    parser.add_argument('--poi', default='data/milano_2007_full_imns_pois.json.gz', help='Path to POI enrichment data.')
    parser.add_argument('--results-dir', default='./results', help='Base directory for results.')
    parser.add_argument('--prob-subdir', default='user_probability_reports', help='Probability reports subdirectory.')
    parser.add_argument('--vis-subdir', default='user_timeline_visualizations', help='Timeline visualizations subdirectory.')
    parser.add_argument('--num-users', type=int, default=100, help='Number of users to randomly sample (default: 100).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42).')
    args = parser.parse_args(argv)

    return PathsConfig(
        imn_path=args.imn,
        poi_path=args.poi,
        results_dir=args.results_dir,
        prob_subdir=args.prob_subdir,
        vis_subdir=args.vis_subdir,
        num_users=args.num_users,
    ), args.seed


