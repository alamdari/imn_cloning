import os
import argparse
from dataclasses import dataclass


@dataclass
class PathsConfig:
    full_imn_path: str = 'data/milano_2007_imns.json.gz'
    test_imn_path: str = 'data/test_milano_imns.json.gz'
    poi_path: str = 'data/test_milano_imns_pois.json.gz'
    results_dir: str = './results'
    prob_subdir: str = 'user_probability_reports4'
    vis_subdir: str = 'user_timeline_visualizations4'

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
    parser.add_argument('--full-imn', default='data/milano_2007_imns.json.gz', help='Path to full IMNs (for user IDs).')
    parser.add_argument('--test-imn', default='data/test_milano_imns.json.gz', help='Path to subset/test IMNs.')
    parser.add_argument('--poi', default='data/test_milano_imns_pois.json.gz', help='Path to POI enrichment data.')
    parser.add_argument('--results-dir', default='./results', help='Base directory for results.')
    parser.add_argument('--prob-subdir', default='user_probability_reports4', help='Probability reports subdirectory.')
    parser.add_argument('--vis-subdir', default='user_timeline_visualizations4', help='Timeline visualizations subdirectory.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility (optional).')
    args = parser.parse_args(argv)

    return PathsConfig(
        full_imn_path=args.full_imn,
        test_imn_path=args.test_imn,
        poi_path=args.poi,
        results_dir=args.results_dir,
        prob_subdir=args.prob_subdir,
        vis_subdir=args.vis_subdir,
    )


