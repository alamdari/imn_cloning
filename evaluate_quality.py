#!/usr/bin/env python3
"""
Standalone script to evaluate quality of synthetic trajectories.

Usage:
    python evaluate_quality.py
    python evaluate_quality.py --trajectories results/synthetic_trajectories --output results/metrics
    python evaluate_quality.py --grid-size 1000
"""

import argparse
from src.metrics.quality_report import generate_quality_report


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate quality of synthetic mobility trajectories"
    )
    
    parser.add_argument(
        '--trajectories',
        type=str,
        default='results/synthetic_trajectories',
        help='Directory containing trajectory CSV files (default: results/synthetic_trajectories)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='results/metrics',
        help='Output directory for metrics and plots (default: results/metrics)'
    )
    
    parser.add_argument(
        '--grid-size',
        type=float,
        default=500.0,
        help='Grid cell size in meters for spatial aggregation (default: 500)'
    )
    
    args = parser.parse_args()
    
    generate_quality_report(
        trajectories_dir=args.trajectories,
        output_dir=args.output,
        grid_size_m=args.grid_size
    )


if __name__ == "__main__":
    main()

