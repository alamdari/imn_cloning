#!/usr/bin/env python3
"""
Standalone script to evaluate quality of synthetic trajectories.

Features:
    - Statistical analysis of trajectory duration, length, and temporal patterns
    - Spatial analysis with OD matrices and density maps
    - Interactive Folium maps with toggleable layers
    - Comparison with original trajectories (optional)

Usage:
    # Basic evaluation
    python evaluate_quality.py
    
    # With custom directories
    python evaluate_quality.py --trajectories results/synthetic_trajectories --output results/metrics
    
    # With custom grid size
    python evaluate_quality.py --grid-size 1000
    
    # Compare synthetic vs original trajectories (JSON format from imn_generation.py)
    python evaluate_quality.py --original data/trajectories --trajectories results/synthetic_trajectories

Output includes:
    - PNG plots: duration, length, temporal, spatial distributions
    - Interactive HTML maps: origin density heatmap, top OD pairs with flow lines
    - CSV files: trajectory statistics, OD matrix, density data
    - JSON: spatial coverage metrics, quality summary
    - Comparison plots (if original data provided): side-by-side distributions
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
        help='Directory containing trajectory files (CSV or JSON) (default: results/synthetic_trajectories)'
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
    
    parser.add_argument(
        '--original',
        type=str,
        default=None,
        help='Optional: Directory with original trajectory files (JSON from imn_generation.py or CSV)'
    )
    
    args = parser.parse_args()
    
    generate_quality_report(
        trajectories_dir=args.trajectories,
        output_dir=args.output,
        grid_size_m=args.grid_size,
        original_trajectories_dir=args.original
    )


if __name__ == "__main__":
    main()

