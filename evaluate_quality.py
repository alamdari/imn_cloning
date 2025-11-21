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

import os
import sys
import argparse
from datetime import datetime
from src.metrics.quality_report import generate_quality_report


class TeeOutput:
    """Redirect stdout/stderr to both console and file."""
    def __init__(self, file_path, mode='w'):
        self.terminal = sys.stdout if 'stdout' in str(type(self)) else sys.stderr
        self.log = open(file_path, mode)
    
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    
    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def close(self):
        self.log.close()


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
    
    parser.add_argument(
        '--source-city',
        type=str,
        default='milan',
        choices=['milan', 'porto'],
        help='Source city name for original trajectory timezone (default: milan)'
    )
    
    parser.add_argument(
        '--target-city',
        type=str,
        default='porto',
        choices=['milan', 'porto'],
        help='Target city name for synthetic trajectory timezone (default: porto)'
    )
    
    parser.add_argument(
        '--compute-path-metrics',
        action='store_true',
        default=False,
        help='Compute path metrics (shortest path, detour ratio, etc.). This is computationally expensive. (default: False)'
    )
    
    args = parser.parse_args()
    
    # Setup output logging
    log_file = os.path.join(args.output, f"evaluation_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    os.makedirs(args.output, exist_ok=True)
    
    # Redirect output to both console and file
    tee_stdout = TeeOutput(log_file, 'w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = tee_stdout
    sys.stderr = tee_stdout
    
    try:
        print(f"=" * 60)
        print(f"Trajectory Quality Evaluation")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"=" * 60)
        print(f"Output log: {log_file}")
        print(f"Trajectories: {args.trajectories}")
        print(f"Output: {args.output}")
        print(f"Grid size: {args.grid_size}m")
        print(f"Source city: {args.source_city} (for original trajectory timezone)")
        print(f"Target city: {args.target_city} (for synthetic trajectory timezone)")
        print(f"Compute path metrics: {args.compute_path_metrics}")
        if args.original:
            print(f"Original trajectories: {args.original}")
        print()
        
        # Run the evaluation
        generate_quality_report(
            trajectories_dir=args.trajectories,
            output_dir=args.output,
            grid_size_m=args.grid_size,
            original_trajectories_dir=args.original,
            source_city=args.source_city,
            target_city=args.target_city,
            compute_path_metrics=args.compute_path_metrics
        )
        
        print()
        print(f"=" * 60)
        print(f"Evaluation completed successfully!")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"=" * 60)
        print(f"Full log saved to: {log_file}")
        
    except Exception as e:
        print()
        print(f"=" * 60)
        print(f"ERROR: Evaluation failed!")
        print(f"=" * 60)
        import traceback
        traceback.print_exc()
        print()
        print(f"Full log (including error) saved to: {log_file}")
        raise
    
    finally:
        # Restore original stdout/stderr
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        tee_stdout.close()


if __name__ == "__main__":
    main()

