#!/usr/bin/env python3
"""
Entry point for IMN cloning with temporal + spatial simulation.
"""

import os
import sys
from typing import Dict, List, Tuple, Any
from datetime import datetime

import numpy as np
import pandas as pd
import pytz
import random

from src.io.paths import PathsConfig, parse_args
from src.synthetic.pipeline import run_pipeline


RANDOMNESS_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
TIMEZONE = pytz.timezone("Europe/Rome")


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


def main(argv=None):
    paths, seed = parse_args(argv)
    
    # Setup output logging
    log_file = os.path.join(paths.results_dir, f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    os.makedirs(paths.results_dir, exist_ok=True)
    
    # Redirect output to both console and file
    tee_stdout = TeeOutput(log_file, 'w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = tee_stdout
    sys.stderr = tee_stdout
    
    try:
        print(f"=" * 60)
        print(f"IMN Cloning Pipeline")
        print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"=" * 60)
        print(f"Output log: {log_file}")
        print()
        
        # Set random seed for reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            print(f"Random seed set to: {seed}")
        
        # Run the pipeline
        run_pipeline(paths, RANDOMNESS_LEVELS, TIMEZONE)
        
        print()
        print(f"=" * 60)
        print(f"Pipeline completed successfully!")
        print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"=" * 60)
        print(f"Full log saved to: {log_file}")
        
    except Exception as e:
        print()
        print(f"=" * 60)
        print(f"ERROR: Pipeline failed!")
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
