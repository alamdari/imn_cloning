#!/usr/bin/env python3
"""
Entry point for IMN cloning with temporal + spatial simulation.
Uses modular src/ package for resources, data, synthetic generation and visualization.
"""

import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import pytz

from src.io.paths import PathsConfig, parse_args
from src.synthetic.pipeline import run_pipeline


RANDOMNESS_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]
TIMEZONE = pytz.timezone("Europe/Rome")



def main(argv=None):
    paths = parse_args(argv)
    # seed optional omitted; user can set via argv and global modules can handle if needed
    run_pipeline(paths, RANDOMNESS_LEVELS, TIMEZONE)


if __name__ == "__main__":
    main()


