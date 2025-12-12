# IMN Cloning: Individual Mobility Network Generation

A computational framework for generating synthetic individual mobility trajectories by cloning Individual Mobility Networks (IMNs) from a source city to a target city. The system preserves both temporal patterns and spatial structure through a two-stage pipeline that combines temporal timeline generation with activity-aware spatial mapping.

## Overview

This project implements a pipeline for synthesizing realistic mobility trajectories by transferring Individual Mobility Networks from a source city (e.g., Milan) to a target city (e.g., Porto). The approach consists of two parallel processing branches:

1. **Temporal Timeline Generation**: Extracts stay patterns from original IMN trips, models temporal distributions (stay durations, activity transitions, trip durations), and generates synthetic daily timelines with configurable randomness levels.

2. **Spatial Mapping**: Maps IMN locations to OpenStreetMap (OSM) nodes in the target city using activity-aware constraints, preserving both semantic meaning (activity types) and spatial structure (relative distances).

These branches converge to reconstruct the IMN in the target city, followed by network-based trajectory routing to generate final GPS trajectories.

## Architecture

The pipeline follows a structured workflow:

```
IMN Extraction
    ├──→ Temporal Timeline Generation
    │       ├── Enrichment (POI-based activity labeling)
    │       ├── Stay Extraction
    │       ├── Temporal Modeling (distributions, transitions)
    │       └── Timeline Generation (synthetic stays)
    │
    └──→ Spatial Mapping
            └── Activity-aware location mapping with distance constraints
                    │
                    ↓
            IMN Reconstruction at Target
                    │
                    ↓
            Trip Routing (network shortest paths)
                    │
                    ↓
            Synthetic Trajectories
```

## Pipeline Components

### 1. Data Loading and Enrichment

**Input Data:**
- **IMN Files**: Individual Mobility Networks in JSON format containing locations, trips, and home/work anchors
- **POI Data**: Pre-computed Point of Interest enrichment data with activity labels and frequencies per location
- **OSM Network**: OpenStreetMap road network graph for the target city (downloaded automatically via OSMnx)

**Enrichment Process:**
- Enriches IMN locations with activity labels derived from POI data
- Maps POI frequency vectors to activity categories (transit, health, admin, finance, eat, utility, school, leisure, shop, other)
- Preserves home and work designations from original IMN

### 2. Temporal Timeline Generation

**Stay Extraction:**
- Extracts stays from trip sequences in the original IMN
- Groups stays by calendar day with timezone-aware processing
- Identifies anchor stays (longest non-home stays) per day

**Temporal Modeling:**
- Builds activity-specific stay duration distributions (histograms)
- Computes activity transition probabilities (Markov chain)
- Models trip duration distributions between consecutive stays
- Generates probability reports for each user

**Timeline Generation:**
- Generates synthetic daily timelines with configurable randomness levels (0.0 to 1.0)
- Preserves anchor stay timing with controlled perturbation
- Samples stay durations from learned distributions
- Samples activity transitions based on transition probabilities
- Supports multiple randomness levels for ablation studies

### 3. Spatial Mapping

**Activity-Aware Node Pools:**
- Pre-computes candidate OSM node pools for each activity type
- Queries OSM amenities, shops, leisure facilities, and other POI categories
- Identifies nodes within proximity (default 200m) of relevant features
- Special handling for home (residential landuse + 50% augmentation) and work (commercial/office/industrial + 50% augmentation)
- Caches pools for efficient reuse

**Spatial Mapping Algorithm:**
- Maps IMN locations to OSM nodes while preserving pairwise distances
- Uses activity-aware candidate selection from pre-computed pools
- Applies distance constraints (0.5x to 1.5x original distance) to prevent unrealistic mappings
- Implements directional penalty (2x) for overshooting target distances to counteract geometric bias
- Maps home and work as anchor points first, then incrementally maps remaining locations
- Computes Root Mean Square Error (RMSE) of distance preservation

**Key Features:**
- Unique seed per user ensures deterministic but diverse mappings
- Top-k selection (top 20% or minimum 20 pairs) prevents convergence to identical home-work pairs
- Random perturbation breaks ties for identical distance errors

### 4. Trajectory Generation

**Node Assignment:**
- Assigns OSM nodes to synthetic stays using precomputed spatial mapping
- Prioritizes fixed home/work nodes for home/work activities
- Uses activity pools for new activities not present in original IMN
- Ensures unique sampling per user via user_id-based seeding

**Network Routing:**
- Generates network shortest paths between consecutive stay locations
- Uses OSM road network with travel time weights
- Caches path computations for efficiency
- Handles cases where no path exists (returns None)

**Output Format:**
- GPS trajectories with latitude, longitude, and timestamp
- Separate trajectory per trip leg (between consecutive stays)
- Combined daily trajectories with trajectory IDs

## External Resources

### OpenStreetMap (OSM)

The pipeline uses OSM data via the OSMnx library:

- **Road Networks**: Downloaded automatically for target cities (Porto, Milan)
- **POI Features**: Queried via OSMnx `features_from_bbox` for amenities, shops, leisure facilities, etc.
- **Graph Processing**: NetworkX-based graph operations for shortest path routing
- **Caching**: OSM graphs are cached as GraphML files to avoid repeated downloads

OSM graphs are automatically downloaded and cached for supported target cities.

### Population Data

- **Source**: GPW v4 Population Density (NASA SEDAC)
- **Format**: GeoTIFF raster (30 arc-second resolution)
- **Usage**: Population-weighted node sampling fallback when activity pools are unavailable
- **Processing**: Converted to cumulative probability distributions for efficient sampling

## Installation

### Prerequisites

**System Dependencies:**
- GDAL (for geopandas/rasterio)
  - macOS: `brew install gdal`
  - Ubuntu/Debian: `sudo apt-get install gdal-bin libgdal-dev`
- SpatialIndex (for rtree)
  - macOS: `brew install spatialindex`
  - Ubuntu/Debian: `sudo apt-get install libspatialindex-dev`

**Python Dependencies:**
```bash
pip install -r requirements.txt
```

Key dependencies include:
- `osmnx>=2.0.6`: OpenStreetMap data access and graph processing
- `geopandas>=1.1.1`: Spatial data manipulation
- `networkx>=3.5`: Graph algorithms
- `pandas>=2.3.1`, `numpy>=2.3.2`: Data manipulation
- `rasterio>=1.4.3`: Population raster processing
- `folium>=0.20.0`: Interactive map visualization

## Usage

### Command-Line Interface

The main entry point is `main.py`:

```bash
python main.py [OPTIONS]
```

#### Required Arguments

None (uses default paths if not specified)

#### Optional Arguments

**Data Input:**
- `--imn PATH`: Path to IMN JSON file (default: `data/milano_2007_full_imns.json.gz`)
- `--poi PATH`: Path to POI enrichment file (default: `data/milano_2007_full_imns_pois.json.gz`)
- `--data-dir PATH`: Directory for spatial resources (default: `data`)

**Output Configuration:**
- `--results-dir PATH`: Base directory for results (default: `./results`)

**Processing Options:**
- `--num-users N`: Number of users to randomly sample (default: all users)
- `--target-city CITY`: Target city for spatial simulation (`porto`, `milan`) (default: `porto`)
- `--spatial-randomness FLOAT`: Randomness level for spatial simulation, 0.0-1.0 (default: 0.5)
- `--use-random-mapping`: Use random spatial mapping instead of activity-aware mapping (for ablation studies)
- `--seed INT`: Random seed for reproducibility (default: 42)

#### Example Commands

**Basic usage (default settings):**
```bash
python main.py
```

**Process 100 users with Porto as target:**
```bash
python main.py --num-users 100 --target-city porto
```

**High spatial randomness (0.8) with custom output directory:**
```bash
python main.py --spatial-randomness 0.8 --results-dir results/experiment_1
```

**Ablation study with random mapping:**
```bash
python main.py --use-random-mapping --results-dir results/ablation_random
```

**Custom data paths:**
```bash
python main.py --imn data/custom_imns.json.gz --poi data/custom_pois.json.gz
```

### IMN Generation

IMN files are generated from raw trajectory data using `imn_generation.py`:

```bash
python imn_generation.py input_trajectories.csv output_imns.json.gz \
    --temporal_thr 20 \
    --spatial_thr 50 \
    --save_trajectories data/trajectories
```

- `input_trajectories.csv`: Input trajectory file (CSV format)
- `output_imns.json.gz`: Output IMN file
- `--temporal_thr`: Temporal threshold in minutes for stay detection (default: 20)
- `--spatial_thr`: Spatial threshold in meters for location clustering (default: 50)
- `--save_trajectories`: Optional directory to save processed trajectories

### POI Enrichment

After generating IMN files, POI data must be added using the `add_poi_to_imns.py` script:

```bash
python add_poi_to_imns.py input_imns.json.gz output_pois.json.gz [--buffer 200]
```

**Parameters:**
- `input_imns.json.gz`: Input IMN file
- `output_pois.json.gz`: Output POI-enriched file
- `--buffer`: POI proximity buffer in meters (default: 200)

**Process:**
- Queries OSM for all amenities, shops, leisure, public transport, railway, highway, and office features
- Builds R-tree spatial index for efficient nearest-neighbor queries
- Assigns activity labels based on proximity to POI features
- Generates POI frequency vectors per location
- Outputs enriched IMN data with POI information

## Output Structure

Results are organized in the specified results directory:

```
results/
├── run_log_YYYYMMDD_HHMMSS.txt          # Complete pipeline log
└── synthetic_trajectories/               # Generated trajectories
    ├── user_0_porto_trajectory.csv      # GPS trajectories (trajectory_id, lat, lon, time, day_date)
    ├── user_1_porto_trajectory.csv
    └── ...
```

### Trajectory CSV Format

Each trajectory CSV contains:
- `trajectory_id`: Unique identifier for each trip leg
- `lat`: Latitude (WGS84)
- `lon`: Longitude (WGS84)
- `time`: Unix timestamp
- `day_date`: Date string (YYYY-MM-DD)

## Algorithm Details

### Spatial Mapping Constraints

The spatial mapping algorithm implements two key constraints to prevent systematic distance inflation:

1. **Spatial Constraint**: Only considers candidate pairs where the OSM distance is within 0.5x to 1.5x the original IMN distance.

2. **Directional Penalty**: Applies a 2x penalty to error when overshooting the target distance (`d_osm > d_imn`), counteracting the geometric bias where more ways exist to overshoot than undershoot.

These constraints ensure that mapped distances remain consistent with the original network and eliminate systematic inflation in synthetic trajectories.

### Temporal Randomness

The temporal timeline generation supports multiple randomness levels (0.0 to 1.0):

- **0.0**: Deterministic (preserves original timeline exactly)
- **0.5**: Balanced (default, mixes original and sampled patterns)
- **1.0**: Fully random (samples entirely from learned distributions)

Randomness affects:
- Activity selection (original vs. sampled from transition probabilities)
- Stay duration (interpolation between original and sampled durations)
- Anchor stay timing (controlled perturbation)

### Activity Pools Augmentation

To mitigate sparse OSM tag data, home and work activity pools are augmented with 50% of all OSM nodes in the target city. This ensures sufficient candidate diversity while maintaining semantic relevance through landuse-based filtering.

## License

All Rights Reserved. This software is provided for research and academic purposes only. Commercial use, distribution, or modification without explicit permission is prohibited. The authors retain all rights to this work until publication.

## Contact

- **Omid Isfahani Alamdari**: omid.isfahan@unfc.ca
- **Mirco Nanni**: mirco.nanni@isti.cnr.it

