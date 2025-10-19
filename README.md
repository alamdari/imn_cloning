# IMN Cloning: Synthetic Mobility Trajectory Generation

This project implements **Individual Mobility Network (IMN) cloning**, a two-stage approach to generate synthetic, privacy-preserving mobility trajectories by learning temporal patterns from one city and spatially mapping them to another.

## Overview

The system takes real mobility data (e.g., Milan 2007) and generates synthetic trajectories for a target city (e.g., Porto, Portugal) that preserve realistic temporal patterns while respecting the target city's spatial structure, road network, and activity distribution.

### Key Innovation: Activity-Aware Spatial Mapping

Unlike naive approaches that randomly sample locations from population grids, this implementation uses **activity-aware candidate pools**:
- **Home** activities → nodes near residential areas (`landuse=residential`)
- **Work** activities → nodes near commercial/industrial zones
- **Eat** activities → nodes near restaurants, cafés, bars, fast food
- **School** activities → nodes near educational institutions
- **Health** activities → nodes near hospitals, clinics, pharmacies
- And more...

This ensures synthetic trajectories are spatially realistic (e.g., people go to actual restaurants for eating, schools for education).

---

## How It Works

### 1. **Temporal Stage: Learning Mobility Patterns**
- Loads IMN data from source city (Milan 2007)
- Extracts stay patterns: locations, durations, activity types, transitions
- Builds probabilistic models of user behavior:
  - Activity sequences (home → work → eat → home)
  - Stay durations (how long at each location)
  - Transition probabilities between locations
  - Temporal anchors (recurring visits to same locations)

### 2. **Spatial Stage: Mapping to Target City**
- Downloads target city's OpenStreetMap (OSM) road network
- Queries OSM Points of Interest (POIs) for activity types
- Loads population density data (GeoTIFF) for residential weighting
- Pre-computes activity-specific candidate node pools
- Maps synthetic activities to realistic target locations:
  - Uses OSM shortest paths for routing
  - Generates GPS trajectories with timestamps
  - Respects road network topology

### 3. **Controllable Randomness**
Supports multiple randomness levels (`r ∈ [0, 0.25, 0.5, 0.75, 1.0]`):
- `r=0.0`: Pure cloning (replicate IMN exactly)
- `r=0.5`: Balanced (50% IMN, 50% random sampling)
- `r=1.0`: Fully random (sample from distributions)

This enables privacy-utility tradeoff analysis.

---

## Project Structure

```
imn_cloning/
├── main.py                          # Entry point
├── requirements.txt                 # Python dependencies
├── data/                            # Input data (IMN, population, OSM cache)
│   ├── milano_2007_imns.json.gz     # Source IMN data
│   ├── milano_2007_imns_pois.json.gz # POI annotations
│   ├── gpw_v4_population_density_*.tif # Population density (GeoTIFF)
│   └── porto_*.pkl / *.geojson      # Cached OSM data
│
├── results/                         # Output trajectories and reports
│   ├── user_timeline_visualizations/
│   ├── user_probability_reports/
│   ├── synthetic_trajectories/      # Generated GPS trajectories (CSV + HTML maps)
│   └── metrics/                     # Quality evaluation results
│
├── evaluate_quality.py              # Quality evaluation script
│
└── src/                             # Modular source code
    ├── io/                          # Input/output utilities
    │   ├── paths.py                 # Path management, CLI args
    │   └── data.py                  # Load IMN, POI data
    │
    ├── synthetic/                   # Trajectory generation pipeline
    │   ├── pipeline.py              # Orchestrates full workflow
    │   ├── timelines.py             # Temporal simulation (stays, days)
    │   ├── spatial_sim.py           # Spatial mapping (IMN → OSM)
    │   ├── stays.py                 # Stay distribution modeling
    │   └── enrich.py                # POI enrichment
    │
    ├── spatial/                     # Spatial mapping components
    │   ├── resources.py             # OSM graph, population map loading
    │   ├── mapping.py               # Core spatial algorithms (shortest path, RMSE)
    │   └── activity_pools.py        # Activity-aware node pools
    │
    ├── population/                  # Population-weighted sampling
    │   └── utils.py                 # GeoTIFF processing, cumulative distributions
    │
    ├── metrics/                     # Quality evaluation and statistics
    │   ├── trajectory_stats.py      # Trip duration, length, temporal features
    │   ├── spatial_stats.py         # OD matrix, origin density, coverage
    │   ├── visualizations.py        # Plots (histograms, heatmaps, boxplots)
    │   └── quality_report.py        # Main evaluation orchestrator
    │
    └── visualization/               # Plotting and reporting
        ├── timelines.py             # Stay timeline visualizations
        ├── reports.py               # Probability distribution reports
        └── maps.py                  # Interactive Folium maps
```

---

## Installation

### Dependencies
```bash
pip install -r requirements.txt
```

**Core libraries:**
- `numpy`, `pandas` – Data processing
- `matplotlib`, `seaborn` – Visualization
- `pytz` – Timezone handling
- `osmnx` – OpenStreetMap network retrieval
- `networkx` – Graph algorithms (shortest paths)
- `geopandas` – Geospatial data manipulation
- `rasterio` – GeoTIFF population data
- `folium` – Interactive maps

### Data Requirements

1. **IMN Data** (required):
   - `data/milano_2007_imns.json.gz` – Individual Mobility Networks
   - `data/milano_2007_imns_pois.json.gz` – POI annotations

2. **Population Data** (optional but recommended):
   - Download from [NASA SEDAC GPWv4](https://sedac.ciesin.columbia.edu/data/collection/gpw-v4)
   - Place `gpw_v4_population_density_*.tif` in `data/`

3. **OSM Data** (auto-downloaded and cached):
   - First run downloads Porto's OSM network (~5-10 minutes)
   - Cached as `data/porto_graph.pkl`
   - Activity pools cached as `data/porto_activity_pools.pkl`
   - Portable GeoJSON exports: `porto_amenities.geojson`, etc.

---

## Usage

### Basic Run
```bash
python main.py
```

### CLI Arguments
```bash
python main.py \
  --imn data/milano_2007_imns.json.gz \
  --poi data/milano_2007_imns_pois.json.gz \
  --out results/ \
  --num-users 10
```

**Available options:**
- `--imn`: Path to IMN file (default: `data/milano_2007_imns.json.gz`)
- `--poi`: Path to POI file (default: `data/milano_2007_imns_pois.json.gz`)
- `--out`: Output directory (default: `results/`)
- `--num-users`: Number of users to process (default: `10`)

### Quality Evaluation
After generating synthetic trajectories, evaluate their quality:

```bash
python evaluate_quality.py
```

**With custom paths:**
```bash
python evaluate_quality.py \
  --trajectories results/synthetic_trajectories \
  --output results/metrics \
  --grid-size 500
```

**Options:**
- `--trajectories`: Directory with trajectory CSV files (default: `results/synthetic_trajectories`)
- `--output`: Output directory for metrics (default: `results/metrics`)
- `--grid-size`: Grid cell size in meters for spatial aggregation (default: `500`)

---

## Output

### 1. **Timeline Visualizations**
**Location:** `results/user_timeline_visualizations/`

**Format:** `user_{id}_timeline.png`

Color-coded stay timelines showing:
- Original stays from IMN (top row)
- Synthetic stays at 5 randomness levels (r=0.0 to 1.0)
- Activity types (home, work, eat, school, health, etc.)
- Temporal patterns across multiple days

### 2. **Probability Reports**
**Location:** `results/user_probability_reports/`

**Format:** `user_probs_report_{id}.json` + `.png`

Statistical analysis:
- Stay duration distributions
- Activity type frequencies
- Location visit counts
- Transition probabilities

### 3. **Synthetic Trajectories**
**Location:** `results/synthetic_trajectories/`

**Format:** `user_{id}_porto_trajectory.csv` + `user_{id}_porto_map.html`

GPS trajectories with columns:
- `trajectory_id` – Unique ID per day's trajectory
- `lat` – Latitude
- `lon` – Longitude  
- `time` – Unix timestamp

Interactive HTML maps showing:
- Multi-day trajectories overlaid on Porto street network
- Per-day layers (toggleable)
- Stay points and routes

### 4. **Quality Evaluation Metrics**
**Location:** `results/metrics/`

**Generated by:** `python evaluate_quality.py`

Comprehensive quality assessment of synthetic trajectories:

**Data files:**
- `trajectory_statistics.csv` – Per-trajectory stats (duration, length, OD, temporal features)
- `od_matrix.csv` – Origin-destination trip counts by grid cells
- `origin_density.csv` – Spatial density of trip origins
- `destination_density.csv` – Spatial density of trip destinations
- `spatial_coverage.json` – Bounding box, area covered
- `quality_summary.json` – Aggregate statistics across all users

**Visualizations (PNG):**
- `trip_duration_distribution.png` – Histogram of trip durations
- `trip_length_distribution.png` – Histogram of OD distances
- `trip_length_boxplot.png` – Box-whisker plot across users
- `temporal_distribution.png` – Trips by hour of day
- `day_of_week_distribution.png` – Trips by day of week
- `origin_density_heatmap.png` – 2D heatmap of origin locations
- `top_od_pairs.png` – Top 20 origin-destination flows
- `path_vs_od_distance.png` – Circuity analysis (path length vs straight-line distance)

---

## Key Features

✅ **Activity-Aware Mapping** – Uses OSM POIs to select realistic locations  
✅ **Population Weighting** – Samples residential areas by population density  
✅ **Efficient Caching** – OSM data cached as pickle + GeoJSON for portability  
✅ **Controllable Privacy** – 5 randomness levels for privacy-utility tradeoff  
✅ **Modular Design** – Clean separation: I/O, synthetic, spatial, visualization  
✅ **Reproducible** – Configurable seeds, deterministic pipeline  
✅ **Scalable** – Processes users in parallel (future: multiprocessing)  

---

## Performance Optimizations

### Spatial Data Caching
All expensive OSM queries are cached in `data/` directory:

**Pickle files (fast loading):**
- **OSM Graph:** `porto_graph.pkl` (~2-5 MB, 1-time download)
- **Activity Pools:** `porto_activity_pools.pkl` (~1-10 MB, computed once)
- **Amenities:** `porto_amenities.pkl` (~500 KB - 5 MB depending on query)
- **Residential Areas:** `porto_residential.pkl` (~1-3 MB)
- **Work Areas:** `porto_work.pkl` (~500 KB - 2 MB)

**GeoJSON files (portable, cross-platform):**
- **Amenities:** `porto_amenities.geojson` (~1-10 MB)
- **Residential Areas:** `porto_residential.geojson` (~2-5 MB)
- **Work Areas:** `porto_work.geojson` (~1-3 MB)

**Transfer to other computers:**
1. Copy `data/*.pkl` and `data/*.geojson` files
2. Place in target machine's `data/` directory
3. Skip re-downloading and re-processing (~10-30 minutes saved)

### Restricted OSM Queries

**Current Configuration (Fast Mode):**  
For rapid prototyping and testing, only **11 essential amenity types** are queried:
- `restaurant`, `cafe`, `bar` → eat activities
- `school`, `university` → education activities
- `hospital`, `clinic`, `doctors` → health activities
- `supermarket` → shopping activities
- `office`, `bank` → work/finance activities

**Query times:**
- All 27K+ amenities: ~5-10 minutes
- Full `AMENITY_CLASSES` set (111 types): ~2-5 minutes, 17K+ features
- **Minimal set (11 types): ~30-60 seconds, 3K-5K features** ✅ *(current)*

**To use the full amenity set:**  
Edit `src/spatial/activity_pools.py` (lines 129-136) and uncomment the `AMENITY_CLASSES` code block:
```python
# Uncomment this block for comprehensive amenity coverage:
needed_amenity_types = []
for amenity_class, amenity_list in AMENITY_CLASSES.items():
    needed_amenity_types.extend(amenity_list)
needed_amenity_types = list(set(needed_amenity_types))
tags = {'amenity': needed_amenity_types}
```

Then delete cached files to force re-query:
```bash
rm data/porto_amenities.pkl data/porto_amenities.geojson
```

---

## Algorithm Details

### Stay Distribution Modeling
For each IMN location, computes:
- **Duration distribution:** Mean, std, min, max stay times
- **Visit count distribution:** How often location is visited
- **Temporal patterns:** Time of day, day of week effects

### Spatial Mapping (IMN → OSM)
1. **Anchor Identification:** Detect recurring locations (home, work)
2. **Node Assignment:**
   - Home/work: Fixed across days (stability)
   - Other activities: Sampled from activity-specific pools
3. **Routing:** OSM shortest path with Haversine distance
4. **Trajectory Generation:** GPS points + timestamps along route

### Quality Metrics
- **RMSE (Root Mean Square Error):** Measures spatial distortion
- **Temporal alignment:** Preserves stay sequence and durations
- **Activity fidelity:** Maintains activity type distributions

---

## References

- **OpenStreetMap (OSM):** [https://www.openstreetmap.org/](https://www.openstreetmap.org/)
- **OSMnx Library:** Boeing, G. (2017). OSMnx: New methods for acquiring, constructing, analyzing, and visualizing complex street networks. *Computers, Environment and Urban Systems*.
- **NASA SEDAC GPWv4:** Gridded Population of the World, Version 4
- **Individual Mobility Networks (IMN):** Pappalardo et al. (2015). Returners and explorers dichotomy in human mobility. *Nature Communications*.

---

## Future Work

- [ ] Multi-city support (beyond Porto)
- [ ] Real-time streaming mode
- [ ] Privacy metric evaluation (k-anonymity, differential privacy)
- [ ] Comparison with baseline methods (EPR, Markov models)
- [ ] Interactive web dashboard for exploration
- [ ] Multiprocessing for large-scale user generation

---

## License

This project is for research purposes. Please cite appropriately if used in publications.

---

## Contact

For questions or collaboration, please open an issue in the repository.

