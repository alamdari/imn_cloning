# Quality Evaluation Improvements

This document summarizes all the improvements made to the quality evaluation system in response to the identified issues.

## Issues Fixed

### 1. ✅ Day of Week Distribution Only Showed Thursday

**Problem**: The trajectory CSV files didn't have a `trajectory_id` column, so all trips were treated as one continuous trajectory, resulting in incorrect day-of-week analysis.

**Solution**:
- Modified `refactored_generation_process_with_spatial.py` (lines 1605-1626) to generate CSV files with proper `trajectory_id` column
- Each day's movements are now treated as a separate trajectory
- Added `day_date` column to track which date each trajectory belongs to

**Files Modified**:
- `refactored_generation_process_with_spatial.py`

**Result**: The day of week distribution now correctly shows counts for all days of the week present in the data.

---

### 2. ✅ Origin Density Heatmap Now Interactive with Folium

**Problem**: The origin density heatmap was a static matplotlib PNG image with unclear grid cells - completely useless.

**Solution**:
- Created new function `plot_origin_density_interactive_map()` in `src/metrics/visualizations.py`
- **REMOVED** the static PNG version entirely (`plot_origin_density_heatmap()`)
- Uses Folium with HeatMap plugin for interactive visualization
- Features:
  - **Interactive heatmap layer** showing origin density with color gradient (blue → cyan → lime → yellow → red)
  - **Toggleable markers layer** (hidden by default) showing individual origin points with trip counts
  - Layer control in top-right corner allows toggling markers on/off
  - Zoom, pan, and click interactions
  - Popup tooltips showing trip counts at each location

**Files Modified**:
- `src/metrics/visualizations.py` (added `plot_origin_density_interactive_map()`)
- `src/metrics/quality_report.py` (removed PNG call, added interactive map)

**Output**: `origin_density_map.html` (no more PNG!)

---

### 3. ✅ Top OD Pairs Now Visualized on Interactive Map

**Problem**: The `top_od_pairs.png` showed grid cell coordinates like "(123,456)→(789,012)" which were unreadable and not meaningful.

**Solution**:
- Created new function `plot_od_pairs_interactive_map()` in `src/metrics/visualizations.py`
- Modified `compute_od_matrix()` in `src/metrics/spatial_stats.py` to store representative lat/lon coordinates for each OD pair
- Features:
  - **Flow lines** (red) connecting origins to destinations, with width proportional to trip count
  - **Origin markers** (green) sized by number of trips
  - **Destination markers** (blue) sized by number of trips
  - Interactive popups showing exact coordinates and trip counts
  - Tooltips on hover
  - Shows top 20 OD pairs by default

**Files Modified**:
- `src/metrics/visualizations.py` (added `plot_od_pairs_interactive_map()`)
- `src/metrics/spatial_stats.py` (modified `compute_od_matrix()` to include coordinates)
- `src/metrics/quality_report.py` (removed static PNG, added interactive map)

**Output**: `top_od_pairs_map.html` (no more PNG!)

---

### 4. ✅ ALL Distributions Now Compare Original vs Synthetic

**Problem**: No comparison between original city trajectories and synthetic Porto trajectories - distributions only showed synthetic data in isolation.

**Solution**:
- Created new function `plot_distribution_comparison()` in `src/metrics/visualizations.py`
- Added optional `--original` parameter to `evaluate_quality.py`
- Modified `generate_quality_report()` to **prioritize comparison plots** when original data is available
- **ALL distribution plots are now side-by-side comparisons** (when `--original` is provided)
- Features:
  - **Side-by-side comparisons** for:
    1. Trip Duration Distribution (original left, synthetic right)
    2. Trip Length Distribution (original left, synthetic right)
    3. Temporal Distribution (original left, synthetic right)
  - Each plot shows mean/median lines for quick comparison
  - Original city in blue, target city (Porto) in green
  - Saved directly to output directory (no subdirectory)
  - If no original data: falls back to synthetic-only plots

**Files Modified**:
- `src/metrics/visualizations.py` (added `plot_distribution_comparison()`)
- `src/metrics/quality_report.py` (restructured to prioritize comparisons)
- `evaluate_quality.py` (added `--original` argument)

**Outputs** (when original data provided):
- `duration_comparison.png` (replaces synthetic-only version)
- `length_comparison.png` (replaces synthetic-only version)
- `temporal_comparison.png` (replaces synthetic-only version)
- `original_trajectory_statistics.csv`

---

## How to Use

### Basic Quality Evaluation (Synthetic Only)
```bash
python evaluate_quality.py --trajectories results/synthetic_trajectories --output results/metrics
```

### With Original Data Comparison
```bash
python evaluate_quality.py \
    --trajectories results/synthetic_trajectories \
    --original data/original_trajectories \
    --output results/metrics
```

### Custom Grid Size
```bash
python evaluate_quality.py --grid-size 1000
```

---

## Complete Output Structure

After running the evaluation, the output directory contains:

### With Original Data (--original provided):
```
results/metrics/
├── trajectory_statistics.csv           # Synthetic trajectory stats
├── original_trajectory_statistics.csv  # Original trajectory stats
├── od_matrix.csv                       # Origin-destination pairs
├── origin_density.csv                  # Spatial density data
├── destination_density.csv             # Destination density data
├── spatial_coverage.json               # Spatial coverage metrics
├── quality_summary.json                # Overall quality summary
├── duration_comparison.png             # ✅ Original vs Synthetic
├── length_comparison.png               # ✅ Original vs Synthetic
├── temporal_comparison.png             # ✅ Original vs Synthetic
├── trip_length_boxplot.png             # Synthetic only
├── day_of_week_distribution.png        # ✅ Now shows all days correctly
├── path_vs_od_distance.png             # Synthetic only
├── origin_density_map.html             # ✅ Interactive heatmap (NO PNG!)
└── top_od_pairs_map.html               # ✅ Interactive OD flows (NO PNG!)
```

### Without Original Data (synthetic only):
```
results/metrics/
├── trajectory_statistics.csv           # Synthetic trajectory stats
├── od_matrix.csv                       # Origin-destination pairs
├── origin_density.csv                  # Spatial density data
├── destination_density.csv             # Destination density data
├── spatial_coverage.json               # Spatial coverage metrics
├── quality_summary.json                # Overall quality summary
├── trip_duration_distribution.png      # Synthetic only
├── trip_length_distribution.png        # Synthetic only
├── temporal_distribution.png           # Synthetic only
├── trip_length_boxplot.png             # Synthetic only
├── day_of_week_distribution.png        # Synthetic only
├── path_vs_od_distance.png             # Synthetic only
├── origin_density_map.html             # Interactive heatmap
└── top_od_pairs_map.html               # Interactive OD flows
```

---

## Technical Details

### Trajectory CSV Format (Updated)
```csv
trajectory_id,lat,lon,time,day_date
0,41.1234,-8.6234,1234567890,2007-11-15
0,41.1235,-8.6235,1234567920,2007-11-15
1,41.1236,-8.6236,1234657890,2007-11-16
1,41.1237,-8.6237,1234657920,2007-11-16
```

Each trajectory_id represents one day's worth of movements.

### Interactive Map Features
- **Folium-based**: Fully interactive, can be opened in any browser
- **Layer control**: Toggle different visualization layers
- **Responsive**: Works on desktop and mobile
- **Export ready**: Can be shared as standalone HTML files

### Comparison Analysis
- Requires original trajectories in same CSV format
- Automatically computes statistics and generates side-by-side plots
- Useful for validating that synthetic trajectories maintain similar characteristics to original data

---

## Dependencies

The improvements require the following Python packages (already in requirements.txt):
- `folium` - Interactive maps
- `matplotlib` - Static plots
- `seaborn` - Statistical visualizations
- `pandas` - Data processing
- `numpy` - Numerical operations

---

## Future Enhancements (Optional)

Possible future improvements:
1. Add circuity comparison (path length / straight-line distance)
2. Include speed/velocity distributions
3. Add user-level quality metrics
4. Implement statistical tests (KS-test, etc.) for distribution comparisons
5. Add trajectory animation on interactive maps
6. Include POI visit frequency analysis

---

## Summary

All issues have been successfully resolved:

1. ✅ **Day of week distribution** - Fixed by adding trajectory_id to CSV files (now shows all days correctly)
2. ✅ **Origin density visualization** - Interactive Folium map with toggleable layers (static PNG removed - it was useless!)
3. ✅ **OD pairs visualization** - Interactive map with flow lines and markers (static PNG removed)
4. ✅ **ALL distributions are comparisons** - Side-by-side plots comparing original vs synthetic when original data is provided

**Key Improvements**:
- **No more useless PNGs**: Spatial visualizations are now interactive HTML maps only
- **Comparisons by default**: When you provide original data, ALL distributions show side-by-side comparisons
- **Better insights**: Interactive maps allow zooming, clicking, and toggling layers
- **Cleaner output**: Removed redundant static images, kept only what's useful

The quality evaluation system now provides comprehensive, interactive visualizations and meaningful comparisons between original and synthetic mobility data.

