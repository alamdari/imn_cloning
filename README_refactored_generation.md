## Synthetic Timeline Generation (Refactored)

This document describes the refactored pipeline implemented in `refactored_generation_process.py` for generating synthetic daily timelines from Individual Mobility Networks (IMNs).

### Overview

Given IMNs (per-user locations and trips with enriched POI semantics), the script:

1. Loads IMNs and POI data
2. Enriches IMN locations with activity labels
3. Extracts per-day stays from trips
4. Builds user-specific distributions (stay durations per activity, activity transitions, trip gap durations)
5. Generates synthetic day timelines across multiple randomness levels
6. Produces per-user probability reports and daily timeline visualizations

Outputs are identical to the pre-refactor behavior by default.

### Key Concepts

- Activity labels are derived from POI categories, with explicit overrides for `home` and `work`.
- Daily timelines operate in relative seconds (0–86400) within a day.
- An anchor stay is the longest non-home stay in a day; synthetic generation preserves its role but perturbs its start and duration slightly based on the randomness level.
- Gaps are not modeled as activities; they are fixed by stretching only synthetic timelines post-generation while respecting the anchor.

### CLI Usage

Run the script:

```bash
python refactored_generation_process.py \
  --full-imn data/milano_2007_imns.json.gz \
  --test-imn data/test_milano_imns.json.gz \
  --poi data/test_milano_imns_pois.json.gz \
  --results-dir ./results \
  --prob-subdir user_probability_reports3 \
  --vis-subdir user_timeline_visualizations3 \
  --seed 42
```

All parameters are optional. Defaults match current repository paths and output directories. Use `--seed` to make sampling reproducible.

### Pipeline Steps

1. Data Loading (`load_datasets`)
   - Reads the full IMNs to get the list of users
   - Reads the test IMNs (subset) and filters to users present in both
   - Reads the POI dataset

2. IMN Enrichment (`enrich_imn_with_poi`)
   - Assigns activity labels to each location based on POI frequencies, with `home` and `work` overrides

3. Stays Extraction (`extract_stays_from_trips` and `extract_stays_by_day`)
   - Converts trips to destination stays
   - Splits cross-day stays into per-day stays in absolute timestamps

4. Probability Building (`build_stay_distributions`)
   - Per-activity duration histograms
   - Transition probabilities between activities
   - Distribution of inter-stay gaps (trip durations)

5. Synthetic Day Generation (`generate_synthetic_day`)
   - Converts absolute times to day-relative seconds
   - Identifies an anchor (longest non-home stay)
   - Perturbs anchor start and duration within ±0.25 × randomness × original_duration; if the anchor is the first activity, its start is clamped to 0
   - Samples activities and durations for pre/post-anchor segments using the learned distributions
   - Applies anchor-aware stretching: extends only the immediately preceding stay to the anchor start if needed, and extends only the last non-anchor stay to midnight

6. Visualization & Reports (`user_probs_report`, `visualize_day_data`)
   - Saves per-user probability report PNG and JSON to the probability reports directory
   - Saves per-day original and synthetic timeline plots to the visualization directory

### Customization

- Use CLI parameters to point to different datasets or output directories.
- Control randomness globally by editing `RANDOMNESS_LEVELS` or by setting seeds.

### Outputs

- Probability reports: `<results-dir>/<prob-subdir>/user_probs_report_<userId>.png|.json`
- Timeline plots: `<results-dir>/<vis-subdir>/user_<userId>_timelines.png`


