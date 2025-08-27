# Organized Generation Process Script

This script is an organized version of the notebook-based generation process for Individual Mobility Networks (IMNs). It processes all users and generates synthetic timelines with organized output structure.

## Features

- **Batch Processing**: Processes all users automatically instead of just one
- **Organized Output**: Creates a clean folder structure for results
- **Error Handling**: Continues processing even if individual users fail
- **Progress Tracking**: Shows progress through all users
- **High-Quality Outputs**: Saves high-resolution visualizations and detailed reports

## Output Structure

The script creates the following folder structure:

```
./results/
├── user_probability_reports/          # User probability analysis reports
│   ├── user_probs_report_650.png     # Visual report (PNG)
│   ├── user_probs_report_650.json    # Detailed data (JSON)
│   └── ...                           # One set per user
└── user_timeline_visualizations/      # Timeline visualizations
    ├── user_650_timelines.png        # User timeline comparison
    └── ...                           # One visualization per user
```

## Usage

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Script

```bash
python organized_generation_process.py
```

### 3. Monitor Progress

The script will show:
- Data loading progress
- User processing progress with counts
- Success/failure messages for each user
- Final completion status

## What the Script Does

For each user, it:

1. **Loads and Enriches Data**: Combines IMN data with POI information
2. **Extracts Stays**: Converts trip data into location stays
3. **Builds Distributions**: Learns user-specific patterns for:
   - Stay durations per activity type
   - Activity transition probabilities
   - Trip duration distributions
4. **Generates Reports**: Creates probability analysis visualizations
5. **Creates Timelines**: Generates synthetic timelines with different randomness levels
6. **Saves Outputs**: Stores all results in organized folders

## Configuration

You can modify these constants at the top of the script:

- `RANDOMNESS_LEVELS`: Controls how much randomness to apply (default: [0.0, 0.25, 0.5, 0.75, 1.0])
- `TIMEZONE`: Timezone for date handling (default: Europe/Rome)
- `ACTIVITY_COLORS`: Color scheme for different activity types

## Data Requirements

The script expects these files in the `data/` directory:

- `milano_2007_imns.json.gz` - Full IMN dataset
- `test_milano_imns.json.gz` - Test subset of IMNs
- `test_milano_imns_pois.json.gz` - POI data for test users

## Error Handling

- **Missing POI Data**: Users without POI data are skipped
- **Processing Errors**: Individual user failures don't stop the entire process
- **Data Validation**: Checks for valid stays before processing

## Output Files

### Probability Reports (PNG + JSON)
- **Duration Distributions**: Box plots showing stay duration patterns
- **Transition Matrix**: Heatmap of activity transition probabilities  
- **Trip Durations**: Distribution of travel times between locations

### Timeline Visualizations (PNG)
- **Original Timeline**: User's actual daily schedule
- **Synthetic Timelines**: Generated schedules with varying randomness
- **Anchor Highlighting**: Longest non-home stay is highlighted
- **Multi-day View**: All days shown in one comprehensive figure

## Performance Notes

- Processing time depends on number of users and data complexity
- Large datasets may take several minutes to complete
- Memory usage scales with user count and timeline length
- High-resolution outputs (300 DPI) ensure publication quality

## Troubleshooting

**Common Issues:**
- Missing data files: Check `data/` directory exists
- Import errors: Install required packages with `pip install -r requirements.txt`
- Memory issues: Process smaller user subsets if needed
- Plot errors: Ensure matplotlib backend is properly configured

**Debug Mode:**
To debug individual users, you can modify the script to process only specific user IDs by filtering the `imns` dictionary.
