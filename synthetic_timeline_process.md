# Synthetic Timeline Generation Process

This document explains the **full process** of generating synthetic mobility timelines from raw IMN (Individual Mobility Network) data. It covers:
1. Reading IMN data
2. Extracting stays
3. Building user-specific probability distributions
4. Generating synthetic days
5. Visualization
6. Current shortcomings and considerations

---

## 1. Reading IMN Data

Each user has an IMN describing their **trips** and **locations**. Example:
```json
{
  "uid": 12498,
  "home": 0,
  "work": 1,
  "locations": {
    "0": {"lat": 45.46, "lon": 9.19},
    "1": {"lat": 45.47, "lon": 9.20},
    "6": {"lat": 45.48, "lon": 9.21}
  },
  "trips": [
    [0, 6, 1175424000, 1175425600],
    [6, 1, 1175430000, 1175432000]
  ]
}
```
- **Trips**: (from_location, to_location, start_time, end_time)
- **Home / Work**: IDs of home and work locations.
- **POIs** (Points of Interest): used to infer activity labels (e.g. `education → school`).

### POI → Activity Mapping
```python
POI_TO_ACTIVITY = {
    "education": "school",
    "food_and_drink": "eat",
    "shopping": "shop",
    "entertainment_and_recreation": "leisure",
    "transportation": "transit",
    "healthcare": "health",
    "public_services": "admin",
    "finance": "finance",
    "utilities": "utility",
    "other": "unknown"
}
```
Overrides: if location matches `home` or `work`, we set it explicitly.

---

## 2. Extracting Stays

We transform **trips** into **stays**:
- Each trip destination becomes a stay.
- The stay duration = (end time of stay) – (start time of stay).
- Last stays get a default duration (e.g. 1h).

**Example:**
Trips:
```
0 → 6 (home → eat), end=09:46:36
6 → 0 (eat → home), end=11:42:16
```

Stays:
```
- home: 00:00–09:46
- eat: 10:04–11:42
- home: 11:54–15:29
- work: 16:11–16:39
- home: 17:25–24:00
```

---

## 3. Building User Probability Distributions

From stays grouped by day, we collect statistics:
- **Duration distributions** per activity
- **Transition probabilities** between activities
- **Trip duration distribution** (gaps between stays)

### Example: Transition Probabilities
```json
{
  "home": {"eat": 0.22, "work": 0.11, "transit": 0.22, "unknown": 0.33},
  "eat": {"home": 0.50, "unknown": 0.50},
  "work": {"home": 1.00},
  "utility": {"home": 0.33, "utility": 0.67},
  "transit": {"home": 1.00}
}
```

### Example: Duration Distribution for "home"
Histogram of stay durations for home (seconds):
```
Bin ranges: [0–2h, 2–4h, …]
Counts:     [3, 5, 8, 4, …]
```

---

## 4. Generating Synthetic Days

We fix the **anchor stay**:
- Definition: the **longest non-home stay** in the day.
- Purpose: ensures daily routine structure (e.g., work or school is preserved).

### Algorithm (per day)

#### Step 1: Before Anchor
For each stay before the anchor:
- **Activity**:
  - With probability *(1 – randomness)* → copy original
  - With probability *(randomness)* → sample from transition_probs using the *previous synthetic activity*
- **Duration**:
  - Blend original duration and sampled duration:
    ```
    dur = (1 – r) * orig_dur + r * sampled_dur
    ```
- Clip to not exceed anchor start.

#### Step 2: Anchor
- Copy anchor exactly (unchanged).

#### Step 3: After Anchor
- Same logic as before anchor, until midnight.
- Clip last stay to end at **24h**.

---

## 4.5. Distribution Blending (Optional Enhancement)

Sometimes user-specific distributions are biased (e.g., very short sleep durations). To mitigate:

- Compute **global distributions** across all users: 
  ```python
  global_duration_probs, global_transition_probs, global_trip_probs = collect_global_statistics(all_users)
  ```

- Blend user and global distributions with weight α:
  ```
  final_dist = (1 - α) * user_dist + α * global_dist
  ```

- Choice of α:
  - α = 0 → use user only.
  - α = 1 → use global only.
  - In practice: increase α when user data is sparse or low-quality.

📌 **Where it applies:**
This blending is done **before calling `generate_synthetic_day`**. Instead of passing raw `user_duration_probs`, etc., we first compute blended distributions and pass those in.

---

## 5. Visualization

- One subplot per day.
- Rows = original + synthetic versions (for `r = 0.0, 0.25, 0.5, 0.75, 1.0`).
- X-axis = 24h timeline.
- Anchor is highlighted (black outline).

**Example (Day: 2007-04-02)**
```
Original:
- home  00:00–09:46
- eat   10:04–11:42
- home  11:54–15:29
- work  16:11–16:39
- home  17:25–24:00

Randomness=0.5:
- home  00:00–06:37
- eat   10:04–11:42
- home  11:42–15:22
- home  15:22–20:51
- home  20:51–24:00
```

---

## 6. Issues and Observations

### ✅ Strengths
- **Rand=0.0** → exact copy of original.
- **Rand=1.0** → fully stochastic using user distributions.
- Anchor ensures daily structure.
- Visualization reveals how randomness modifies timelines.

### ⚠️ Shortcomings
1. **Activity Sampling Bias**
   - Uses *original stays loop* rather than fully generative process.
   - Can produce unrealistic sequences (e.g. eat → eat → eat).

2. **Duration Shrinkage**
   - If the duration distribution is biased toward small values, home/work can shrink unrealistically.
   - More visible at high randomness.

3. **Gaps in Timeline**
   - If sampled durations don’t exactly fill the day, small gaps appear.
   - Currently only “fixed” by clipping last stay to midnight.
   - Solution: preprocess original data to fill gaps explicitly.

4. **No Anchor Perturbation**
   - Anchor is copied exactly → makes high-rand days look too similar.
   - Possible fix: perturb anchor start/end by ±(0.25 * r * duration).

---

## 7. Example Problem: Gaps

Day: **2007-04-05** (original stays)
```
- transit: 00:00–07:14
- unknown: 09:58–12:15
- unknown: 13:09–14:57
- work:    15:55–24:00
```

Synthetic (Rand=1.0) sometimes shows **gaps** (e.g., 07:14–09:58 empty).
Reason:
- Original had a gap (not represented as a stay).
- Synthetic timeline inherits this shift.
- Since sampling doesn’t explicitly fill gaps, activities may be “anticipated.”

Proposed fix: **pre-fill gaps as explicit stays** in preprocessing.

---

## 8. Future Improvements

1. **Pre-fill gaps** → ensure no empty time in original timeline.
2. **Anchor perturbation** → add realism at high randomness.
3. **Blend user and global distributions**:
   - Use global statistics when user data is sparse.
   - Heuristic: switch if user’s histogram has <N samples.
4. **Fully generative mode**:
   - Drop 1:1 mapping with original stays.
   - Generate sequence only from transitions + durations.

---

## 9. Summary

- **Pipeline**: IMN → stays → daily stays → user distributions → synthetic day generation.
- **Anchor**: stabilizes structure.
- **Randomness parameter (r)**:
  - 0 → exact copy
  - 1 → fully stochastic
- **Problems**: gaps, duration shrinkage, activity repetition.
- **Next steps**: gap preprocessing, anchor perturbation, hybrid distributions.

