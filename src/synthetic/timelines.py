from typing import Dict, List, Tuple, Any
from collections import defaultdict
import numpy as np
from datetime import datetime, timedelta


def sample_from_hist(hist, bins):
    if hist.sum() == 0:
        return np.mean(bins)
    probs = hist / hist.sum()
    bin_idx = np.random.choice(len(hist), p=probs)
    return np.random.uniform(bins[bin_idx], bins[bin_idx + 1])


def find_anchor_stay_for_day(stays) -> Any:
    non_home_stays = [s for s in stays if s.activity_label != 'home']
    if not non_home_stays:
        return None
    return max(non_home_stays, key=lambda s: s.duration)


def build_stay_distributions(stays_by_day: Dict) -> Tuple[Dict, Dict, Any]:
    duration_dist = defaultdict(list)
    activity_transitions = defaultdict(list)
    trip_durations = []
    for day, day_stays in stays_by_day.items():
        for i in range(len(day_stays) - 1):
            current_stay = day_stays[i]
            next_stay = day_stays[i + 1]
            if current_stay.duration is not None:
                duration_dist[current_stay.activity_label].append(current_stay.duration)
            activity_transitions[current_stay.activity_label].append(next_stay.activity_label)
            if current_stay.end_time is not None and next_stay.start_time is not None:
                trip_duration = next_stay.start_time - current_stay.end_time
                if trip_duration > 0:
                    trip_durations.append(trip_duration)
    duration_probs = {}
    for activity, durations in duration_dist.items():
        if len(durations) > 0:
            hist, bins = np.histogram(durations, bins=20, density=False)
            duration_probs[activity] = (hist, bins)
    transition_probs = {}
    for from_activity, to_activities in activity_transitions.items():
        if len(to_activities) > 0:
            unique_activities, counts = np.unique(to_activities, return_counts=True)
            probs = counts / counts.sum()
            transition_probs[from_activity] = dict(zip(unique_activities, probs))
    trip_duration_probs = None
    if len(trip_durations) > 0:
        hist, bins = np.histogram(trip_durations, bins=20, density=False)
        trip_duration_probs = (hist, bins)
    return duration_probs, transition_probs, trip_duration_probs


def generate_synthetic_day(original_stays, duration_probs: Dict, transition_probs: Dict, randomness: float = 0.5, day_length: int = 24 * 3600, tz=None) -> List[Tuple[str, int, int]]:
    if not original_stays:
        return []
    first_dt = datetime.fromtimestamp(original_stays[0].start_time, tz)
    day_start_dt = first_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    day_start = int(day_start_dt.timestamp())
    rel_stays = []
    for s in original_stays:
        rel_start = int(s.start_time - day_start)
        rel_end = int(s.end_time - day_start)
        rel_stays.append((s, rel_start, rel_end))
    anchor_stay = find_anchor_stay_for_day(original_stays)
    if anchor_stay is None:
        return []
    anchor_rel_start = int(anchor_stay.start_time - day_start)
    anchor_rel_end = int(anchor_stay.end_time - day_start)
    anchor_orig_dur = max(1, anchor_rel_end - anchor_rel_start)
    max_shift = int(0.25 * randomness * anchor_orig_dur)
    if max_shift > 0:
        delta_start = int(np.random.uniform(-max_shift, max_shift))
        delta_dur = int(np.random.uniform(-max_shift, max_shift))
    else:
        delta_start = 0
        delta_dur = 0
    pert_start = max(0, min(anchor_rel_start + delta_start, day_length))
    pert_dur = max(1, anchor_orig_dur + delta_dur)
    if rel_stays and rel_stays[0][0] is anchor_stay:
        pert_start = 0
    pert_end = min(pert_start + pert_dur, day_length)
    if pert_end > day_length:
        pert_end = day_length
        pert_dur = max(1, pert_end - pert_start)
    synthetic_stays = []
    # Start at the first original stay's start time; if it's after midnight, add a home stay from 0 to that time.
    first_rel_start = rel_stays[0][1] if rel_stays else 0
    current_time = first_rel_start
    prev_activity = "home"
    if first_rel_start > 0:
        synthetic_stays.append(("home", 0, first_rel_start))
    for (s, rel_start, rel_end) in rel_stays:
        if s is anchor_stay:
            break
        if current_time == 0:
            # Check if this is a split activity continuation (starts at midnight)
            if rel_start == 0:
                # Split activity continuation - preserve the activity label
                act = s.activity_label
            else:
                # No split activity - enforce "home" at time 0
            act = "home"
        elif np.random.random() < (1 - randomness):
            act = s.activity_label
        else:
            if prev_activity in transition_probs:
                to_probs = transition_probs[prev_activity]
                act = list(to_probs.keys())[np.random.choice(len(to_probs), p=list(to_probs.values()))]
            else:
                act = s.activity_label
        orig_dur = rel_end - rel_start
        if act in duration_probs:
            hist, bins = duration_probs[act]
            sampled_dur = sample_from_hist(hist, bins)
        else:
            sampled_dur = orig_dur
        dur = int((1 - randomness) * orig_dur + randomness * sampled_dur)
        end_time = min(current_time + dur, pert_start)
        if end_time > current_time:
            synthetic_stays.append((act, current_time, end_time))
            prev_activity = act
        current_time = end_time
        if current_time >= pert_start:
            break
    synthetic_stays.append((anchor_stay.activity_label, pert_start, pert_end))
    current_time = pert_end
    prev_activity = anchor_stay.activity_label
    passed_anchor = False
    for (s, rel_start, rel_end) in rel_stays:
        if s is anchor_stay:
            passed_anchor = True
            continue
        if not passed_anchor:
            continue
        if np.random.random() < (1 - randomness):
            act = s.activity_label
        else:
            if prev_activity in transition_probs:
                to_probs = transition_probs[prev_activity]
                act = list(to_probs.keys())[np.random.choice(len(to_probs), p=list(to_probs.values()))]
            else:
                act = s.activity_label
        orig_dur = rel_end - rel_start
        if act in duration_probs:
            hist, bins = duration_probs[act]
            sampled_dur = sample_from_hist(hist, bins)
        else:
            sampled_dur = orig_dur
        dur = int((1 - randomness) * orig_dur + randomness * sampled_dur)
        end_time = min(current_time + dur, day_length)
        if end_time > current_time:
            synthetic_stays.append((act, current_time, end_time))
            prev_activity = act
        current_time = end_time
        if current_time >= day_length:
            break
    
    # Anchor-aware stretching: do not change anchor; extend pre-anchor last stay to
    # anchor start if needed; extend final non-anchor stay to day end.
    def _stretch_anchor_aware(
        stays_rel: List[Tuple[str, int, int]],
        anchor_tuple: Tuple[str, int, int],
        total_len: int
    ) -> List[Tuple[str, int, int]]:
        """Stretch stays to cover full day while preserving anchor timing."""
        if not stays_rel:
            return []
        # Sort by start time
        ordered = sorted(stays_rel, key=lambda x: x[1])

        # Identify anchor index if present exactly
        anchor_idx = None
        for i, t in enumerate(ordered):
            if t == anchor_tuple:
                anchor_idx = i
                break

        # Ensure the first stay starts at 0 if it's not the anchor
        out: List[Tuple[str, int, int]] = []
        act0, s0, e0 = ordered[0]
        if anchor_idx == 0:
            # Keep anchor unchanged; do not force start to 0
            out.append((act0, s0, e0))
        else:
            s0 = 0
            e0 = max(0, min(e0, total_len))
            if e0 < s0:
                e0 = s0
            out.append((act0, s0, e0))

        # Iterate and enforce continuity, preserving anchor start
        for i in range(1, len(ordered)):
            act, st, et = ordered[i]
            st = max(0, min(st, total_len))
            et = max(0, min(et, total_len))
            prev_act, prev_st, prev_et = out[-1]

            if anchor_idx is not None and i == anchor_idx:
                # Do not move anchor start; stretch previous end to anchor start if gap
                st_fixed = st
                prev_et = min(max(st_fixed, 0), total_len)
                out[-1] = (prev_act, prev_st, prev_et)
                if et < st_fixed:
                    et = st_fixed
                out.append((act, st_fixed, et))
            else:
                # Force current start to previous end
                st = prev_et
                if et < st:
                    et = st
                out.append((act, st, et))

        # Stretch last non-anchor stay to day end
        last_idx = len(out) - 1
        if anchor_idx is not None and last_idx == anchor_idx:
            # Do not stretch if anchor is the last; leave as is
            pass
        else:
            last_act, last_st, _ = out[-1]
            out[-1] = (last_act, last_st, total_len)

        return out

    anchor_tuple = (anchor_stay.activity_label, pert_start, pert_end)
    synthetic_stays = _stretch_anchor_aware(synthetic_stays, anchor_tuple, day_length)
    return synthetic_stays


def prepare_day_data(stays_by_day: Dict, user_duration_probs: Dict, user_transition_probs: Dict, randomness_levels: List[float], tz) -> Dict:
    from datetime import datetime
    day_data = {}
    for day_date in sorted(stays_by_day.keys()):
        day = stays_by_day[day_date]
        first_dt = datetime.fromtimestamp(day[0].start_time, tz)
        midnight_dt = first_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        day_start = int(midnight_dt.timestamp())
        # Original stays in relative seconds
        original_stays = [
            (s.activity_label, int(s.start_time - day_start), int(s.end_time - day_start))
            for s in day if (s.start_time is not None and s.end_time is not None)
        ]
        synthetic_dict = {}
        for r in randomness_levels:
            synthetic = generate_synthetic_day(day, user_duration_probs, user_transition_probs, randomness=r, tz=tz)
            synthetic_dict[r] = synthetic
        anchor = find_anchor_stay_for_day(day)
        anchor_tuple = None
        if anchor:
            anchor_tuple = (
                anchor.activity_label,
                anchor.start_time - day_start,
                anchor.end_time - day_start,
            )
        day_data[day_date] = {
            "original": original_stays,
            "synthetic": synthetic_dict,
            "anchor": anchor_tuple
        }
    return day_data

