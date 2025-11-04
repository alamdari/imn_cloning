from typing import Dict, List, Tuple


class Stay:
    def __init__(self, location_id: int, activity_label: str, start_time: int, end_time: int):
        self.location_id = location_id
        self.activity_label = activity_label
        self.start_time = start_time
        self.end_time = end_time
        self.duration = self.end_time - self.start_time if self.end_time is not None and self.start_time is not None else None

    def set_end_time(self, end_time: int):
        self.end_time = end_time
        if self.start_time is not None:
            self.duration = self.end_time - self.start_time


def read_stays_from_trips(trips: List[Tuple], locations: Dict) -> List[Stay]:
    stays = []
    for from_id, to_id, st, et in trips:
        activity_label = locations[to_id].get('activity_label', 'unknown')
        stays.append(Stay(to_id, activity_label, et, None))
    for i in range(len(stays)):
        current_stay = stays[i]
        if i < len(stays) - 1:
            next_trip_start = trips[i + 1][2]
            current_stay.set_end_time(next_trip_start)
        else:
            if current_stay.start_time is not None:
                current_stay.set_end_time(current_stay.start_time + 3600)
    return stays


def extract_stays_by_day(stays: List[Stay], tz) -> Dict:
    from collections import defaultdict
    from datetime import datetime, timedelta
    stays_by_day = defaultdict(list)
    for stay in stays:
        if stay.start_time is None or stay.end_time is None:
            continue
        start_dt = datetime.fromtimestamp(stay.start_time, tz)
        end_dt = datetime.fromtimestamp(stay.end_time, tz)
        current_dt = start_dt.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = current_dt + timedelta(days=1)
        while current_dt < end_dt:
            day_start = max(start_dt, current_dt)
            day_end = min(end_dt, end_of_day)
            day_stay = Stay(stay.location_id, stay.activity_label, int(day_start.timestamp()), int(day_end.timestamp()))
            stays_by_day[current_dt.date()].append(day_stay)
            current_dt = end_of_day
            end_of_day = current_dt + timedelta(days=1)
    return stays_by_day


