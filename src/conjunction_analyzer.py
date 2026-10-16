from skyfield.api import load
from datetime import timedelta
import numpy as np
from tle_manager import fetch_active_satellites

def find_conjunctions(
        target_id:int,
        search_window_mins:int = 90,
        fine_threshold_km:float = 10.0,
        broad_threshold_km:float = 500
    ) -> list[dict]:
    satellites = fetch_active_satellites()
    by_id = {sat.model.satnum : sat for sat in satellites}
    target = by_id.get(target_id)
    if not target:
        return []

    ts = load.timescale()
    t_now = ts.now()
    time_steps = [t_now.utc_datetime() + timedelta(minutes=i) for i in range(search_window_mins)]
    times = ts.from_datetimes(time_steps)

    target_trajectory = target.at(times)

    close_approaches = []
    for sat_id, sat in by_id.items():
        if sat_id == target_id:
            continue
        
        diff = sat.at(times) - target_trajectory
        distances = diff.distance().km

        if (np.min(distances) < broad_threshold_km):
            if np.max(distances) < broad_threshold_km:
                continue

            broad_mid_idx = np.argmin(distances)
            broad_tca_time = time_steps[broad_mid_idx]

            fine_start = broad_tca_time - timedelta(seconds=60)
            fine_time_steps = [fine_start + timedelta(seconds=i) for i in range(120)]
            fine_times = ts.from_datetimes(fine_time_steps)
            
            fine_target = target.at(fine_times)
            fine_diff = sat.at(fine_times) - fine_target
            fine_distances = fine_diff.distance().km

            fine_min_idx = np.argmin(fine_distances)
            fine_tca_time = fine_time_steps[fine_min_idx]
            fine_min_dist = fine_distances[fine_min_idx]
            
            if fine_min_dist <= fine_threshold_km:
                close_approaches.append({
                    "satellite_id" : sat_id,
                    "satellite_name" : sat.name,
                    "time" : fine_tca_time,
                    "distance_km" : fine_min_dist
                })

    return close_approaches