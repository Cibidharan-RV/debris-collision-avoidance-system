from skyfield.api import load
from datetime import timedelta
from tle_manager import fetch_active_satellites

def calculate_distance():
    print("Loading satellites......")
    satellites= fetch_active_satellites()
    by_id = {sat.model.satnum : sat for sat in satellites}

    iss = by_id.get(25544)
    tiangong = by_id.get(48274)

    if not iss or not tiangong:
        print("Data not found")
        return
    print("Tracking satellites...")

    ts = load.timescale()
    t = ts.now()

    position_iss = iss.at(t)
    position_tiangong = tiangong.at(t)

    difference = position_iss - position_tiangong
    distance_km = difference.distance().km

    print(f"Distance between ISS and Tiangong: {distance_km:.2f} km")

def find_conjunctions(target_id:int, threshold_km:float, search_window_mins:int = 90) -> list[dict]:
    print("loading conjunctions...")
    satellites = fetch_active_satellites()
    by_id = {sat.model.satnum : sat for sat in satellites}
    target = by_id.get(target_id)
    if not target:
        print(f"Target {target_id} not found.")
        return []

    ts = load.timescale()
    t_now = ts.now()
    time_steps = [t_now.utc_datetime() + timedelta(minutes=i) for i in range(search_window_mins)]
    times = ts.from_datetimes(time_steps)
    print(f"calculating positions over {search_window_mins} minutes for {target.name} ({target_id})")

    target_trajectory = target.at(times)
    found_conjunctions = False

    for sat_id, sat in by_id.items():
        if sat_id == target_id:
            continue

        diff = sat.at(times) - target_trajectory
        distances = diff.distance().km

        close_approaches = []
        in_conjunction = False
        current_event = []
        for i, dist in enumerate(distances):
            if dist < threshold_km:
                in_conjunction = True
                current_event.append((i, dist))
            else:
                if in_conjunction and len(current_event) < 60: # to prevent permanently docked parts, like in ISS ans CSS
                    best_idx, min_dist = min(current_event, key=lambda item: item[1])

                    close_approaches.append({
                        "satellite_id": sat_id,
                        "satellite_name": sat.name,
                        "time": times[best_idx].utc_datetime(),
                        "distance_km": min_dist
                    })
                    
                    in_conjunction = False
                    current_event = []

        if in_conjunction and len(current_event) < 60:
            best_idx, min_dist = min(current_event, key=lambda item: item[1])
            
            close_approaches.append({
                "satellite_id": sat_id,
                "satellite_name": sat.name,
                "time": times[best_idx].utc_datetime(),
                "distance_km": min_dist
            })
            
            in_conjunction = False
            current_event = []
        if (len(close_approaches)):
            found_conjunctions= True
        for approach in close_approaches:
            print(f"Conjunction detected: {approach['satellite_name']} \t at {approach['time'].strftime('%Y-%m-%d %H:%M:%S')} UTC | {approach['distance_km']:.2f} km")
    
    if not found_conjunctions:
        print(f"No conjuctions found for {target.name} ({target_id}) within {search_window_mins} minutes")
    return close_approaches
if __name__ == "__main__":
    find_conjunctions(25544, 10.0, 1440)