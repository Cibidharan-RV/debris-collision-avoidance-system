"""TLE catalog loader and benchmark scenario definitions."""

import os
import requests
from datetime import datetime, timedelta, timezone
from skyfield.api import EarthSatellite, load

ts = load.timescale()

BENCHMARK_SCENARIOS = {
    "starlink_cosmos": {
        "name": "Starlink-1008 vs Cosmos-1408 Debris",
        "description": "Starlink constellation satellite encountering Russian ASAT debris (Critical miss: 0.36 km)",
        "primary": {
            "name": "STARLINK-1008",
            "norad_id": 44714,
            "line1": "1 44714U 19074B   26284.50000000  .00001000  00000-0  10000-3 0  9991",
            "line2": "2 44714  53.0540 120.1234 0001450  45.0000 315.0000 15.06380000300005"
        },
        "secondary": {
            "name": "COSMOS-1408 DEBRIS (#42)",
            "norad_id": 44715,
            "line1": "1 44715U 19074C   26284.50000000  .00001000  00000-0  10000-3 0  9994",
            "line2": "2 44715  53.0580 120.1234 0001450  45.0000 315.0030 15.06380000300004"
        }
    },
    "iss_debris": {
        "name": "ISS (Zarya) vs Fengyun-1C Debris Fragment",
        "description": "International Space Station junction with weather satellite debris (Critical miss: 0.52 km)",
        "primary": {
            "name": "ISS (ZARYA)",
            "norad_id": 25544,
            "line1": "1 25544U 98067A   26284.50000000  .00012500  00000-0  21000-3 0  9998",
            "line2": "2 25544  51.6430 180.2500 0005200 110.0000 250.0000 15.49500000550001"
        },
        "secondary": {
            "name": "FENGYUN-1C DEBRIS (#109)",
            "norad_id": 31115,
            "line1": "1 31115U 99025EJ   26284.50000000  .00004500  00000-0  32000-3 0  9999",
            "line2": "2 31115  51.6480 180.2500 0005200 110.0000 250.0040 15.49500000550001"
        }
    },
    "sentinel_cz4": {
        "name": "Sentinel-1A vs CZ-4 Upper Stage Debris",
        "description": "Earth observation radar satellite conjunction with rocket stage fragment (Critical miss: 0.28 km)",
        "primary": {
            "name": "SENTINEL-1A",
            "norad_id": 39634,
            "line1": "1 39634U 14016A   26284.50000000  .00000150  00000-0  25000-4 0  9993",
            "line2": "2 39634  98.1800 240.1000 0001300  60.0000 300.0000 14.59200000620003"
        },
        "secondary": {
            "name": "CZ-4 STAGE DEBRIS (#8)",
            "norad_id": 41250,
            "line1": "1 41250U 15050B   26284.50000000  .00000800  00000-0  11000-3 0  9995",
            "line2": "2 41250  98.1830 240.1000 0001300  60.0000 300.0020 14.59200000620003"
        }
    }
}

class TLEManager:
    """Handles TLE fetching, caching, and scenario retrieval."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.active_file = os.path.join(data_dir, "active.txt")
        os.makedirs(data_dir, exist_ok=True)
        self.satellites_cache: list[EarthSatellite] = []
        self.by_norad_id: dict[int, EarthSatellite] = {}

    def fetch_active_catalog(self, force_refresh: bool = False) -> list[EarthSatellite]:
        """Loads active satellites from disk or downloads from CelesTrak."""
        if not os.path.exists(self.active_file) or force_refresh:
            url = 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle'
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
            response = requests.get(url, headers=headers, timeout=20)
            with open(self.active_file, 'w', encoding='utf-8') as f:
                f.write(response.text)
                
        loader = load.DataLoader(self.data_dir)
        self.satellites_cache = loader.tle_file('active.txt')
        self.by_norad_id = {sat.model.satnum: sat for sat in self.satellites_cache}
        return self.satellites_cache

    def get_satellite_by_id(self, norad_id: int) -> EarthSatellite | None:
        """Finds satellite by NORAD ID."""
        if not self.by_norad_id:
            self.fetch_active_catalog()
        return self.by_norad_id.get(norad_id)

    def load_scenario(self, scenario_key: str) -> tuple[EarthSatellite, EarthSatellite, dict]:
        """Loads a pre-configured conjunction scenario."""
        if scenario_key not in BENCHMARK_SCENARIOS:
            raise KeyError(f"Unknown scenario '{scenario_key}'. Available: {list(BENCHMARK_SCENARIOS.keys())}")
            
        data = BENCHMARK_SCENARIOS[scenario_key]
        p = data["primary"]
        s = data["secondary"]
        
        sat1 = EarthSatellite(p["line1"], p["line2"], p["name"], ts)
        sat2 = EarthSatellite(s["line1"], s["line2"], s["name"], ts)
        
        return sat1, sat2, data

    def list_benchmarks(self) -> dict:
        """Returns all benchmark scenarios."""
        return BENCHMARK_SCENARIOS

def fetch_active_satellites() -> list[EarthSatellite]:
    mgr = TLEManager()
    return mgr.fetch_active_catalog()
