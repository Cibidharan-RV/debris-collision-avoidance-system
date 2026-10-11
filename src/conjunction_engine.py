"""Close-approach screening and TCA estimation using SGP4 and Brent minimization."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import numpy as np
from scipy.optimize import minimize_scalar, brentq
from sgp4.api import jday
from skyfield.api import EarthSatellite, load

from src.constants import MU_EARTH, SAFETY_DISTANCE_KM, BROAD_THRESHOLD_KM
from src.orbital_frames import compute_semi_major_axis, compute_relative_state

ts = load.timescale()

def datetime_to_sgp4_jd(dt: datetime) -> tuple[float, float]:
    """Converts a datetime to SGP4 whole JD and day fraction."""
    sec = dt.second + (dt.microsecond / 1e6)
    return jday(dt.year, dt.month, dt.day, dt.hour, dt.minute, sec)

@dataclass
class ConjunctionEvent:
    """Summary of a conjunction event at TCA."""
    primary_name: str
    primary_id: int
    secondary_name: str
    secondary_id: int
    
    tca_epoch: datetime
    tca_seconds_from_t0: float
    
    miss_distance_km: float
    relative_velocity_km_s: float
    range_rate_km_s: float
    
    # State vectors in TEME at TCA
    r_primary_tca: np.ndarray
    v_primary_tca: np.ndarray
    r_secondary_tca: np.ndarray
    v_secondary_tca: np.ndarray
    
    pre_burn_sma_km: float
    collision_probability_pc: float
    is_risk: bool
    risk_threshold_km: float = SAFETY_DISTANCE_KM

    def summary(self) -> str:
        status = "CRITICAL COLLISION RISK" if self.is_risk else "SAFE PASS"
        return (
            f"[{status}] {self.primary_name} x {self.secondary_name}\n"
            f"  TCA (UTC):        {self.tca_epoch.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC\n"
            f"  Miss Distance:    {self.miss_distance_km:.4f} km (Threshold: {self.risk_threshold_km:.1f} km)\n"
            f"  Relative Speed:   {self.relative_velocity_km_s:.3f} km/s ({self.relative_velocity_km_s * 3600:.1f} km/h)\n"
            f"  Collision Prob:   {self.collision_probability_pc:.2e}\n"
            f"  Pre-Burn SMA:     {self.pre_burn_sma_km:.3f} km"
        )

def compute_2d_collision_probability(
    miss_distance_km: float,
    combined_sigma_km: float = 0.150,
    hard_body_radius_km: float = 0.010
) -> float:
    """Estimates 2D collision probability (Pc) on the encounter B-plane."""
    if miss_distance_km <= 0:
        return 1.0
    u = (miss_distance_km ** 2) / (2.0 * (combined_sigma_km ** 2))
    factor = (hard_body_radius_km ** 2) / (2.0 * (combined_sigma_km ** 2))
    pc = 1.0 - np.exp(-factor)
    pc_scaled = float(pc * np.exp(-u))
    return min(1.0, max(1e-12, pc_scaled))

class ConjunctionEngine:
    """Conjunction screening engine using coarse grid search followed by Brent optimization."""

    def __init__(
        self,
        coarse_step_sec: float = 20.0,
        safety_distance_km: float = SAFETY_DISTANCE_KM
    ):
        self.coarse_step_sec = coarse_step_sec
        self.safety_distance_km = safety_distance_km

    def evaluate_distance_at_time(
        self,
        sat1: EarthSatellite,
        sat2: EarthSatellite,
        epoch_input
    ) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculates distance and range-rate at a given epoch."""
        if isinstance(epoch_input, datetime):
            jd, fraction = datetime_to_sgp4_jd(epoch_input)
        elif hasattr(epoch_input, 'utc_datetime'):
            jd, fraction = datetime_to_sgp4_jd(epoch_input.utc_datetime())
        else:
            jd = epoch_input.whole
            fraction = epoch_input.tai_fraction - epoch_input._leap_seconds() / 86400.0

        _, r1, v1 = sat1.model.sgp4(jd, fraction)
        _, r2, v2 = sat2.model.sgp4(jd, fraction)
        
        r1 = np.array(r1, dtype=float)
        v1 = np.array(v1, dtype=float)
        r2 = np.array(r2, dtype=float)
        v2 = np.array(v2, dtype=float)
        
        _, _, dist_km, range_rate = compute_relative_state(r1, v1, r2, v2)
        return dist_km, range_rate, r1, v1, r2, v2

    def analyze_pair(
        self,
        sat1: EarthSatellite,
        sat2: EarthSatellite,
        start_time: datetime | None = None,
        duration_minutes: float = 90.0
    ) -> ConjunctionEvent:
        """Finds closest approach point over a given time window."""
        if start_time is None:
            start_time = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
        elif start_time.tzinfo is None:
            start_time = start_time.replace(tzinfo=timezone.utc)

        total_seconds = duration_minutes * 60.0
        time_offsets = np.arange(0.0, total_seconds, self.coarse_step_sec)
        
        # Coarse grid pass
        distances_coarse = []
        for sec in time_offsets:
            dt = start_time + timedelta(seconds=float(sec))
            jd, fraction = datetime_to_sgp4_jd(dt)
            _, r1, _ = sat1.model.sgp4(jd, fraction)
            _, r2, _ = sat2.model.sgp4(jd, fraction)
            diff = np.array(r2) - np.array(r1)
            distances_coarse.append(float(np.linalg.norm(diff)))
        distances_coarse = np.array(distances_coarse)
        
        min_idx = int(np.argmin(distances_coarse))
        t_coarse_min_sec = float(time_offsets[min_idx])
        
        # Fine search around coarse minimum
        bracket_min = max(0.0, t_coarse_min_sec - 2.0 * self.coarse_step_sec)
        bracket_max = min(total_seconds, t_coarse_min_sec + 2.0 * self.coarse_step_sec)
        
        def scalar_dist_obj(t_sec: float) -> float:
            dt = start_time + timedelta(seconds=float(t_sec))
            jd, fraction = datetime_to_sgp4_jd(dt)
            _, r1, _ = sat1.model.sgp4(jd, fraction)
            _, r2, _ = sat2.model.sgp4(jd, fraction)
            diff = np.array(r2) - np.array(r1)
            return float(np.linalg.norm(diff))
            
        opt_res = minimize_scalar(
            scalar_dist_obj,
            bounds=(bracket_min, bracket_max),
            method='bounded',
            options={'xatol': 1e-4}
        )
        
        refined_tca_sec = float(opt_res.x)
        tca_epoch = start_time + timedelta(seconds=refined_tca_sec)
        
        # Evaluate states at refined TCA
        d_miss, range_rate, r1_tca, v1_tca, r2_tca, v2_tca = self.evaluate_distance_at_time(
            sat1, sat2, tca_epoch
        )
        
        v_rel = float(np.linalg.norm(v2_tca - v1_tca))
        sma_pre = compute_semi_major_axis(r1_tca, v1_tca, MU_EARTH)
        pc = compute_2d_collision_probability(d_miss)
        is_risk = d_miss < self.safety_distance_km
        
        return ConjunctionEvent(
            primary_name=sat1.name,
            primary_id=int(sat1.model.satnum),
            secondary_name=sat2.name,
            secondary_id=int(sat2.model.satnum),
            tca_epoch=tca_epoch,
            tca_seconds_from_t0=refined_tca_sec,
            miss_distance_km=d_miss,
            relative_velocity_km_s=v_rel,
            range_rate_km_s=range_rate,
            r_primary_tca=r1_tca,
            v_primary_tca=v1_tca,
            r_secondary_tca=r2_tca,
            v_secondary_tca=v2_tca,
            pre_burn_sma_km=sma_pre,
            collision_probability_pc=pc,
            is_risk=is_risk,
            risk_threshold_km=self.safety_distance_km
        )

if __name__ == "__main__":
    from src.tle_manager import TLEManager
    mgr = TLEManager()
    sat1, sat2, meta = mgr.load_scenario("starlink_cosmos")
    engine = ConjunctionEngine()
    print("Running conjunction screening...")
    event = engine.analyze_pair(sat1, sat2)
    print(event.summary())
