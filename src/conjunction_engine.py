"""Close-approach screening and TCA estimation using SGP4 and Brent minimization."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import numpy as np
from scipy.optimize import minimize_scalar
from sgp4.api import jday, SGP4_ERRORS
from skyfield.api import EarthSatellite, load

from src.constants import MU_EARTH, SAFETY_DISTANCE_KM, BROAD_THRESHOLD_KM
from src.orbital_frames import compute_semi_major_axis, compute_relative_state

ts = load.timescale()

class SGP4PropagationError(RuntimeError):
    """Raised when SGP4 propagation fails with a nonzero error code or invalid states."""
    def __init__(self, sat_name: str, sat_id: int | str, error_code: int, epoch: datetime):
        msg_detail = SGP4_ERRORS.get(error_code, f"Unknown error code {error_code}")
        super().__init__(
            f"SGP4 propagation failed for {sat_name} (ID: {sat_id}) at {epoch.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC: "
            f"Error code {error_code} ({msg_detail})"
        )
        self.sat_name = sat_name
        self.sat_id = sat_id
        self.error_code = error_code
        self.epoch = epoch


def propagate_sgp4_safe(sat: EarthSatellite, jd: float, fraction: float, epoch: datetime) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates satellite state with strict SGP4 error code and finite numerical checks."""
    err, r, v = sat.model.sgp4(jd, fraction)
    sat_name = getattr(sat, 'name', 'UNKNOWN')
    sat_id = getattr(getattr(sat, 'model', None), 'satnum', 'UNKNOWN')

    if err != 0:
        raise SGP4PropagationError(sat_name, sat_id, err, epoch)

    r_arr = np.array(r, dtype=float)
    v_arr = np.array(v, dtype=float)

    if not (np.all(np.isfinite(r_arr)) and np.all(np.isfinite(v_arr))):
        raise ValueError(
            f"SGP4 produced non-finite state vectors for {sat_name} (ID: {sat_id}) at {epoch.isoformat()}"
        )
    if np.linalg.norm(r_arr) < 1.0:
        raise ValueError(
            f"SGP4 produced degenerate zero position for {sat_name} (ID: {sat_id}) at {epoch.isoformat()}"
        )

    return r_arr, v_arr


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
    
    r_primary_tca: np.ndarray
    v_primary_tca: np.ndarray
    r_secondary_tca: np.ndarray
    v_secondary_tca: np.ndarray
    
    pre_burn_sma_km: float
    heuristic_risk_proxy: float
    is_risk: bool
    risk_threshold_km: float = SAFETY_DISTANCE_KM
    collision_probability_pc: float | None = None
    uncalibrated_collision_risk_indicator: float | None = None

    def __post_init__(self):
        if self.collision_probability_pc is None:
            self.collision_probability_pc = self.heuristic_risk_proxy
        if self.uncalibrated_collision_risk_indicator is None:
            self.uncalibrated_collision_risk_indicator = self.heuristic_risk_proxy

    def summary(self) -> str:
        status = "CRITICAL COLLISION RISK" if self.is_risk else "SAFE PASS"
        return (
            f"[{status}] {self.primary_name} x {self.secondary_name}\n"
            f"  TCA (UTC):                 {self.tca_epoch.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]} UTC\n"
            f"  Predicted Miss Distance:   {self.miss_distance_km:.4f} km (Threshold: {self.risk_threshold_km:.1f} km)\n"
            f"  Relative Speed:            {self.relative_velocity_km_s * 1000.0:.2f} m/s ({self.relative_velocity_km_s:.4f} km/s | {self.relative_velocity_km_s * 3600.0:.2f} km/h)\n"
            f"  Heuristic Risk Proxy:      {self.heuristic_risk_proxy:.2e} [UNRELIABLE / UNCALIBRATED]\n"
            f"  [WARNING] Collision risk indicator is an uncalibrated 2D isotropic proxy, NOT an operational covariance-based Pc. Do not use for operational threshold compliance.\n"
            f"  Pre-Burn SMA:              {self.pre_burn_sma_km:.3f} km"
        )

def compute_heuristic_risk_proxy(
    miss_distance_km: float,
    combined_sigma_km: float = 0.150,
    hard_body_radius_km: float = 0.010
) -> float:
    """Estimates an uncalibrated 2D geometric collision risk indicator proxy on the encounter B-plane.
    
    WARNING: This calculation is a simplified isotropic circular-normal heuristic proxy.
    It is NOT an operational covariance-based collision probability (which requires full 3D
    state error covariances propagated to TCA). It must NEVER be used to claim compliance
    with an operational collision probability threshold (e.g. 1e-4).
    """
    if miss_distance_km <= 0:
        return 1.0
    u = (miss_distance_km ** 2) / (2.0 * (combined_sigma_km ** 2))
    factor = (hard_body_radius_km ** 2) / (2.0 * (combined_sigma_km ** 2))
    pc = 1.0 - np.exp(-factor)
    pc_scaled = float(pc * np.exp(-u))
    return min(1.0, max(1e-12, pc_scaled))

def compute_2d_collision_probability(
    miss_distance_km: float,
    combined_sigma_km: float = 0.150,
    hard_body_radius_km: float = 0.010
) -> float:
    """Alias for compute_heuristic_risk_proxy for backward compatibility."""
    return compute_heuristic_risk_proxy(
        miss_distance_km=miss_distance_km,
        combined_sigma_km=combined_sigma_km,
        hard_body_radius_km=hard_body_radius_km
    )

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
            epoch_dt = epoch_input if epoch_input.tzinfo else epoch_input.replace(tzinfo=timezone.utc)
            jd, fraction = datetime_to_sgp4_jd(epoch_dt)
        elif hasattr(epoch_input, 'utc_datetime'):
            epoch_dt = epoch_input.utc_datetime()
            jd, fraction = datetime_to_sgp4_jd(epoch_dt)
        else:
            jd = epoch_input.whole
            fraction = epoch_input.tai_fraction - epoch_input._leap_seconds() / 86400.0
            epoch_dt = datetime(1970, 1, 1, tzinfo=timezone.utc)

        r1, v1 = propagate_sgp4_safe(sat1, jd, fraction, epoch_dt)
        r2, v2 = propagate_sgp4_safe(sat2, jd, fraction, epoch_dt)
        
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
        
        distances_coarse = []
        for sec in time_offsets:
            dt = start_time + timedelta(seconds=float(sec))
            jd, fraction = datetime_to_sgp4_jd(dt)
            r1, _ = propagate_sgp4_safe(sat1, jd, fraction, dt)
            r2, _ = propagate_sgp4_safe(sat2, jd, fraction, dt)
            diff = r2 - r1
            distances_coarse.append(float(np.linalg.norm(diff)))
        distances_coarse = np.array(distances_coarse)
        
        min_idx = int(np.argmin(distances_coarse))
        t_coarse_min_sec = float(time_offsets[min_idx])
        
        bracket_min = max(0.0, t_coarse_min_sec - 2.0 * self.coarse_step_sec)
        bracket_max = min(total_seconds, t_coarse_min_sec + 2.0 * self.coarse_step_sec)
        
        def scalar_dist_obj(t_sec: float) -> float:
            dt = start_time + timedelta(seconds=float(t_sec))
            jd, fraction = datetime_to_sgp4_jd(dt)
            r1, _ = propagate_sgp4_safe(sat1, jd, fraction, dt)
            r2, _ = propagate_sgp4_safe(sat2, jd, fraction, dt)
            diff = r2 - r1
            return float(np.linalg.norm(diff))
            
        opt_res = minimize_scalar(
            scalar_dist_obj,
            bounds=(bracket_min, bracket_max),
            method='bounded',
            options={'xatol': 1e-4}
        )
        
        refined_tca_sec = float(opt_res.x)
        tca_epoch = start_time + timedelta(seconds=refined_tca_sec)
        
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
            heuristic_risk_proxy=pc,
            collision_probability_pc=pc,
            is_risk=is_risk,
            risk_threshold_km=self.safety_distance_km
        )
