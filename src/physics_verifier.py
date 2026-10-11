"""Numerical orbit propagation with RK45 and Earth J2 gravity."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from src.constants import MU_EARTH, R_EARTH, J2_EARTH, SAFETY_DISTANCE_KM
from src.orbital_frames import compute_semi_major_axis

@dataclass
class VerificationResult:
    """Maneuver verification outcome and trajectory data."""
    passed: bool
    status: str
    pre_burn_miss_distance_km: float
    post_burn_miss_distance_km: float
    clearance_margin_km: float
    delta_v_magnitude_m_s: float
    delta_v_eci_km_s: np.ndarray
    delta_v_ric_km_s: np.ndarray
    orbit_sma_deviation_km: float
    burn_epoch_offset_sec: float
    t_eval_history: np.ndarray
    sat_trajectory_history: np.ndarray
    debris_trajectory_history: np.ndarray

    def summary(self) -> str:
        tag = "[CERTIFIED SAFE - PASS]" if self.passed else "[VERIFICATION REJECTED - FAIL]"
        return (
            f"{tag}\n"
            f"  Pre-Burn Miss Distance:   {self.pre_burn_miss_distance_km:.4f} km\n"
            f"  Post-Burn Miss Distance:  {self.post_burn_miss_distance_km:.4f} km\n"
            f"  Safety Margin (>1.0 km):  +{self.clearance_margin_km:.4f} km\n"
            f"  Delta-V Expended:         {self.delta_v_magnitude_m_s:.2f} m/s\n"
            f"  Orbit Deviation (|Δa|):   {self.orbit_sma_deviation_km:.4f} km\n"
            f"  Physics Integrator:       RK45 (Two-Body + Earth J2 Perturbation)"
        )

def j2_equations_of_motion(t: float, y: np.ndarray) -> np.ndarray:
    """Computes state derivative [v, a] with two-body gravity and J2 geopotential."""
    rx, ry, rz, vx, vy, vz = y
    r2 = rx*rx + ry*ry + rz*rz
    r = np.sqrt(r2)
    
    if r < 100.0:
        return np.zeros(6)
        
    r3 = r2 * r
    r5 = r2 * r3
    
    # Keplerian gravity
    a_kepler = -MU_EARTH / r3
    
    # J2 harmonic acceleration
    j2_coeff = 1.5 * J2_EARTH * MU_EARTH * (R_EARTH ** 2) / r5
    z2_over_r2 = (rz * rz) / r2
    
    ax = (a_kepler * rx) + (j2_coeff * rx * (5.0 * z2_over_r2 - 1.0))
    ay = (a_kepler * ry) + (j2_coeff * ry * (5.0 * z2_over_r2 - 1.0))
    az = (a_kepler * rz) + (j2_coeff * rz * (5.0 * z2_over_r2 - 3.0))
    
    return np.array([vx, vy, vz, ax, ay, az])

class RK45PhysicsVerifier:
    """RK45 numerical orbit propagator with J2 perturbation."""

    def __init__(self, safety_threshold_km: float = SAFETY_DISTANCE_KM):
        self.safety_threshold_km = safety_threshold_km

    def propagate_state(
        self,
        y0: np.ndarray,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrates orbital state y0 = [r, v] across t_span."""
        sol = solve_ivp(
            fun=j2_equations_of_motion,
            t_span=t_span,
            y0=y0,
            method='RK45',
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-9
        )
        return sol.t, sol.y

    def verify_maneuver(
        self,
        r_sat_burn: np.ndarray,
        v_sat_burn: np.ndarray,
        r_deb_burn: np.ndarray,
        v_deb_burn: np.ndarray,
        delta_v_eci: np.ndarray,
        delta_v_ric: np.ndarray,
        t_burn_sec: float,
        t_tca_sec: float,
        pre_burn_miss_distance_km: float,
        pre_burn_sma_km: float
    ) -> VerificationResult:
        """Propagates post-burn trajectories past TCA to verify clearance."""
        # Apply burn impulse
        v_sat_post_burn = v_sat_burn + delta_v_eci
        y_sat_post = np.hstack((r_sat_burn, v_sat_post_burn))
        y_deb_start = np.hstack((r_deb_burn, v_deb_burn))
        
        # Propagate from burn epoch to 15 min past TCA
        propagation_duration = (t_tca_sec - t_burn_sec) + 900.0
        t_eval = np.linspace(0.0, propagation_duration, 150)
        
        _, sat_traj = self.propagate_state(y_sat_post, (0.0, propagation_duration), t_eval)
        _, deb_traj = self.propagate_state(y_deb_start, (0.0, propagation_duration), t_eval)
        
        sat_pos = sat_traj[0:3, :]
        deb_pos = deb_traj[0:3, :]
        diff = deb_pos - sat_pos
        distances = np.linalg.norm(diff, axis=0)
        
        # Search for minimum separation around expected TCA window
        t_target_sec = max(0.0, t_tca_sec - t_burn_sec)
        window_mask = (t_eval >= max(0.0, t_target_sec - 600.0)) & (t_eval <= min(propagation_duration, t_target_sec + 600.0))
        if np.any(window_mask):
            window_indices = np.where(window_mask)[0]
            coarse_min_idx = int(window_indices[np.argmin(distances[window_mask])])
        else:
            coarse_min_idx = int(np.argmin(distances))
        t_guess = float(t_eval[coarse_min_idx])
        
        t_min_bracket = max(0.0, t_guess - 120.0)
        t_max_bracket = min(propagation_duration, t_guess + 120.0)
        if t_max_bracket <= t_min_bracket:
            t_max_bracket = t_min_bracket + 1.0
            
        def dist_at_t(t_val: float) -> float:
            _, sol_s = self.propagate_state(y_sat_post, (0.0, t_val))
            _, sol_d = self.propagate_state(y_deb_start, (0.0, t_val))
            r_s = sol_s[0:3, -1]
            r_d = sol_d[0:3, -1]
            return float(np.linalg.norm(r_d - r_s))
        
        opt = minimize_scalar(
            dist_at_t,
            bounds=(t_min_bracket, t_max_bracket),
            method='bounded',
            options={'xatol': 1e-4}
        )
        post_burn_miss_dist = float(opt.fun)
        
        post_burn_sma = compute_semi_major_axis(r_sat_burn, v_sat_post_burn, MU_EARTH)
        orbit_deviation = abs(post_burn_sma - pre_burn_sma_km)
        
        dv_mag_m_s = float(np.linalg.norm(delta_v_eci) * 1000.0)
        clearance_margin = post_burn_miss_dist - self.safety_threshold_km
        passed = post_burn_miss_dist >= self.safety_threshold_km
        status = "CERTIFIED" if passed else "REJECTED_UNSAFE"
        
        return VerificationResult(
            passed=passed,
            status=status,
            pre_burn_miss_distance_km=pre_burn_miss_distance_km,
            post_burn_miss_distance_km=post_burn_miss_dist,
            clearance_margin_km=clearance_margin,
            delta_v_magnitude_m_s=dv_mag_m_s,
            delta_v_eci_km_s=delta_v_eci,
            delta_v_ric_km_s=delta_v_ric,
            orbit_sma_deviation_km=orbit_deviation,
            burn_epoch_offset_sec=t_burn_sec,
            t_eval_history=t_eval,
            sat_trajectory_history=sat_traj,
            debris_trajectory_history=deb_traj
        )

if __name__ == "__main__":
    verifier = RK45PhysicsVerifier()
    # Smoke test on circular LEO orbit
    r0 = np.array([6878.0, 0.0, 0.0])
    v0 = np.array([0.0, 7.61, 0.0])
    t_out, y_out = verifier.propagate_state(np.hstack((r0, v0)), (0.0, 5400.0))
    print(f"Propagated one orbit: {len(t_out)} steps. Final radius: {np.linalg.norm(y_out[0:3, -1]):.2f} km")
