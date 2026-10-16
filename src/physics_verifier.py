"""Numerical orbit propagation with RK45 and Earth J2 gravity."""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from src.constants import MU_EARTH, R_EARTH, J2_EARTH, SAFETY_DISTANCE_KM
from src.orbital_frames import compute_semi_major_axis

DEFAULT_ENCOUNTER_WINDOW_STAGES: tuple[float, ...] = (300.0, 600.0, 900.0, 1200.0)

@dataclass
class WindowEvaluationRecord:
    """Record of closest-approach evaluation within a specific encounter window."""
    window_half_width_sec: float
    min_distance_km: float
    tca_relative_sec: float
    tca_shift_sec: float
    is_boundary_minimum: bool

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
    is_boundary_minimum: bool = False
    window_half_width_sec: float = 300.0
    expansion_history: list[WindowEvaluationRecord] = None
    t_tca_offset_sec: float = 0.0
    burn_lead_time_sec: float = 0.0
    tca_relative_tau_sec: float = 0.0
    tca_shift_sec: float = 0.0
    nominal_tca_epoch: datetime | None = None
    burn_epoch_utc: datetime | None = None
    screening_epoch_utc: datetime | None = None
    post_burn_tca_epoch: datetime | None = None

    def summary(self) -> str:
        tag = "[FEASIBLE_UNDER_MODEL - PASS]" if self.passed else f"[INFEASIBLE - FAIL: {self.status}]"
        bound_note = " (TRUNCATED AT WINDOW BOUNDARY - NOT DEMONSTRATED SAFE)" if self.is_boundary_minimum else " (interior minimum)"
        t_burn_dt_str = f" ({self.burn_epoch_utc.strftime('%Y-%m-%d %H:%M:%S UTC')})" if self.burn_epoch_utc else ""
        t_tca_dt_str = f" ({self.nominal_tca_epoch.strftime('%Y-%m-%d %H:%M:%S UTC')})" if self.nominal_tca_epoch else ""
        t0_dt_str = f" ({self.screening_epoch_utc.strftime('%Y-%m-%d %H:%M:%S UTC')})" if self.screening_epoch_utc else ""
        post_tca_dt_str = f" ({self.post_burn_tca_epoch.strftime('%Y-%m-%d %H:%M:%S UTC')})" if self.post_burn_tca_epoch else ""
        lines = [
            f"{tag}",
            f"  Feasibility Status:       {self.status}",
            f"  Propagation Model:        RK45 Numerical Integration (Two-Body + Earth J2 Perturbation, Dormand-Prince)",
            f"  Reference Frame:          GCRS / EarthMJ2000Eq (Inertial)",
            f"  Pre-Burn Miss Distance:   {self.pre_burn_miss_distance_km:.4f} km",
            f"  Post-Burn Miss Distance:  {self.post_burn_miss_distance_km:.4f} km{bound_note}",
            f"  Safety Margin (>1.0 km):  {'+' if self.clearance_margin_km >= 0 else ''}{self.clearance_margin_km:.4f} km",
            f"  Delta-V Expended:         {self.delta_v_magnitude_m_s:.2f} m/s",
            f"  Orbit Deviation (|Δa|):   {self.orbit_sma_deviation_km:.4f} km",
            f"  Timeline Verification (5 Quantities):",
            f"    Screening Epoch (t0):     0.0 s (reference epoch){t0_dt_str}",
            f"    Nominal TCA (t_TCA):      {self.t_tca_offset_sec:.1f} s from t0{t_tca_dt_str}",
            f"    Burn Epoch Offset:        {self.burn_epoch_offset_sec:.1f} s from t0{t_burn_dt_str}",
            f"    Burn Lead Time (Δt_lead): {self.burn_lead_time_sec:.1f} s ({self.burn_lead_time_sec/60:.1f} mins before nominal TCA: t_TCA - t_burn)",
            f"    Encounter Time (tau):     tau = {self.tca_relative_tau_sec:.1f} s from burn (TCA shift = {self.tca_shift_sec:+.2f} s; nominal TCA at tau = {self.burn_lead_time_sec:.1f} s){post_tca_dt_str}",
            f"  Encounter Window Search:    Final half-width = ±{self.window_half_width_sec:.1f} s around nominal TCA"
        ]
        if self.expansion_history:
            lines.append("  Encounter Window Expansion History:")
            for rec in self.expansion_history:
                loc = "BOUNDARY" if rec.is_boundary_minimum else "INTERIOR"
                lines.append(f"    ±{rec.window_half_width_sec:5.0f}s: min={rec.min_distance_km:.4f} km, tau={rec.tca_relative_sec:.1f}s, shift={rec.tca_shift_sec:+.2f}s [{loc}]")
        lines.append("  Modeling Assumptions & Limitations:")
        lines.append("    - SGP4 propagation is utilized solely for initial screening and nominal TCA discovery.")
        lines.append("    - Maneuver evaluation utilizes Cartesian RK45 numerical integration with two-body gravity + Earth J2.")
        lines.append("    - SGP4 and numerical Cartesian J2 are distinct dynamical models; backward propagation from SGP4 state is approximate.")
        lines.append("    - Flight-ready operational assessment requires high-order gravity harmonics, atmospheric drag, SRP, and covariance propagation.")
        return "\n".join(lines)

@dataclass
class EncounterOutcome:
    """Outcome of encounter window evaluation."""
    min_distance_km: float
    tca_relative_sec: float
    tca_shift_sec: float
    sat_state_at_min: np.ndarray
    deb_state_at_min: np.ndarray
    passed: bool
    status: str
    is_boundary_minimum: bool = False
    window_half_width_sec: float = 300.0
    expansion_history: list[WindowEvaluationRecord] = None
    sat_sol: object = None
    deb_sol: object = None

    def window_comparison_summary(self) -> str:
        if not self.expansion_history:
            return "No window expansion history."
        lines = ["Encounter Window Expansion History:"]
        for rec in self.expansion_history:
            tag = "BOUNDARY MINIMUM" if rec.is_boundary_minimum else "INTERIOR MINIMUM"
            lines.append(
                f"  ±{rec.window_half_width_sec:5.1f}s window: Min Separation = {rec.min_distance_km:.4f} km, "
                f"tau = {rec.tca_relative_sec:.1f}s, TCA Shift = {rec.tca_shift_sec:+.2f}s [{tag}]"
            )
        return "\n".join(lines)

def check_perigee_altitude(y: np.ndarray, min_alt_km: float = 120.0) -> bool:
    """Returns True if osculating perigee radius is at or above R_EARTH + min_alt_km."""
    r = float(np.linalg.norm(y[0:3]))
    v = float(np.linalg.norm(y[3:6]))
    energy = 0.5 * (v ** 2) - (MU_EARTH / r)
    if energy >= 0:
        return True
    a = -MU_EARTH / (2.0 * energy)
    h_vec = np.cross(y[0:3], y[3:6])
    h = float(np.linalg.norm(h_vec))
    p = (h ** 2) / MU_EARTH
    ecc = float(np.sqrt(max(0.0, 1.0 - p / a)))
    r_perigee = a * (1.0 - ecc)
    return r_perigee >= (R_EARTH + min_alt_km)

def j2_equations_of_motion(t: float, y: np.ndarray) -> np.ndarray:
    """Computes state derivative [v, a] with two-body gravity and J2 geopotential."""
    rx, ry, rz, vx, vy, vz = y
    r2 = rx*rx + ry*ry + rz*rz
    r = np.sqrt(r2)
    
    if r < 1e-6:
        raise ValueError(f"Singular position radius (r = {r:.3e} km) in equations of motion.")
        
    r3 = r2 * r
    r5 = r2 * r3
    
    a_kepler = -MU_EARTH / r3
    j2_coeff = 1.5 * J2_EARTH * MU_EARTH * (R_EARTH ** 2) / r5
    z2_over_r2 = (rz * rz) / r2
    
    ax = (a_kepler * rx) + (j2_coeff * rx * (5.0 * z2_over_r2 - 1.0))
    ay = (a_kepler * ry) + (j2_coeff * ry * (5.0 * z2_over_r2 - 1.0))
    az = (a_kepler * rz) + (j2_coeff * rz * (5.0 * z2_over_r2 - 3.0))
    
    return np.array([vx, vy, vz, ax, ay, az])


def validate_state_vector(y: np.ndarray, name: str = "State vector") -> None:
    """Validates state vector dimensions, finiteness, and nonzero position."""
    if not isinstance(y, np.ndarray) or y.shape != (6,):
        raise ValueError(f"{name} must be a 1D NumPy array with shape (6,). Got {getattr(y, 'shape', None)}")
    if not np.all(np.isfinite(y)):
        raise ValueError(f"{name} contains non-finite values (NaN or Inf): {y}")
    r_norm = float(np.linalg.norm(y[0:3]))
    if r_norm < 1.0:
        raise ValueError(f"{name} has unphysical or zero position magnitude ({r_norm:.3e} km).")


def make_earth_surface_event(earth_radius_km: float = R_EARTH):
    """Terminal event triggered when trajectory radius reaches or drops below Earth surface."""
    def event(t: float, y: np.ndarray) -> float:
        return float(np.linalg.norm(y[0:3]) - earth_radius_km)
    event.terminal = True
    event.direction = 0
    return event


class RK45PhysicsVerifier:
    """RK45 numerical orbit propagator with J2 perturbation."""

    def __init__(
        self,
        safety_threshold_km: float = SAFETY_DISTANCE_KM,
        earth_radius_km: float = R_EARTH
    ):
        self.safety_threshold_km = safety_threshold_km
        self.earth_radius_km = earth_radius_km
        self.last_sol = None

    def propagate_state(
        self,
        y0: np.ndarray,
        t_span: tuple[float, float],
        t_eval: np.ndarray | None = None,
        check_earth_impact: bool = True,
        dense_output: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrates orbital state y0 = [r, v] across t_span."""
        validate_state_vector(y0, "Initial state y0")
        
        events = [make_earth_surface_event(self.earth_radius_km)] if check_earth_impact else None
        sol = solve_ivp(
            fun=j2_equations_of_motion,
            t_span=t_span,
            y0=y0,
            method='RK45',
            t_eval=t_eval,
            events=events,
            dense_output=dense_output,
            rtol=1e-8,
            atol=1e-9
        )
        self.last_sol = sol
        if not sol.success and sol.status != 1:
            raise RuntimeError(f"Numerical integration failed: {sol.message}")
        return sol.t, sol.y

    def evaluate_encounter(
        self,
        y_sat_burn: np.ndarray,
        y_deb_burn: np.ndarray,
        lead_time_sec: float,
        post_tca_margin_sec: float = 900.0,
        window_half_width_sec: float = 300.0,
        window_stages: tuple[float, ...] | None = None,
        coarse_step_sec: float = 10.0,
        boundary_tolerance_sec: float = 1.0
    ) -> EncounterOutcome:
        """Evaluates true post-burn closest approach across encounter window in burn-relative time.
        
        Burn-relative time coordinate: tau in [0, lead_time_sec + post_tca_margin_sec].
        Nominal TCA is at tau = lead_time_sec.
        Encounter window: [max(0, tau_nom - W), min(tau_max, tau_nom + W)].
        
        Checks for boundary-truncated minima:
        1. Evaluates initial encounter window (default ±300s).
        2. If the minimum separation is within boundary_tolerance_sec of either window edge,
           expands window (e.g. to ±600s, ±900s).
        3. Compares minimum separation and TCA shift across windows.
        4. If minimum remains at boundary after expansion, flags status as WINDOW_BOUNDARY_MINIMUM
           with passed=False, preventing silent truncation from masquerading as a true local minimum.
        """
        validate_state_vector(y_sat_burn, "Satellite burn state")
        validate_state_vector(y_deb_burn, "Debris burn state")
        
        if not check_perigee_altitude(y_sat_burn, min_alt_km=120.0):
            return EncounterOutcome(
                min_distance_km=0.0,
                tca_relative_sec=lead_time_sec,
                tca_shift_sec=0.0,
                sat_state_at_min=y_sat_burn,
                deb_state_at_min=y_deb_burn,
                passed=False,
                status="PERIGEE_TOO_LOW",
                window_half_width_sec=window_half_width_sec,
                expansion_history=[]
            )
            
        if window_stages is None:
            if window_half_width_sec == 300.0:
                window_stages = DEFAULT_ENCOUNTER_WINDOW_STAGES
            else:
                window_stages = (window_half_width_sec, window_half_width_sec * 2.0, window_half_width_sec * 3.0, window_half_width_sec * 4.0)
                
        max_w = max(window_stages)
        tau_max = lead_time_sec + max(post_tca_margin_sec, max_w + 300.0)
        events = [make_earth_surface_event(self.earth_radius_km)]
        
        sat_sol = solve_ivp(
            fun=j2_equations_of_motion,
            t_span=(0.0, tau_max),
            y0=y_sat_burn,
            method='RK45',
            events=events,
            dense_output=True,
            rtol=1e-8,
            atol=1e-9
        )
        deb_sol = solve_ivp(
            fun=j2_equations_of_motion,
            t_span=(0.0, tau_max),
            y0=y_deb_burn,
            method='RK45',
            events=events,
            dense_output=True,
            rtol=1e-8,
            atol=1e-9
        )
        
        if sat_sol.status == 1 or deb_sol.status == 1:
            return EncounterOutcome(
                min_distance_km=0.0,
                tca_relative_sec=lead_time_sec,
                tca_shift_sec=0.0,
                sat_state_at_min=y_sat_burn,
                deb_state_at_min=y_deb_burn,
                passed=False,
                status="EARTH_IMPACT_DETECTED",
                window_half_width_sec=window_half_width_sec,
                expansion_history=[],
                sat_sol=sat_sol,
                deb_sol=deb_sol
            )
            
        if not (sat_sol.success and deb_sol.success):
            return EncounterOutcome(
                min_distance_km=0.0,
                tca_relative_sec=lead_time_sec,
                tca_shift_sec=0.0,
                sat_state_at_min=y_sat_burn,
                deb_state_at_min=y_deb_burn,
                passed=False,
                status="SOLVER_FAILED",
                window_half_width_sec=window_half_width_sec,
                expansion_history=[],
                sat_sol=sat_sol,
                deb_sol=deb_sol
            )
            
        def dist_func(t: float) -> float:
            r_s = sat_sol.sol(t)[0:3]
            r_d = deb_sol.sol(t)[0:3]
            return float(np.linalg.norm(r_d - r_s))

        expansion_history: list[WindowEvaluationRecord] = []
        
        for w_stage in window_stages:
            tau_w_min = max(0.0, lead_time_sec - w_stage)
            tau_w_max = min(tau_max, lead_time_sec + w_stage)
            if tau_w_max <= tau_w_min:
                tau_w_max = tau_w_min + 1.0
                
            tau_grid = np.arange(tau_w_min, tau_w_max + coarse_step_sec, coarse_step_sec)
            if tau_grid[-1] < tau_w_max:
                tau_grid = np.append(tau_grid, tau_w_max)
                
            d_grid = np.array([dist_func(t) for t in tau_grid])
            candidates = [(tau_w_min, float(d_grid[0])), (tau_w_max, float(d_grid[-1]))]
            
            found_local = False
            for i in range(1, len(d_grid) - 1):
                if d_grid[i] <= d_grid[i - 1] and d_grid[i] <= d_grid[i + 1]:
                    b_min = float(tau_grid[i - 1])
                    b_max = float(tau_grid[i + 1])
                    res = minimize_scalar(dist_func, bounds=(b_min, b_max), method='bounded', options={'xatol': 1e-4})
                    candidates.append((float(res.x), float(res.fun)))
                    found_local = True
                    
            if not found_local:
                min_idx = int(np.argmin(d_grid))
                b_min = float(max(tau_w_min, tau_grid[max(0, min_idx - 1)]))
                b_max = float(min(tau_w_max, tau_grid[min(len(tau_grid) - 1, min_idx + 1)]))
                if b_max <= b_min:
                    b_max = b_min + 1.0
                res = minimize_scalar(dist_func, bounds=(b_min, b_max), method='bounded', options={'xatol': 1e-4})
                candidates.append((float(res.x), float(res.fun)))
                
            stage_tau, stage_dist = min(candidates, key=lambda c: c[1])
            stage_shift = stage_tau - lead_time_sec
            is_bound = (abs(stage_tau - tau_w_min) <= boundary_tolerance_sec) or (abs(stage_tau - tau_w_max) <= boundary_tolerance_sec)
            
            rec = WindowEvaluationRecord(
                window_half_width_sec=float(w_stage),
                min_distance_km=stage_dist,
                tca_relative_sec=stage_tau,
                tca_shift_sec=stage_shift,
                is_boundary_minimum=is_bound
            )
            expansion_history.append(rec)
            
            if not is_bound:
                break
                
        final_rec = expansion_history[-1]
        is_boundary_minimum = final_rec.is_boundary_minimum
        best_tau = final_rec.tca_relative_sec
        best_dist = final_rec.min_distance_km
        tca_shift = final_rec.tca_shift_sec
        final_window_w = final_rec.window_half_width_sec
        
        sat_state_min = sat_sol.sol(best_tau)
        deb_state_min = deb_sol.sol(best_tau)
        
        if is_boundary_minimum:
            passed = False
            if best_dist < self.safety_threshold_km:
                status = "REJECTED_UNSAFE"
            else:
                status = "WINDOW_BOUNDARY_MINIMUM"
        else:
            passed = best_dist >= self.safety_threshold_km
            status = "FEASIBLE_UNDER_MODEL" if passed else "REJECTED_UNSAFE"
            
        return EncounterOutcome(
            min_distance_km=best_dist,
            tca_relative_sec=best_tau,
            tca_shift_sec=tca_shift,
            sat_state_at_min=sat_state_min,
            deb_state_at_min=deb_state_min,
            passed=passed,
            status=status,
            is_boundary_minimum=is_boundary_minimum,
            window_half_width_sec=final_window_w,
            expansion_history=expansion_history,
            sat_sol=sat_sol,
            deb_sol=deb_sol
        )

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
        pre_burn_sma_km: float,
        nominal_tca_epoch: datetime | None = None,
        window_stages: tuple[float, ...] | None = None
    ) -> VerificationResult:
        if t_burn_sec < 0.0:
            raise ValueError(f"Burn epoch offset ({t_burn_sec:.1f}s) precedes screening window start (0.0s).")
        if t_burn_sec >= t_tca_sec:
            raise ValueError(f"Burn epoch offset ({t_burn_sec:.1f}s) must strictly precede TCA ({t_tca_sec:.1f}s).")

        v_sat_post_burn = v_sat_burn + delta_v_eci
        y_sat_post = np.hstack((r_sat_burn, v_sat_post_burn))
        y_deb_start = np.hstack((r_deb_burn, v_deb_burn))
        
        validate_state_vector(y_sat_post, "Post-burn satellite state")
        validate_state_vector(y_deb_start, "Debris initial state")
        
        lead_time_sec = t_tca_sec - t_burn_sec
        burn_epoch_utc = nominal_tca_epoch - timedelta(seconds=lead_time_sec) if nominal_tca_epoch else None
        screening_epoch_utc = nominal_tca_epoch - timedelta(seconds=t_tca_sec) if nominal_tca_epoch else None
        
        stages = window_stages if window_stages is not None else DEFAULT_ENCOUNTER_WINDOW_STAGES
        outcome = self.evaluate_encounter(
            y_sat_burn=y_sat_post,
            y_deb_burn=y_deb_start,
            lead_time_sec=lead_time_sec,
            post_tca_margin_sec=1500.0,
            window_stages=stages
        )
        post_burn_tca_epoch = burn_epoch_utc + timedelta(seconds=outcome.tca_relative_sec) if burn_epoch_utc else None
        
        dv_mag_m_s = float(np.linalg.norm(delta_v_eci) * 1000.0)
        
        if outcome.status in ("EARTH_IMPACT_DETECTED", "PERIGEE_TOO_LOW"):
            return VerificationResult(
                passed=False,
                status=outcome.status,
                pre_burn_miss_distance_km=pre_burn_miss_distance_km,
                post_burn_miss_distance_km=0.0,
                clearance_margin_km=-self.safety_threshold_km,
                delta_v_magnitude_m_s=dv_mag_m_s,
                delta_v_eci_km_s=delta_v_eci,
                delta_v_ric_km_s=delta_v_ric,
                orbit_sma_deviation_km=0.0,
                burn_epoch_offset_sec=t_burn_sec,
                t_eval_history=np.array([0.0]),
                sat_trajectory_history=np.zeros((6, 1)),
                debris_trajectory_history=np.zeros((6, 1)),
                is_boundary_minimum=False,
                window_half_width_sec=outcome.window_half_width_sec,
                expansion_history=outcome.expansion_history,
                t_tca_offset_sec=t_tca_sec,
                burn_lead_time_sec=lead_time_sec,
                tca_relative_tau_sec=outcome.tca_relative_sec,
                tca_shift_sec=outcome.tca_shift_sec,
                nominal_tca_epoch=nominal_tca_epoch,
                burn_epoch_utc=burn_epoch_utc,
                screening_epoch_utc=screening_epoch_utc,
                post_burn_tca_epoch=post_burn_tca_epoch
            )
            
        if outcome.status == "SOLVER_FAILED":
            return VerificationResult(
                passed=False,
                status="SOLVER_FAILED",
                pre_burn_miss_distance_km=pre_burn_miss_distance_km,
                post_burn_miss_distance_km=0.0,
                clearance_margin_km=-self.safety_threshold_km,
                delta_v_magnitude_m_s=dv_mag_m_s,
                delta_v_eci_km_s=delta_v_eci,
                delta_v_ric_km_s=delta_v_ric,
                orbit_sma_deviation_km=0.0,
                burn_epoch_offset_sec=t_burn_sec,
                t_eval_history=np.array([0.0]),
                sat_trajectory_history=np.zeros((6, 1)),
                debris_trajectory_history=np.zeros((6, 1)),
                is_boundary_minimum=False,
                window_half_width_sec=outcome.window_half_width_sec,
                expansion_history=outcome.expansion_history,
                t_tca_offset_sec=t_tca_sec,
                burn_lead_time_sec=lead_time_sec,
                tca_relative_tau_sec=outcome.tca_relative_sec,
                tca_shift_sec=outcome.tca_shift_sec,
                nominal_tca_epoch=nominal_tca_epoch,
                burn_epoch_utc=burn_epoch_utc,
                screening_epoch_utc=screening_epoch_utc,
                post_burn_tca_epoch=post_burn_tca_epoch
            )
            
        post_burn_sma = compute_semi_major_axis(r_sat_burn, v_sat_post_burn, MU_EARTH)
        orbit_deviation = abs(post_burn_sma - pre_burn_sma_km)
        clearance_margin = outcome.min_distance_km - self.safety_threshold_km
        
        t_eval = np.linspace(0.0, lead_time_sec + 1500.0, 150)
        sat_traj = outcome.sat_sol.sol(t_eval) if outcome.sat_sol else np.zeros((6, len(t_eval)))
        deb_traj = outcome.deb_sol.sol(t_eval) if outcome.deb_sol else np.zeros((6, len(t_eval)))
        
        return VerificationResult(
            passed=outcome.passed,
            status=outcome.status,
            pre_burn_miss_distance_km=pre_burn_miss_distance_km,
            post_burn_miss_distance_km=outcome.min_distance_km,
            clearance_margin_km=clearance_margin,
            delta_v_magnitude_m_s=dv_mag_m_s,
            delta_v_eci_km_s=delta_v_eci,
            delta_v_ric_km_s=delta_v_ric,
            orbit_sma_deviation_km=orbit_deviation,
            burn_epoch_offset_sec=t_burn_sec,
            t_eval_history=t_eval,
            sat_trajectory_history=sat_traj,
            debris_trajectory_history=deb_traj,
            is_boundary_minimum=outcome.is_boundary_minimum,
            window_half_width_sec=outcome.window_half_width_sec,
            expansion_history=outcome.expansion_history,
            t_tca_offset_sec=t_tca_sec,
            burn_lead_time_sec=lead_time_sec,
            tca_relative_tau_sec=outcome.tca_relative_sec,
            tca_shift_sec=outcome.tca_shift_sec,
            nominal_tca_epoch=nominal_tca_epoch,
            burn_epoch_utc=burn_epoch_utc,
            screening_epoch_utc=screening_epoch_utc,
            post_burn_tca_epoch=post_burn_tca_epoch
        )
