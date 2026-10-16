"""Genetic algorithm optimizer for impulsive collision avoidance maneuvers."""

import sys
import os

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import random
import numpy as np
from deap import base, creator, tools

from src.constants import (
    MU_EARTH,
    SAFETY_DISTANCE_KM,
    DELTA_V_REF_KM_S,
    SMA_REF_KM,
    WEIGHT_FUEL,
    WEIGHT_SMA,
    WEIGHT_SAFETY
)
from src.orbital_frames import (
    delta_v_to_eci,
    spherical_to_ric_vector,
    compute_semi_major_axis
)
from src.physics_verifier import j2_equations_of_motion, RK45PhysicsVerifier, DEFAULT_ENCOUNTER_WINDOW_STAGES

@dataclass
class ManeuverSolution:
    """Optimal maneuver parameters and predicted outcome."""
    burn_lead_time_sec: float
    burn_epoch_sec: float
    delta_v_mag_km_s: float
    delta_v_mag_m_s: float
    theta_rad: float
    phi_rad: float
    delta_v_ric: np.ndarray
    delta_v_eci: np.ndarray
    predicted_miss_distance_km: float
    orbit_sma_deviation_km: float
    fitness_score: float
    generations_run: int
    population_size: int
    convergence_history: list[float]
    is_feasible: bool = True
    feasibility_status: str = "FEASIBLE"
    post_burn_tca_shift_sec: float = 0.0
    is_boundary_minimum: bool = False
    window_half_width_sec: float = 300.0
    expansion_history: list = None
    t_tca_offset_sec: float = 0.0
    nominal_tca_epoch: datetime | None = None
    burn_epoch_utc: datetime | None = None
    screening_epoch_utc: datetime | None = None
    post_burn_tca_epoch: datetime | None = None

    def summary(self) -> str:
        u_r, u_i, u_c = self.delta_v_ric / (self.delta_v_mag_km_s + 1e-12)
        tag = "[OPTIMIZED MANEUVER CANDIDATE - FEASIBLE_UNDER_MODEL]" if self.is_feasible else f"[OPTIMIZED MANEUVER CANDIDATE - {self.feasibility_status}]"
        bound_note = " (TRUNCATED AT WINDOW BOUNDARY - NOT DEMONSTRATED SAFE)" if self.is_boundary_minimum else " (interior minimum)"
        t_burn_dt_str = f" ({self.burn_epoch_utc.strftime('%Y-%m-%d %H:%M:%S UTC')})" if self.burn_epoch_utc else ""
        t_tca_dt_str = f" ({self.nominal_tca_epoch.strftime('%Y-%m-%d %H:%M:%S UTC')})" if self.nominal_tca_epoch else ""
        t0_dt_str = f" ({self.screening_epoch_utc.strftime('%Y-%m-%d %H:%M:%S UTC')})" if self.screening_epoch_utc else ""
        post_tca_dt_str = f" ({self.post_burn_tca_epoch.strftime('%Y-%m-%d %H:%M:%S UTC')})" if self.post_burn_tca_epoch else ""
        lines = [
            f"{tag}",
            f"  Feasibility:            {self.feasibility_status}",
            f"  Propagation Model:      RK45 Numerical Integration (Two-Body + Earth J2 Perturbation)",
            f"  Reference Frame:        GCRS / EarthMJ2000Eq (Inertial)",
            f"  Timeline Verification (5 Quantities):",
            f"    Screening Epoch (t0):     0.0 s (reference epoch){t0_dt_str}",
            f"    Nominal TCA (t_TCA):      {self.t_tca_offset_sec:.1f} s from t0{t_tca_dt_str}",
            f"    Burn Epoch Offset:        {self.burn_epoch_sec:.1f} s from t0{t_burn_dt_str}",
            f"    Burn Lead Time (Δt_lead): {self.burn_lead_time_sec:.1f} s ({self.burn_lead_time_sec / 60.0:.1f} mins before nominal TCA: t_TCA - t_burn)",
            f"    Encounter Time (tau):     tau = {self.burn_lead_time_sec + self.post_burn_tca_shift_sec:.1f} s from burn (TCA shift = {self.post_burn_tca_shift_sec:+.2f} s; nominal TCA at tau = {self.burn_lead_time_sec:.1f} s){post_tca_dt_str}",
            f"  Delta-V Required:       {self.delta_v_mag_m_s:.2f} m/s ({self.delta_v_mag_km_s * 1000:.2f} m/s)",
            f"  RIC Unit Vector:        R={u_r:+.3f}, I={u_i:+.3f}, C={u_c:+.3f}",
            f"  RIC Spherical Angles:   theta={np.degrees(self.theta_rad):.1f} deg, phi={np.degrees(self.phi_rad):.1f} deg",
            f"  Predicted Clearance:    {self.predicted_miss_distance_km:.4f} km{bound_note}",
            f"  Encounter Window:       Final half-width = ±{self.window_half_width_sec:.1f} s around nominal TCA",
            f"  Post-Burn TCA Shift:    {self.post_burn_tca_shift_sec:+.2f} s",
            f"  Orbit Deviation (|Δa|): {self.orbit_sma_deviation_km:.4f} km",
            f"  GA Fitness Score (J):   {self.fitness_score:.4f}"
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

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)

class GAManeuverOptimizer:
    """Optimizes burn timing, magnitude, and RIC direction using DEAP."""

    def __init__(
        self,
        population_size: int = 40,
        generations: int = 20,
        crossover_prob: float = 0.7,
        mutation_prob: float = 0.3,
        safety_distance_km: float = SAFETY_DISTANCE_KM,
        seed: int = 42,
        window_stages: tuple[float, ...] | None = None
    ):
        self.pop_size = population_size
        self.generations = generations
        self.cx_prob = crossover_prob
        self.mut_prob = mutation_prob
        self.safety_distance_km = safety_distance_km
        self.window_stages = window_stages or DEFAULT_ENCOUNTER_WINDOW_STAGES
        random.seed(seed)
        np.random.seed(seed)
        self.verifier = RK45PhysicsVerifier(safety_threshold_km=safety_distance_km)

    def _clamp_bounds(self, individual: list) -> list:
        """Clamps candidate parameters within physical search bounds."""
        max_lead = min(3600.0, self.tca_sec) if hasattr(self, 'tca_sec') and self.tca_sec is not None else 3600.0
        min_lead = min(900.0, max(60.0, max_lead * 0.5))
        individual[0] = float(np.clip(individual[0], min_lead, max_lead))
        individual[1] = float(np.clip(individual[1], 0.0001, 0.0025))
        individual[2] = float(individual[2] % (2.0 * np.pi))
        individual[3] = float(np.clip(individual[3], 0.0, np.pi))
        return individual

    def evaluate_chromosome(
        self,
        individual: list,
        tca_sec: float,
        r_sat_tca: np.ndarray,
        v_sat_tca: np.ndarray,
        r_deb_tca: np.ndarray,
        v_deb_tca: np.ndarray,
        pre_burn_sma: float
    ) -> tuple[float]:
        """Evaluates fitness: weighted sum of fuel, orbit deviation, and hard safety constraints."""
        lead_time_sec, dv_mag, theta, phi = individual
        
        if lead_time_sec > tca_sec or lead_time_sec < 60.0:
            return (1e6,)
        if dv_mag > 0.0025 or dv_mag < 0.0001:
            return (1e6,)

        t_burn_sec = tca_sec - lead_time_sec
        burn_duration = lead_time_sec
        
        y_sat_tca = np.hstack((r_sat_tca, v_sat_tca))
        y_deb_tca = np.hstack((r_deb_tca, v_deb_tca))
        
        try:
            _, sat_back = self.verifier.propagate_state(y_sat_tca, (0.0, -burn_duration))
            _, deb_back = self.verifier.propagate_state(y_deb_tca, (0.0, -burn_duration))
        except Exception:
            return (1e6,)
        
        r_sat_burn = sat_back[0:3, -1]
        v_sat_burn = sat_back[3:6, -1]
        r_deb_burn = deb_back[0:3, -1]
        v_deb_burn = deb_back[3:6, -1]
        
        dv_eci = delta_v_to_eci(r_sat_burn, v_sat_burn, dv_mag, theta, phi)
        v_sat_post_burn = v_sat_burn + dv_eci
        y_sat_post = np.hstack((r_sat_burn, v_sat_post_burn))
        y_deb_start = np.hstack((r_deb_burn, v_deb_burn))
        
        outcome = self.verifier.evaluate_encounter(
            y_sat_burn=y_sat_post,
            y_deb_burn=y_deb_start,
            lead_time_sec=lead_time_sec,
            window_stages=self.window_stages
        )
        
        if outcome.status in ("EARTH_IMPACT_DETECTED", "SOLVER_FAILED", "PERIGEE_TOO_LOW"):
            return (1e6,)
            
        post_miss_km = outcome.min_distance_km
        
        if post_miss_km < self.safety_distance_km:
            deficit = self.safety_distance_km - post_miss_km
            return (1e5 + (deficit * 1e4),)
            
        if outcome.is_boundary_minimum:
            return (5e4,)
            
        post_sma = compute_semi_major_axis(r_sat_burn, v_sat_post_burn, MU_EARTH)
        sma_deviation = abs(post_sma - pre_burn_sma)
        
        fuel_cost = dv_mag / DELTA_V_REF_KM_S
        sma_cost = sma_deviation / SMA_REF_KM
        
        total_fitness = (WEIGHT_FUEL * fuel_cost) + (WEIGHT_SMA * sma_cost)
        return (total_fitness,)

    def optimize(
        self,
        tca_sec: float,
        r_sat_tca: np.ndarray,
        v_sat_tca: np.ndarray,
        r_deb_tca: np.ndarray,
        v_deb_tca: np.ndarray,
        pre_burn_sma: float,
        nominal_tca_epoch: datetime | None = None
    ) -> ManeuverSolution:
        """Runs the genetic algorithm and returns the best maneuver candidate."""
        self.tca_sec = tca_sec
        toolbox = base.Toolbox()
        
        max_lead_init = min(3000.0, tca_sec)
        min_lead_init = min(900.0, max(60.0, max_lead_init * 0.25))
        if min_lead_init >= max_lead_init:
            min_lead_init = max(10.0, max_lead_init * 0.5)
            
        toolbox.register("attr_lead", random.uniform, min_lead_init, max_lead_init)
        toolbox.register("attr_dv", random.uniform, 0.0003, 0.0018)
        toolbox.register("attr_theta", random.uniform, 0.0, 2.0 * np.pi)
        toolbox.register("attr_phi", random.uniform, 0.1, np.pi - 0.1)
        
        toolbox.register(
            "individual",
            tools.initCycle,
            creator.Individual,
            (toolbox.attr_lead, toolbox.attr_dv, toolbox.attr_theta, toolbox.attr_phi),
            n=1
        )
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        
        toolbox.register(
            "evaluate",
            self.evaluate_chromosome,
            tca_sec=tca_sec,
            r_sat_tca=r_sat_tca,
            v_sat_tca=v_sat_tca,
            r_deb_tca=r_deb_tca,
            v_deb_tca=v_deb_tca,
            pre_burn_sma=pre_burn_sma
        )
        
        toolbox.register("mate", tools.cxBlend, alpha=0.4)
        toolbox.register(
            "mutate",
            tools.mutGaussian,
            mu=0.0,
            sigma=[300.0, 0.0003, 0.5, 0.3],
            indpb=0.25
        )
        toolbox.register("select", tools.selTournament, tournsize=3)
        
        population = toolbox.population(n=self.pop_size)
        hof = tools.HallOfFame(1)
        
        fitnesses = list(map(toolbox.evaluate, population))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
            
        hof.update(population)
        convergence_history = [hof[0].fitness.values[0]]
        
        for gen in range(self.generations):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))
            
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.cx_prob:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                    
            for mutant in offspring:
                if random.random() < self.mut_prob:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
                    
            for ind in offspring:
                self._clamp_bounds(ind)
                
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
                
            population[:] = offspring
            hof.update(population)
            convergence_history.append(hof[0].fitness.values[0])
            
        best = hof[0]
        lead_time, dv_mag, theta, phi = best
        if lead_time > tca_sec:
            lead_time = tca_sec
        t_burn_sec = tca_sec - lead_time
        
        y_sat_tca = np.hstack((r_sat_tca, v_sat_tca))
        y_deb_tca = np.hstack((r_deb_tca, v_deb_tca))
        _, sat_back = self.verifier.propagate_state(y_sat_tca, (0.0, -lead_time))
        _, deb_back = self.verifier.propagate_state(y_deb_tca, (0.0, -lead_time))
        r_sat_burn = sat_back[0:3, -1]
        v_sat_burn = sat_back[3:6, -1]
        r_deb_burn = deb_back[0:3, -1]
        v_deb_burn = deb_back[3:6, -1]
        
        dv_ric = spherical_to_ric_vector(dv_mag, theta, phi)
        dv_eci = delta_v_to_eci(r_sat_burn, v_sat_burn, dv_mag, theta, phi)
        v_sat_post_burn = v_sat_burn + dv_eci
        y_sat_post = np.hstack((r_sat_burn, v_sat_post_burn))
        y_deb_start = np.hstack((r_deb_burn, v_deb_burn))
        
        outcome = self.verifier.evaluate_encounter(
            y_sat_burn=y_sat_post,
            y_deb_burn=y_deb_start,
            lead_time_sec=lead_time,
            window_stages=self.window_stages
        )
        
        post_sma = compute_semi_major_axis(r_sat_burn, v_sat_post_burn, MU_EARTH)
        sma_deviation = abs(post_sma - pre_burn_sma)
        
        is_safe = outcome.min_distance_km >= self.safety_distance_km
        is_dv_ok = (dv_mag <= 0.0025 + 1e-9) and (dv_mag >= 0.0001 - 1e-9)
        is_timing_ok = (0.0 <= t_burn_sec < tca_sec)
        is_solver_ok = (outcome.status in ("FEASIBLE_UNDER_MODEL", "SAFE", "CERTIFIED")) and not outcome.is_boundary_minimum
        
        is_feasible = bool(is_safe and is_dv_ok and is_timing_ok and is_solver_ok)
        if not is_feasible:
            if not is_safe:
                feasibility_status = "INFEASIBLE_SAFETY_VIOLATION"
            elif not is_dv_ok:
                feasibility_status = "INFEASIBLE_DV_EXCEEDED"
            elif not is_timing_ok:
                feasibility_status = "INFEASIBLE_TIMING_INVALID"
            elif outcome.is_boundary_minimum:
                feasibility_status = "WINDOW_BOUNDARY_MINIMUM"
            else:
                feasibility_status = f"INFEASIBLE_{outcome.status}"
        else:
            feasibility_status = "FEASIBLE_UNDER_MODEL"
            
        burn_epoch_utc = nominal_tca_epoch - timedelta(seconds=lead_time) if nominal_tca_epoch else None
        screening_epoch_utc = nominal_tca_epoch - timedelta(seconds=tca_sec) if nominal_tca_epoch else None
        post_burn_tca_epoch = burn_epoch_utc + timedelta(seconds=outcome.tca_relative_sec) if burn_epoch_utc else None

        return ManeuverSolution(
            burn_lead_time_sec=lead_time,
            burn_epoch_sec=t_burn_sec,
            delta_v_mag_km_s=dv_mag,
            delta_v_mag_m_s=dv_mag * 1000.0,
            theta_rad=theta,
            phi_rad=phi,
            delta_v_ric=dv_ric,
            delta_v_eci=dv_eci,
            predicted_miss_distance_km=outcome.min_distance_km,
            orbit_sma_deviation_km=sma_deviation,
            fitness_score=best.fitness.values[0],
            generations_run=self.generations,
            population_size=self.pop_size,
            convergence_history=convergence_history,
            is_feasible=is_feasible,
            feasibility_status=feasibility_status,
            post_burn_tca_shift_sec=outcome.tca_shift_sec,
            is_boundary_minimum=outcome.is_boundary_minimum,
            window_half_width_sec=outcome.window_half_width_sec,
            expansion_history=outcome.expansion_history,
            t_tca_offset_sec=tca_sec,
            nominal_tca_epoch=nominal_tca_epoch,
            burn_epoch_utc=burn_epoch_utc,
            screening_epoch_utc=screening_epoch_utc,
            post_burn_tca_epoch=post_burn_tca_epoch
        )
