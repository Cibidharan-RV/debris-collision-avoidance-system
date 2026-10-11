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
from src.physics_verifier import j2_equations_of_motion, RK45PhysicsVerifier

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

    def summary(self) -> str:
        u_r, u_i, u_c = self.delta_v_ric / (self.delta_v_mag_km_s + 1e-12)
        return (
            f"[OPTIMIZED MANEUVER CANDIDATE]\n"
            f"  Burn Lead Time:         {self.burn_lead_time_sec / 60.0:.1f} mins before TCA\n"
            f"  Delta-V Required:       {self.delta_v_mag_m_s:.2f} m/s ({self.delta_v_mag_km_s * 1000:.2f} m/s)\n"
            f"  RIC Unit Vector:        R={u_r:+.3f}, I={u_i:+.3f}, C={u_c:+.3f}\n"
            f"  RIC Spherical Angles:   theta={np.degrees(self.theta_rad):.1f} deg, phi={np.degrees(self.phi_rad):.1f} deg\n"
            f"  Predicted Clearance:    {self.predicted_miss_distance_km:.4f} km\n"
            f"  Orbit Deviation (|da|): {self.orbit_sma_deviation_km:.4f} km\n"
            f"  GA Fitness Score (J):   {self.fitness_score:.4f}"
        )

# DEAP initialization
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
        seed: int = 42
    ):
        self.pop_size = population_size
        self.generations = generations
        self.cx_prob = crossover_prob
        self.mut_prob = mutation_prob
        self.safety_distance_km = safety_distance_km
        random.seed(seed)
        np.random.seed(seed)
        self.verifier = RK45PhysicsVerifier(safety_threshold_km=safety_distance_km)

    def _clamp_bounds(self, individual: list) -> list:
        """Clamps candidate parameters within physical search bounds."""
        # lead time: 15 to 60 min
        individual[0] = float(np.clip(individual[0], 900.0, 3600.0))
        # delta-v: 0.1 to 2.5 m/s
        individual[1] = float(np.clip(individual[1], 0.0001, 0.0025))
        # theta: [0, 2pi)
        individual[2] = float(individual[2] % (2.0 * np.pi))
        # phi: [0, pi]
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
        """Evaluates fitness: weighted sum of fuel, orbit deviation, and safety penalty."""
        lead_time_sec, dv_mag, theta, phi = individual
        t_burn_sec = max(0.0, tca_sec - lead_time_sec)
        burn_duration = lead_time_sec
        
        # Propagate back to burn epoch
        y_sat_tca = np.hstack((r_sat_tca, v_sat_tca))
        y_deb_tca = np.hstack((r_deb_tca, v_deb_tca))
        
        _, sat_back = self.verifier.propagate_state(y_sat_tca, (0.0, -burn_duration))
        _, deb_back = self.verifier.propagate_state(y_deb_tca, (0.0, -burn_duration))
        
        r_sat_burn = sat_back[0:3, -1]
        v_sat_burn = sat_back[3:6, -1]
        
        # Apply delta-v impulse at burn epoch
        dv_eci = delta_v_to_eci(r_sat_burn, v_sat_burn, dv_mag, theta, phi)
        v_sat_post_burn = v_sat_burn + dv_eci
        y_sat_post = np.hstack((r_sat_burn, v_sat_post_burn))
        
        # Propagate forward to TCA
        _, sat_fwd = self.verifier.propagate_state(y_sat_post, (0.0, burn_duration))
        r_sat_post_tca = sat_fwd[0:3, -1]
        
        post_miss_km = float(np.linalg.norm(r_deb_tca - r_sat_post_tca))
        
        post_sma = compute_semi_major_axis(r_sat_burn, v_sat_post_burn, MU_EARTH)
        sma_deviation = abs(post_sma - pre_burn_sma)
        
        fuel_cost = dv_mag / DELTA_V_REF_KM_S
        sma_cost = sma_deviation / SMA_REF_KM
        
        if post_miss_km < self.safety_distance_km:
            deficit = (self.safety_distance_km - post_miss_km) / self.safety_distance_km
            safety_penalty = (deficit ** 2) * WEIGHT_SAFETY
        else:
            safety_penalty = 0.0
            
        total_fitness = (WEIGHT_FUEL * fuel_cost) + (WEIGHT_SMA * sma_cost) + safety_penalty
        return (total_fitness,)

    def optimize(
        self,
        tca_sec: float,
        r_sat_tca: np.ndarray,
        v_sat_tca: np.ndarray,
        r_deb_tca: np.ndarray,
        v_deb_tca: np.ndarray,
        pre_burn_sma: float
    ) -> ManeuverSolution:
        """Runs the genetic algorithm and returns the best maneuver candidate."""
        toolbox = base.Toolbox()
        
        toolbox.register("attr_lead", random.uniform, 900.0, 3000.0)
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
        t_burn_sec = max(0.0, tca_sec - lead_time)
        
        # Re-evaluate best candidate
        y_sat_tca = np.hstack((r_sat_tca, v_sat_tca))
        _, sat_back = self.verifier.propagate_state(y_sat_tca, (0.0, -lead_time))
        r_sat_burn = sat_back[0:3, -1]
        v_sat_burn = sat_back[3:6, -1]
        
        dv_ric = spherical_to_ric_vector(dv_mag, theta, phi)
        dv_eci = delta_v_to_eci(r_sat_burn, v_sat_burn, dv_mag, theta, phi)
        v_sat_post_burn = v_sat_burn + dv_eci
        y_sat_post = np.hstack((r_sat_burn, v_sat_post_burn))
        
        _, sat_fwd = self.verifier.propagate_state(y_sat_post, (0.0, lead_time))
        r_sat_post_tca = sat_fwd[0:3, -1]
        post_miss_km = float(np.linalg.norm(r_deb_tca - r_sat_post_tca))
        
        post_sma = compute_semi_major_axis(r_sat_burn, v_sat_post_burn, MU_EARTH)
        sma_deviation = abs(post_sma - pre_burn_sma)
        
        return ManeuverSolution(
            burn_lead_time_sec=lead_time,
            burn_epoch_sec=t_burn_sec,
            delta_v_mag_km_s=dv_mag,
            delta_v_mag_m_s=dv_mag * 1000.0,
            theta_rad=theta,
            phi_rad=phi,
            delta_v_ric=dv_ric,
            delta_v_eci=dv_eci,
            predicted_miss_distance_km=post_miss_km,
            orbit_sma_deviation_km=sma_deviation,
            fitness_score=best.fitness.values[0],
            generations_run=self.generations,
            population_size=self.pop_size,
            convergence_history=convergence_history
        )

if __name__ == "__main__":
    from src.tle_manager import TLEManager
    from src.conjunction_engine import ConjunctionEngine
    
    mgr = TLEManager()
    sat1, sat2, _ = mgr.load_scenario("starlink_cosmos")
    engine = ConjunctionEngine()
    event = engine.analyze_pair(sat1, sat2)
    print(event.summary())
    
    if event.is_risk:
        print("\nRisk detected! Running maneuver optimizer...")
        opt = GAManeuverOptimizer(population_size=30, generations=15)
        sol = opt.optimize(
            event.tca_seconds_from_t0,
            event.r_primary_tca,
            event.v_primary_tca,
            event.r_secondary_tca,
            event.v_secondary_tca,
            event.pre_burn_sma_km
        )
        print(sol.summary())
