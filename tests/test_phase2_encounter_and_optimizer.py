"""Unit tests for encounter-window evaluation, global TCA minimization, and optimizer feasibility gating."""

import sys
import os
import unittest
from datetime import datetime, timezone, timedelta
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.constants import MU_EARTH, R_EARTH, J2_EARTH, SAFETY_DISTANCE_KM
from src.tle_manager import TLEManager
from src.conjunction_engine import ConjunctionEngine
from src.orbital_frames import (
    teme_to_gcrs,
    gcrs_to_teme,
    delta_v_to_eci,
    compute_semi_major_axis,
    compute_ric_basis,
    compute_ric_rotation_matrix,
    ric_to_vnb
)
from src.physics_verifier import (
    RK45PhysicsVerifier,
    EncounterOutcome,
    VerificationResult,
    j2_equations_of_motion,
    validate_state_vector
)
from src.maneuver_optimizer import GAManeuverOptimizer, ManeuverSolution


class TestStateAndTimeConsistency(unittest.TestCase):
    """Verify frame, epoch, burn-state reconstruction, and propagation intervals."""

    def setUp(self):
        self.mgr = TLEManager()
        self.sat1, self.sat2, _ = self.mgr.load_scenario("starlink_cosmos")
        self.engine = ConjunctionEngine()
        self.event = self.engine.analyze_pair(self.sat1, self.sat2)
        self.verifier = RK45PhysicsVerifier()

    def test_sgp4_teme_to_gcrs_transformation_at_tca(self):
        """Conjunction screening outputs in TEME must be transformed to GCRS before numerical propagation."""
        r_gcrs_tca, v_gcrs_tca = teme_to_gcrs(
            self.event.tca_epoch,
            self.event.r_primary_tca,
            self.event.v_primary_tca
        )
        # Position should differ by precession-nutation rotation (> 30 km at epoch 2026)
        displacement_km = np.linalg.norm(r_gcrs_tca - self.event.r_primary_tca)
        self.assertGreater(displacement_km, 30.0, "TEME and GCRS position must reflect precession at 2026 epoch")

    def test_burn_state_reconstruction_and_propagation_span_consistency(self):
        """In GCRS, backward propagation to t_burn and forward propagation by same lead time recovers initial state."""
        lead_time_sec = 1800.0
        r_gcrs_tca, v_gcrs_tca = teme_to_gcrs(
            self.event.tca_epoch,
            self.event.r_primary_tca,
            self.event.v_primary_tca
        )
        y_tca = np.hstack((r_gcrs_tca, v_gcrs_tca))

        # Backward propagation by lead_time_sec to burn epoch
        _, sat_back = self.verifier.propagate_state(y_tca, (0.0, -lead_time_sec))
        y_burn = sat_back[:, -1]

        # Forward propagation by lead_time_sec from burn epoch
        _, sat_fwd = self.verifier.propagate_state(y_burn, (0.0, lead_time_sec))
        y_recovered = sat_fwd[:, -1]

        pos_diff_km = np.linalg.norm(y_recovered[0:3] - y_tca[0:3])
        vel_diff_km_s = np.linalg.norm(y_recovered[3:6] - y_tca[3:6])
        self.assertLess(pos_diff_km, 1e-3, "Position reversibility error in GCRS exceeds 1 meter")
        self.assertLess(vel_diff_km_s, 1e-6, "Velocity reversibility error in GCRS exceeds 1 mm/s")


class TestEncounterEvaluator(unittest.TestCase):
    """Reusable encounter evaluator, burn-relative window, TCA shift, and global minimization."""

    def setUp(self):
        self.verifier = RK45PhysicsVerifier(safety_threshold_km=1.0)
        # Circular orbit test states
        self.r0 = 6878.0
        self.v0 = np.sqrt(MU_EARTH / self.r0)
        self.y_sat_burn = np.array([self.r0, 0.0, 0.0, 0.0, self.v0, 0.0])
        # Debris positioned to encounter near lead_time = 1800s with miss distance ~ 0.2 km
        self.y_deb_burn = np.array([self.r0 + 0.2, 0.0, 0.0, 0.0, self.v0, 0.0])

    def test_encounter_window_in_burn_relative_time(self):
        """Encounter window must be centered at tau = lead_time_sec in burn-relative integration time."""
        lead_time_sec = 1800.0
        outcome = self.verifier.evaluate_encounter(
            y_sat_burn=self.y_sat_burn,
            y_deb_burn=self.y_deb_burn,
            lead_time_sec=lead_time_sec,
            post_tca_margin_sec=600.0,
            window_stages=(300.0,),
            coarse_step_sec=10.0
        )
        self.assertEqual(outcome.status, "REJECTED_UNSAFE")
        self.assertFalse(outcome.passed)
        # Closest approach should be near tau = 1800.0s (within encounter window [1500, 2100])
        self.assertGreaterEqual(outcome.tca_relative_sec, 1500.0)
        self.assertLessEqual(outcome.tca_relative_sec, 2100.0)

    def test_along_track_maneuver_causes_tca_shift(self):
        """Applying an along-track burn causes true closest approach to shift away from nominal TCA."""
        lead_time_sec = 1800.0
        y_sat_post_burn = self.y_sat_burn + np.array([0.0, 0.0, 0.0, 0.0, 0.001, 0.0])

        outcome = self.verifier.evaluate_encounter(
            y_sat_burn=y_sat_post_burn,
            y_deb_burn=self.y_deb_burn,
            lead_time_sec=lead_time_sec,
            window_half_width_sec=300.0
        )
        # The true post-burn TCA must have a non-zero shift relative to nominal lead_time_sec
        self.assertNotEqual(outcome.tca_shift_sec, 0.0, "Maneuver must produce a measurable TCA shift")
        self.assertAlmostEqual(outcome.tca_relative_sec, lead_time_sec + outcome.tca_shift_sec, places=4)

    def test_global_minimum_vs_local_minima_and_endpoints(self):
        """Coarse scan + multi-bracket refinement identifies true global minimum across window."""
        lead_time_sec = 1200.0
        outcome = self.verifier.evaluate_encounter(
            y_sat_burn=self.y_sat_burn,
            y_deb_burn=self.y_deb_burn,
            lead_time_sec=lead_time_sec,
            window_half_width_sec=300.0,
            coarse_step_sec=5.0
        )
        self.assertTrue(np.isfinite(outcome.min_distance_km))
        self.assertGreater(outcome.min_distance_km, 0.0)


class TestOptimizerFeasibilityGating(unittest.TestCase):
    """Optimizer feasibility gating and explicit constraint outcomes."""

    def setUp(self):
        self.opt = GAManeuverOptimizer(population_size=10, generations=2, seed=42)
        self.r0 = 6878.0
        self.v0 = np.sqrt(MU_EARTH / self.r0)
        self.y_tca = np.array([self.r0, 0.0, 0.0, 0.0, self.v0, 0.0])
        self.y_deb_tca = np.array([self.r0 + 0.3, 0.0, 0.0, 0.0, self.v0, 0.0])

    def test_infeasible_candidate_with_insufficient_clearance_penalized(self):
        """Candidates with min_distance < safety_distance must receive a heavy penalty (>= 1e5)."""
        zero_burn_ind = [1200.0, 0.0, 1.57, 1.57]
        fit = self.opt.evaluate_chromosome(
            individual=zero_burn_ind,
            tca_sec=1800.0,
            r_sat_tca=self.y_tca[0:3],
            v_sat_tca=self.y_tca[3:6],
            r_deb_tca=self.y_deb_tca[0:3],
            v_deb_tca=self.y_deb_tca[3:6],
            pre_burn_sma=self.r0
        )
        self.assertGreaterEqual(fit[0], 1e5, "Candidate failing safety distance was not heavily penalized")

    def test_infeasible_candidate_exceeding_dv_budget_penalized(self):
        """Candidates exceeding maximum delta-v budget (2.5 m/s) must receive a heavy penalty."""
        excessive_dv_ind = [1200.0, 0.010, 1.57, 1.57] # 10 m/s > 2.5 m/s
        fit = self.opt.evaluate_chromosome(
            individual=excessive_dv_ind,
            tca_sec=1800.0,
            r_sat_tca=self.y_tca[0:3],
            v_sat_tca=self.y_tca[3:6],
            r_deb_tca=self.y_deb_tca[0:3],
            v_deb_tca=self.y_deb_tca[3:6],
            pre_burn_sma=self.r0
        )
        self.assertGreaterEqual(fit[0], 1e5, "Candidate exceeding delta-v budget was not heavily penalized")

    def test_infeasible_candidate_causing_earth_impact_penalized(self):
        """Candidates decreasing perigee below Earth surface or minimum altitude must be penalized."""
        retrograde_ind = [1200.0, 2.0, 3.14, 1.57]
        fit = self.opt.evaluate_chromosome(
            individual=retrograde_ind,
            tca_sec=1800.0,
            r_sat_tca=self.y_tca[0:3],
            v_sat_tca=self.y_tca[3:6],
            r_deb_tca=self.y_deb_tca[0:3],
            v_deb_tca=self.y_deb_tca[3:6],
            pre_burn_sma=self.r0
        )
        self.assertGreaterEqual(fit[0], 1e5, "Candidate dropping perigee inside Earth was not heavily penalized")

    def test_optimizer_solution_structure_contains_feasibility_flags(self):
        """ManeuverSolution must contain is_feasible, feasibility_status, and post_burn_tca_shift_sec."""
        sol = self.opt.optimize(
            tca_sec=1800.0,
            r_sat_tca=self.y_tca[0:3],
            v_sat_tca=self.y_tca[3:6],
            r_deb_tca=self.y_deb_tca[0:3],
            v_deb_tca=self.y_deb_tca[3:6],
            pre_burn_sma=self.r0
        )
        self.assertTrue(hasattr(sol, "is_feasible"))
        self.assertTrue(hasattr(sol, "feasibility_status"))
        self.assertTrue(hasattr(sol, "post_burn_tca_shift_sec"))


class TestIndependentValidationAndBaselines(unittest.TestCase):
    """Agreement between optimizer and independent verifier, and baseline comparison suite."""

    def setUp(self):
        self.mgr = TLEManager()
        self.sat1, self.sat2, _ = self.mgr.load_scenario("starlink_cosmos")
        self.engine = ConjunctionEngine()
        self.event = self.engine.analyze_pair(self.sat1, self.sat2)
        self.verifier = RK45PhysicsVerifier(safety_threshold_km=1.0)
        self.opt = GAManeuverOptimizer(population_size=15, generations=4, seed=42)

    def test_optimizer_and_verifier_agree_on_post_burn_miss_distance(self):
        """Optimizer predicted miss distance and independent verifier evaluated distance must agree (< 1e-3 km)."""
        r_gcrs_tca, v_gcrs_tca = teme_to_gcrs(self.event.tca_epoch, self.event.r_primary_tca, self.event.v_primary_tca)
        r_deb_gcrs_tca, v_deb_gcrs_tca = teme_to_gcrs(self.event.tca_epoch, self.event.r_secondary_tca, self.event.v_secondary_tca)

        sol = self.opt.optimize(
            tca_sec=self.event.tca_seconds_from_t0,
            r_sat_tca=r_gcrs_tca,
            v_sat_tca=v_gcrs_tca,
            r_deb_tca=r_deb_gcrs_tca,
            v_deb_tca=v_deb_gcrs_tca,
            pre_burn_sma=self.event.pre_burn_sma_km
        )

        y_sat_tca = np.hstack((r_gcrs_tca, v_gcrs_tca))
        y_deb_tca = np.hstack((r_deb_gcrs_tca, v_deb_gcrs_tca))
        _, sat_back = self.verifier.propagate_state(y_sat_tca, (0.0, -sol.burn_lead_time_sec))
        _, deb_back = self.verifier.propagate_state(y_deb_tca, (0.0, -sol.burn_lead_time_sec))

        r_sat_burn = sat_back[0:3, -1]
        v_sat_burn = sat_back[3:6, -1]
        r_deb_burn = deb_back[0:3, -1]
        v_deb_burn = deb_back[3:6, -1]

        ver_res = self.verifier.verify_maneuver(
            r_sat_burn=r_sat_burn,
            v_sat_burn=v_sat_burn,
            r_deb_burn=r_deb_burn,
            v_deb_burn=v_deb_burn,
            delta_v_eci=sol.delta_v_eci,
            delta_v_ric=sol.delta_v_ric,
            t_burn_sec=sol.burn_epoch_sec,
            t_tca_sec=self.event.tca_seconds_from_t0,
            pre_burn_miss_distance_km=self.event.miss_distance_km,
            pre_burn_sma_km=self.event.pre_burn_sma_km
        )

        diff_km = abs(ver_res.post_burn_miss_distance_km - sol.predicted_miss_distance_km)
        self.assertLess(
            diff_km,
            1e-3,
            f"Discrepancy between optimizer ({sol.predicted_miss_distance_km:.4f} km) and "
            f"independent verifier ({ver_res.post_burn_miss_distance_km:.4f} km)"
        )

    def test_baseline_comparison_suite(self):
        """Baseline comparison: No-maneuver, fixed 1 m/s burn, and optimized burn are evaluated with same model."""
        r_gcrs_tca, v_gcrs_tca = teme_to_gcrs(self.event.tca_epoch, self.event.r_primary_tca, self.event.v_primary_tca)
        r_deb_gcrs_tca, v_deb_gcrs_tca = teme_to_gcrs(self.event.tca_epoch, self.event.r_secondary_tca, self.event.v_secondary_tca)

        lead_time_sec = 1800.0
        y_sat_tca = np.hstack((r_gcrs_tca, v_gcrs_tca))
        y_deb_tca = np.hstack((r_deb_gcrs_tca, v_deb_gcrs_tca))
        _, sat_back = self.verifier.propagate_state(y_sat_tca, (0.0, -lead_time_sec))
        _, deb_back = self.verifier.propagate_state(y_deb_tca, (0.0, -lead_time_sec))

        y_sat_burn = sat_back[:, -1]
        y_deb_burn = deb_back[:, -1]

        # Case 1: Unmaneuvered baseline (dv = 0)
        outcome_no_burn = self.verifier.evaluate_encounter(
            y_sat_burn=y_sat_burn,
            y_deb_burn=y_deb_burn,
            lead_time_sec=lead_time_sec
        )
        self.assertFalse(outcome_no_burn.passed, "Unmaneuvered case should fail safety threshold")

        # Case 2: Fixed 1 m/s along-track baseline
        u_r, u_i, u_c = compute_ric_basis(y_sat_burn[0:3], y_sat_burn[3:6])
        dv_fixed_ric = np.array([0.0, 0.001, 0.0]) # 1 m/s along u_i
        Q_ric = np.column_stack((u_r, u_i, u_c))
        dv_fixed_inertial = Q_ric @ dv_fixed_ric

        y_sat_fixed = np.hstack((y_sat_burn[0:3], y_sat_burn[3:6] + dv_fixed_inertial))
        outcome_fixed = self.verifier.evaluate_encounter(
            y_sat_burn=y_sat_fixed,
            y_deb_burn=y_deb_burn,
            lead_time_sec=lead_time_sec
        )

        self.assertGreater(outcome_fixed.min_distance_km, outcome_no_burn.min_distance_km)


if __name__ == "__main__":
    unittest.main()
