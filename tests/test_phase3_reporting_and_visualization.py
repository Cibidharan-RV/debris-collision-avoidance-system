"""Unit tests for reporting terminology, risk proxy qualification,
visualization scale integrity, and encounter-window boundary minimum gating.
"""

import sys
import os
import unittest
from datetime import datetime, timezone, timedelta
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.constants import MU_EARTH, R_EARTH, SAFETY_DISTANCE_KM
from src.tle_manager import TLEManager
from src.conjunction_engine import ConjunctionEngine, compute_heuristic_risk_proxy, compute_2d_collision_probability
from src.orbital_frames import teme_to_gcrs, delta_v_to_eci, spherical_to_ric_vector
from src.physics_verifier import RK45PhysicsVerifier, EncounterOutcome, VerificationResult
from src.maneuver_optimizer import GAManeuverOptimizer, ManeuverSolution
from src.visualizer import OrbitVisualizer


class TestEncounterWindowBoundaryHandling(unittest.TestCase):
    """Verify encounter-window expansion, boundary minimum detection, and window comparison records."""

    def setUp(self):
        self.verifier = RK45PhysicsVerifier(safety_threshold_km=1.0)
        self.mgr = TLEManager()
        self.sat1, self.sat2, _ = self.mgr.load_scenario("starlink_cosmos")
        self.engine = ConjunctionEngine()
        self.event = self.engine.analyze_pair(self.sat1, self.sat2)

    def test_monotonic_separation_expands_window_and_flags_boundary_minimum(self):
        """Monotonic post-burn separation starting near window edge expands across all 4 stages (±300s, ±600s, ±900s, ±1200s) and flags WINDOW_BOUNDARY_MINIMUM."""
        lead_time_sec = 1800.0
        r_gcrs_tca, v_gcrs_tca = teme_to_gcrs(self.event.tca_epoch, self.event.r_primary_tca, self.event.v_primary_tca)
        r_deb_gcrs_tca, v_deb_gcrs_tca = teme_to_gcrs(self.event.tca_epoch, self.event.r_secondary_tca, self.event.v_secondary_tca)

        y_sat_tca = np.hstack((r_gcrs_tca, v_gcrs_tca))
        y_deb_tca = np.hstack((r_deb_gcrs_tca, v_deb_gcrs_tca))
        _, sat_back = self.verifier.propagate_state(y_sat_tca, (0.0, -lead_time_sec))
        _, deb_back = self.verifier.propagate_state(y_deb_tca, (0.0, -lead_time_sec))
        r_sat_burn = sat_back[0:3, -1]
        v_sat_burn = sat_back[3:6, -1]
        r_deb_burn = deb_back[0:3, -1]
        v_deb_burn = deb_back[3:6, -1]

        # In-track burn of 2.5 m/s creates monotonic separation across the encounter window
        dv_eci = delta_v_to_eci(r_sat_burn, v_sat_burn, 0.0025, np.radians(90), np.radians(90))
        y_sat_post = np.hstack((r_sat_burn, v_sat_burn + dv_eci))
        y_deb_start = np.hstack((r_deb_burn, v_deb_burn))

        outcome = self.verifier.evaluate_encounter(
            y_sat_burn=y_sat_post,
            y_deb_burn=y_deb_start,
            lead_time_sec=lead_time_sec,
            window_stages=(300.0, 600.0, 900.0, 1200.0)
        )

        self.assertTrue(outcome.is_boundary_minimum, "Monotonic separation must be flagged as boundary minimum")
        self.assertEqual(outcome.status, "WINDOW_BOUNDARY_MINIMUM")
        self.assertFalse(outcome.passed, "Boundary-truncated minimum cannot pass verification")
        self.assertEqual(len(outcome.expansion_history), 4, "Window must expand across all 4 configured stages")
        self.assertEqual(outcome.expansion_history[0].window_half_width_sec, 300.0)
        self.assertEqual(outcome.expansion_history[1].window_half_width_sec, 600.0)
        self.assertEqual(outcome.expansion_history[2].window_half_width_sec, 900.0)
        self.assertEqual(outcome.expansion_history[3].window_half_width_sec, 1200.0)
        for rec in outcome.expansion_history:
            self.assertTrue(rec.is_boundary_minimum, f"Stage ±{rec.window_half_width_sec}s must be a boundary minimum")
        for i in range(1, len(outcome.expansion_history)):
            self.assertLess(outcome.expansion_history[i].min_distance_km, outcome.expansion_history[i-1].min_distance_km,
                            "Wider window must uncover lower separation for monotonic profile")

    def test_window_expansion_compares_separation_and_shift_across_windows(self):
        """EncounterOutcome compares minimum distance and shift across evaluated windows."""
        lead_time_sec = 1800.0
        r_gcrs_tca, v_gcrs_tca = teme_to_gcrs(self.event.tca_epoch, self.event.r_primary_tca, self.event.v_primary_tca)
        r_deb_gcrs_tca, v_deb_gcrs_tca = teme_to_gcrs(self.event.tca_epoch, self.event.r_secondary_tca, self.event.v_secondary_tca)

        y_sat_tca = np.hstack((r_gcrs_tca, v_gcrs_tca))
        y_deb_tca = np.hstack((r_deb_gcrs_tca, v_deb_gcrs_tca))
        _, sat_back = self.verifier.propagate_state(y_sat_tca, (0.0, -lead_time_sec))
        _, deb_back = self.verifier.propagate_state(y_deb_tca, (0.0, -lead_time_sec))
        r_sat_burn = sat_back[0:3, -1]
        v_sat_burn = sat_back[3:6, -1]
        r_deb_burn = deb_back[0:3, -1]
        v_deb_burn = deb_back[3:6, -1]

        dv_eci = delta_v_to_eci(r_sat_burn, v_sat_burn, 0.0025, np.radians(90), np.radians(90))
        y_sat_post = np.hstack((r_sat_burn, v_sat_burn + dv_eci))
        y_deb_start = np.hstack((r_deb_burn, v_deb_burn))

        outcome = self.verifier.evaluate_encounter(
            y_sat_burn=y_sat_post,
            y_deb_burn=y_deb_start,
            lead_time_sec=lead_time_sec,
            window_stages=(300.0, 600.0, 900.0, 1200.0)
        )

        rec1 = outcome.expansion_history[0]
        rec2 = outcome.expansion_history[1]
        rec3 = outcome.expansion_history[2]
        rec4 = outcome.expansion_history[3]
        self.assertGreater(rec1.min_distance_km, rec2.min_distance_km)
        self.assertGreater(rec2.min_distance_km, rec3.min_distance_km)
        self.assertGreater(rec3.min_distance_km, rec4.min_distance_km)
        self.assertEqual(rec1.tca_shift_sec, -300.0)
        self.assertEqual(rec2.tca_shift_sec, -600.0)
        self.assertEqual(rec3.tca_shift_sec, -900.0)
        self.assertEqual(rec4.tca_shift_sec, -1200.0)
        summary = outcome.window_comparison_summary()
        self.assertIn("±300.0s", summary)
        self.assertIn("±600.0s", summary)
        self.assertIn("±900.0s", summary)
        self.assertIn("±1200.0s", summary)
        self.assertIn("BOUNDARY", summary)

    def test_wider_window_uncovers_hidden_minimum_and_invalidates_safety(self):
        """A narrow window (±300s) falsely suggests a safe pass (5.51 km > 1.0 km),
        while expanding to ±1600s uncovers the true minimum (0.50 km < 1.0 km),
        proving why narrow-window truncation is dangerous and must be rejected."""
        lead_time_sec = 1800.0
        r_gcrs_tca, v_gcrs_tca = teme_to_gcrs(self.event.tca_epoch, self.event.r_primary_tca, self.event.v_primary_tca)
        r_deb_gcrs_tca, v_deb_gcrs_tca = teme_to_gcrs(self.event.tca_epoch, self.event.r_secondary_tca, self.event.v_secondary_tca)

        y_sat_tca = np.hstack((r_gcrs_tca, v_gcrs_tca))
        y_deb_tca = np.hstack((r_deb_gcrs_tca, v_deb_gcrs_tca))
        _, sat_back = self.verifier.propagate_state(y_sat_tca, (0.0, -lead_time_sec))
        _, deb_back = self.verifier.propagate_state(y_deb_tca, (0.0, -lead_time_sec))
        r_sat_burn = sat_back[0:3, -1]
        v_sat_burn = sat_back[3:6, -1]
        r_deb_burn = deb_back[0:3, -1]
        v_deb_burn = deb_back[3:6, -1]

        dv_eci = delta_v_to_eci(r_sat_burn, v_sat_burn, 0.0025, np.radians(90), np.radians(90))
        y_sat_post = np.hstack((r_sat_burn, v_sat_burn + dv_eci))
        y_deb_start = np.hstack((r_deb_burn, v_deb_burn))

        outcome = self.verifier.evaluate_encounter(
            y_sat_burn=y_sat_post,
            y_deb_burn=y_deb_start,
            lead_time_sec=lead_time_sec,
            window_stages=(300.0, 600.0, 900.0, 1200.0, 1600.0)
        )
        d_300 = outcome.expansion_history[0].min_distance_km
        self.assertGreater(d_300, 1.0, "Narrow window ±300s incorrectly appears safe")
        d_final = outcome.min_distance_km
        self.assertLess(d_final, 1.0, "Expanded window uncovers true hidden minimum below 1.0 km threshold")
        self.assertEqual(outcome.status, "REJECTED_UNSAFE")
        self.assertFalse(outcome.passed)

    def test_known_trajectory_below_safety_threshold_rejected(self):
        """A candidate trajectory with closest approach below 1.0 km is rejected with REJECTED_UNSAFE."""
        lead_time_sec = 1800.0
        r_gcrs_tca, v_gcrs_tca = teme_to_gcrs(self.event.tca_epoch, self.event.r_primary_tca, self.event.v_primary_tca)
        r_deb_gcrs_tca, v_deb_gcrs_tca = teme_to_gcrs(self.event.tca_epoch, self.event.r_secondary_tca, self.event.v_secondary_tca)

        y_sat_tca = np.hstack((r_gcrs_tca, v_gcrs_tca))
        y_deb_tca = np.hstack((r_deb_gcrs_tca, v_deb_gcrs_tca))
        _, sat_back = self.verifier.propagate_state(y_sat_tca, (0.0, -lead_time_sec))
        _, deb_back = self.verifier.propagate_state(y_deb_tca, (0.0, -lead_time_sec))

        outcome = self.verifier.evaluate_encounter(
            y_sat_burn=sat_back[:, -1],
            y_deb_burn=deb_back[:, -1],
            lead_time_sec=lead_time_sec,
            window_stages=(300.0, 600.0)
        )
        self.assertFalse(outcome.passed)
        self.assertEqual(outcome.status, "REJECTED_UNSAFE")
        self.assertLess(outcome.min_distance_km, 1.0)

    def test_interior_minimum_does_not_falsely_flag_boundary(self):
        """Unperturbed forward trajectory with true stationary minimum does not trigger false boundary flag."""
        lead_time_sec = 1800.0
        r_gcrs_tca, v_gcrs_tca = teme_to_gcrs(self.event.tca_epoch, self.event.r_primary_tca, self.event.v_primary_tca)
        r_deb_gcrs_tca, v_deb_gcrs_tca = teme_to_gcrs(self.event.tca_epoch, self.event.r_secondary_tca, self.event.v_secondary_tca)

        y_sat_tca = np.hstack((r_gcrs_tca, v_gcrs_tca))
        y_deb_tca = np.hstack((r_deb_gcrs_tca, v_deb_gcrs_tca))
        _, sat_back = self.verifier.propagate_state(y_sat_tca, (0.0, -lead_time_sec))
        _, deb_back = self.verifier.propagate_state(y_deb_tca, (0.0, -lead_time_sec))

        # Unmaneuvered (delta_v = 0) has interior minimum exactly at tau = 1800s
        outcome = self.verifier.evaluate_encounter(
            y_sat_burn=sat_back[:, -1],
            y_deb_burn=deb_back[:, -1],
            lead_time_sec=lead_time_sec,
            window_stages=(300.0, 600.0)
        )

        self.assertFalse(outcome.is_boundary_minimum, "True stationary minimum must not be flagged as boundary")
        self.assertAlmostEqual(outcome.tca_shift_sec, 0.0, delta=1.0)
        self.assertEqual(len(outcome.expansion_history), 1, "Interior minimum does not need secondary expansion")


class TestReportingTerminologyAndModelIdentification(unittest.TestCase):
    """Model labels, frames, windows, epochs, and exclusion of 'certified'."""

    def setUp(self):
        self.verifier = RK45PhysicsVerifier(safety_threshold_km=1.0)
        r0 = np.array([6878.0, 0.0, 0.0])
        v0 = np.array([0.0, 7.61, 0.0])
        self.res_pass = VerificationResult(
            passed=True,
            status="FEASIBLE_UNDER_MODEL",
            pre_burn_miss_distance_km=0.35,
            post_burn_miss_distance_km=2.45,
            clearance_margin_km=1.45,
            delta_v_magnitude_m_s=1.20,
            delta_v_eci_km_s=np.array([0.0, 0.0012, 0.0]),
            delta_v_ric_km_s=np.array([0.0, 0.0012, 0.0]),
            orbit_sma_deviation_km=2.1,
            burn_epoch_offset_sec=1069.0,
            t_eval_history=np.linspace(0, 1800, 10),
            sat_trajectory_history=np.zeros((6, 10)),
            debris_trajectory_history=np.zeros((6, 10)),
            is_boundary_minimum=False,
            window_half_width_sec=300.0
        )
        self.res_fail = VerificationResult(
            passed=False,
            status="WINDOW_BOUNDARY_MINIMUM",
            pre_burn_miss_distance_km=0.35,
            post_burn_miss_distance_km=3.46,
            clearance_margin_km=2.46,
            delta_v_magnitude_m_s=2.50,
            delta_v_eci_km_s=np.array([0.0, 0.0025, 0.0]),
            delta_v_ric_km_s=np.array([0.0, 0.0025, 0.0]),
            orbit_sma_deviation_km=5.8,
            burn_epoch_offset_sec=1069.0,
            t_eval_history=np.linspace(0, 1800, 10),
            sat_trajectory_history=np.zeros((6, 10)),
            debris_trajectory_history=np.zeros((6, 10)),
            is_boundary_minimum=True,
            window_half_width_sec=600.0
        )

    def test_summary_never_uses_certified_terminology(self):
        """Summaries must never use 'certified' or 'certified safe'; must prefer FEASIBLE_UNDER_MODEL or INFEASIBLE."""
        self.assertNotIn("CERTIFIED", self.res_pass.summary().upper())
        self.assertNotIn("CERTIFIED", self.res_fail.summary().upper())
        self.assertIn("FEASIBLE_UNDER_MODEL", self.res_pass.summary())
        self.assertIn("INFEASIBLE", self.res_fail.summary())

    def test_summary_identifies_propagation_model_frame_epoch_and_window(self):
        """Summary explicitly labels RK45 J2 model, GCRS frame, burn epoch, and encounter window."""
        summary = self.res_pass.summary()
        self.assertIn("RK45", summary)
        self.assertIn("Two-Body + Earth J2 Perturbation", summary)
        self.assertIn("GCRS / EarthMJ2000Eq", summary)
        self.assertIn("Burn Epoch Offset", summary)
        self.assertIn("Encounter Window", summary)

    def test_summary_identifies_propagation_model_limitations_notice(self):
        """Summary explicitly details modeling assumptions, SGP4 vs J2 distinctions, and flight readiness requirements."""
        summary = self.res_pass.summary()
        self.assertIn("Modeling Assumptions & Limitations:", summary)
        self.assertIn("SGP4 propagation is utilized solely for initial screening", summary)
        self.assertIn("Cartesian RK45 numerical integration with two-body gravity + Earth J2", summary)
        self.assertIn("Flight-ready operational assessment requires high-order gravity harmonics", summary)


class TestOptimizerInfeasibleCandidateGating(unittest.TestCase):
    """Confirm infeasible and boundary candidates cannot be presented as successful avoidance solutions."""

    def setUp(self):
        self.verifier = RK45PhysicsVerifier(safety_threshold_km=1.0)
        self.mgr = TLEManager()
        self.sat1, self.sat2, _ = self.mgr.load_scenario("starlink_cosmos")
        self.engine = ConjunctionEngine()
        self.event = self.engine.analyze_pair(self.sat1, self.sat2)
        self.opt = GAManeuverOptimizer(population_size=15, generations=4, seed=42)

    def test_boundary_minimum_candidate_gated_from_success(self):
        """Candidates resulting in boundary minima must be gated with is_feasible=False and status WINDOW_BOUNDARY_MINIMUM."""
        sol = self.opt.optimize(
            tca_sec=self.event.tca_seconds_from_t0,
            r_sat_tca=self.event.r_primary_tca,
            v_sat_tca=self.event.v_primary_tca,
            r_deb_tca=self.event.r_secondary_tca,
            v_deb_tca=self.event.v_secondary_tca,
            pre_burn_sma=self.event.pre_burn_sma_km
        )
        if sol.is_boundary_minimum:
            self.assertFalse(sol.is_feasible, "Boundary-truncated minimum must not be flagged feasible")
            self.assertEqual(sol.feasibility_status, "WINDOW_BOUNDARY_MINIMUM")
            self.assertNotIn("FEASIBLE_UNDER_MODEL", sol.summary())
            self.assertIn("WINDOW_BOUNDARY_MINIMUM", sol.summary())

    def test_independent_verifier_rejects_boundary_minimum_as_infeasible(self):
        """Independent verifier marks boundary minimum candidates as passed=False and INFEASIBLE."""
        r_gcrs_tca, v_gcrs_tca = teme_to_gcrs(self.event.tca_epoch, self.event.r_primary_tca, self.event.v_primary_tca)
        r_deb_gcrs_tca, v_deb_gcrs_tca = teme_to_gcrs(self.event.tca_epoch, self.event.r_secondary_tca, self.event.v_secondary_tca)

        lead_time_sec = 1800.0
        y_sat_tca = np.hstack((r_gcrs_tca, v_gcrs_tca))
        y_deb_tca = np.hstack((r_deb_gcrs_tca, v_deb_gcrs_tca))
        _, sat_back = self.verifier.propagate_state(y_sat_tca, (0.0, -lead_time_sec))
        _, deb_back = self.verifier.propagate_state(y_deb_tca, (0.0, -lead_time_sec))

        r_sat_burn = sat_back[0:3, -1]
        v_sat_burn = sat_back[3:6, -1]
        r_deb_burn = deb_back[0:3, -1]
        v_deb_burn = deb_back[3:6, -1]

        # In-track burn of 2.5 m/s
        dv_eci = delta_v_to_eci(r_sat_burn, v_sat_burn, 0.0025, np.radians(90), np.radians(90))
        dv_ric = np.array([0.0, 0.0025, 0.0])

        ver_res = self.verifier.verify_maneuver(
            r_sat_burn=r_sat_burn,
            v_sat_burn=v_sat_burn,
            r_deb_burn=r_deb_burn,
            v_deb_burn=v_deb_burn,
            delta_v_eci=dv_eci,
            delta_v_ric=dv_ric,
            t_burn_sec=self.event.tca_seconds_from_t0 - lead_time_sec,
            t_tca_sec=self.event.tca_seconds_from_t0,
            pre_burn_miss_distance_km=self.event.miss_distance_km,
            pre_burn_sma_km=self.event.pre_burn_sma_km
        )

        self.assertFalse(ver_res.passed, "Boundary-truncated verification result must have passed=False")
        self.assertEqual(ver_res.status, "WINDOW_BOUNDARY_MINIMUM")
        self.assertIn("INFEASIBLE", ver_res.summary())


class TestCollisionProbabilityProxyQualification(unittest.TestCase):
    """Renaming, qualification, and warning for the collision risk proxy."""

    def setUp(self):
        self.engine = ConjunctionEngine()
        self.mgr = TLEManager()
        self.sat1, self.sat2, _ = self.mgr.load_scenario("starlink_cosmos")
        self.event = self.engine.analyze_pair(self.sat1, self.sat2)

    def test_heuristic_risk_proxy_attribute_and_warning(self):
        """ConjunctionEvent exposes heuristic_risk_proxy and displays an uncalibrated proxy warning."""
        self.assertTrue(hasattr(self.event, "heuristic_risk_proxy"))
        self.assertTrue(hasattr(self.event, "uncalibrated_collision_risk_indicator"))
        self.assertAlmostEqual(self.event.heuristic_risk_proxy, self.event.collision_probability_pc, places=9)

        summary = self.event.summary()
        self.assertIn("Heuristic Risk Proxy", summary)
        self.assertIn("UNRELIABLE / UNCALIBRATED", summary)
        self.assertIn("[WARNING]", summary)
        self.assertIn("NOT an operational covariance-based Pc", summary)

    def test_proxy_function_docstring_contains_operational_disclaimer(self):
        """Proxy functions document that they must never claim operational compliance."""
        doc = compute_heuristic_risk_proxy.__doc__
        self.assertIn("WARNING", doc)
        self.assertIn("operational collision probability threshold", doc)

    def test_relative_velocity_unrounded_display_and_conversion(self):
        """Relative velocity conversion is computed from the unrounded float and displayed with unrounded precision."""
        v_rel_km_s = self.event.relative_velocity_km_s
        v_rel_m_s = v_rel_km_s * 1000.0
        v_rel_km_h = v_rel_km_s * 3600.0

        # For Starlink-1008 vs Cosmos-1408, relative speed is ~0.66 m/s (~0.00066 km/s)
        self.assertAlmostEqual(v_rel_m_s, 0.66, delta=0.05)
        self.assertAlmostEqual(v_rel_km_h, 2.38, delta=0.1)

        summary = self.event.summary()
        # Verify that summary displays m/s, km/s, and km/h matching the unrounded calculation
        self.assertIn(f"{v_rel_m_s:.2f} m/s", summary)
        self.assertIn(f"{v_rel_km_s:.4f} km/s", summary)
        self.assertIn(f"{v_rel_km_h:.2f} km/h", summary)
        # Verify that it is NOT spuriously converted from a pre-rounded 0.001 km/s (which would give 3.6 km/h)
        self.assertNotIn("3.6 km/h", summary)


class TestVisualizationScalesAndUnits(unittest.TestCase):
    """Visualizer units on every axis, magnified safety indicator annotations."""

    def setUp(self):
        self.vis = OrbitVisualizer(output_dir="output")
        self.mgr = TLEManager()
        self.sat1, self.sat2, _ = self.mgr.load_scenario("starlink_cosmos")
        self.engine = ConjunctionEngine()
        self.event = self.engine.analyze_pair(self.sat1, self.sat2)

    def test_scene_axes_include_km_units_and_gcrs_frame(self):
        """3D scene layout axes include units [km] and frame (GCRS / J2000)."""
        sat_traj = np.zeros((3, 20))
        deb_traj = np.zeros((3, 20))
        html_path = self.vis.generate_3d_encounter_scene(
            event=self.event,
            sat_traj=sat_traj,
            deb_traj=deb_traj,
            filename="test_scene.html"
        )
        self.assertTrue(os.path.exists(html_path))
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("X [km]", content)
        self.assertIn("Y [km]", content)
        self.assertIn("Z [km]", content)
        self.assertIn("J2000 Inertial", content)

    def test_safety_sphere_trace_annotates_magnification_factor(self):
        """Safety sphere surface trace clearly states 150x visual magnification for Earth scale."""
        sat_traj = np.zeros((3, 20))
        deb_traj = np.zeros((3, 20))
        html_path = self.vis.generate_3d_encounter_scene(
            event=self.event,
            sat_traj=sat_traj,
            deb_traj=deb_traj,
            filename="test_scene_scale.html"
        )
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()

        self.assertIn("magnified 150x", content)
        self.assertIn("Scale Note", content)


class TestTimelineConsistencyAndEpochAccounting(unittest.TestCase):
    """Verify exact relationships between t0, t_TCA, t_burn, lead time Δt, and propagation duration tau."""

    def setUp(self):
        self.verifier = RK45PhysicsVerifier(safety_threshold_km=1.0)
        self.mgr = TLEManager()
        self.sat1, self.sat2, _ = self.mgr.load_scenario("starlink_cosmos")
        self.engine = ConjunctionEngine()
        self.event = self.engine.analyze_pair(self.sat1, self.sat2)

    def test_timeline_five_quantities_consistency(self):
        """The 5 timeline quantities satisfy exact physical synchronization:
        1. t0 is reference origin (0.0s).
        2. t_TCA is nominal conjunction epoch offset from t0.
        3. t_burn is maneuver epoch offset from t0.
        4. Δt_lead = t_TCA - t_burn.
        5. Nominal TCA occurs at tau = Δt_lead when forward integration begins at the burn.
        """
        lead_time_sec = 1800.0
        t_burn_sec = self.event.tca_seconds_from_t0 - lead_time_sec
        r_gcrs_tca, v_gcrs_tca = teme_to_gcrs(self.event.tca_epoch, self.event.r_primary_tca, self.event.v_primary_tca)
        r_deb_gcrs_tca, v_deb_gcrs_tca = teme_to_gcrs(self.event.tca_epoch, self.event.r_secondary_tca, self.event.v_secondary_tca)

        y_sat_tca = np.hstack((r_gcrs_tca, v_gcrs_tca))
        y_deb_tca = np.hstack((r_deb_gcrs_tca, v_deb_gcrs_tca))
        _, sat_back = self.verifier.propagate_state(y_sat_tca, (0.0, -lead_time_sec))
        _, deb_back = self.verifier.propagate_state(y_deb_tca, (0.0, -lead_time_sec))

        r_sat_burn = sat_back[0:3, -1]
        v_sat_burn = sat_back[3:6, -1]
        r_deb_burn = deb_back[0:3, -1]
        v_deb_burn = deb_back[3:6, -1]

        dv_eci = delta_v_to_eci(r_sat_burn, v_sat_burn, 0.0025, np.radians(90), np.radians(90))
        dv_ric = np.array([0.0, 0.0025, 0.0])

        ver_res = self.verifier.verify_maneuver(
            r_sat_burn=r_sat_burn,
            v_sat_burn=v_sat_burn,
            r_deb_burn=r_deb_burn,
            v_deb_burn=v_deb_burn,
            delta_v_eci=dv_eci,
            delta_v_ric=dv_ric,
            t_burn_sec=t_burn_sec,
            t_tca_sec=self.event.tca_seconds_from_t0,
            pre_burn_miss_distance_km=self.event.miss_distance_km,
            pre_burn_sma_km=self.event.pre_burn_sma_km,
            nominal_tca_epoch=self.event.tca_epoch
        )

        # 1. Verification of relationship: Δt_lead = t_TCA - t_burn
        self.assertAlmostEqual(ver_res.burn_lead_time_sec, ver_res.t_tca_offset_sec - ver_res.burn_epoch_offset_sec, places=6)
        self.assertAlmostEqual(ver_res.burn_lead_time_sec, lead_time_sec, places=6)

        # 2. Nominal TCA corresponds to tau = Δt_lead
        self.assertAlmostEqual(ver_res.tca_relative_tau_sec - ver_res.tca_shift_sec, ver_res.burn_lead_time_sec, places=4)

        # 3. Epoch UTC arithmetic
        expected_burn_epoch = self.event.tca_epoch - timedelta(seconds=lead_time_sec)
        self.assertEqual(ver_res.burn_epoch_utc, expected_burn_epoch)

        expected_screening_epoch = self.event.tca_epoch - timedelta(seconds=self.event.tca_seconds_from_t0)
        self.assertEqual(ver_res.screening_epoch_utc, expected_screening_epoch)

        expected_encounter_epoch = expected_burn_epoch + timedelta(seconds=ver_res.tca_relative_tau_sec)
        self.assertEqual(ver_res.post_burn_tca_epoch, expected_encounter_epoch)

        # 4. Summary displays all 5 timeline quantities clearly
        summary = ver_res.summary()
        self.assertIn("Screening Epoch (t0):     0.0 s", summary)
        self.assertIn("Nominal TCA (t_TCA):", summary)
        self.assertIn("Burn Epoch Offset:", summary)
        self.assertIn("Burn Lead Time (Δt_lead):", summary)
        self.assertIn("Encounter Time (tau):", summary)


class TestIndependentNumericalVerification(unittest.TestCase):
    """Independent verification of the RK45 numerical propagator against an independent DOP853 solver."""

    def test_rk45_matches_independent_dop853_over_full_orbit(self):
        """Compare RK45 trajectory against SciPy DOP853 (Prince-Dormand RK89 order 8(5,3)) over 5400s under J2."""
        from scipy.integrate import solve_ivp
        from src.physics_verifier import j2_equations_of_motion

        # Initial circular LEO state at 500 km altitude, 51.6 deg inclination (norm = 6878 km)
        r0 = np.array([6878.0, 0.0, 0.0])
        v_circ = np.sqrt(MU_EARTH / 6878.0)
        v0 = np.array([0.0, v_circ * np.cos(np.radians(51.6)), v_circ * np.sin(np.radians(51.6))])
        y0 = np.hstack((r0, v0))
        t_span = (0.0, 5400.0)  # ~90 minute orbit

        # Solver 1: Production RK45
        verifier = RK45PhysicsVerifier()
        t_eval = np.linspace(0.0, 5400.0, 100)
        t_rk45, y_rk45 = verifier.propagate_state(y0, t_span, t_eval=t_eval)

        # Solver 2: Independent DOP853
        sol_dop = solve_ivp(
            fun=j2_equations_of_motion,
            t_span=t_span,
            y0=y0,
            method='DOP853',
            t_eval=t_eval,
            rtol=1e-11,
            atol=1e-12
        )

        pos_diff_km = np.linalg.norm(y_rk45[0:3, :] - sol_dop.y[0:3, :], axis=0)
        max_diff_m = np.max(pos_diff_km) * 1000.0
        # Agreement within 5 meters over an entire 90-minute orbit verifies numerical fidelity independently
        self.assertLess(max_diff_m, 5.0, f"RK45 and DOP853 differ by {max_diff_m:.3f} m, exceeds 5m tolerance")


if __name__ == "__main__":
    unittest.main()
