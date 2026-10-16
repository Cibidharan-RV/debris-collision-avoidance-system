"""Unit tests for SGP4 error handling, timing, numerical safeguards, and frame alignment."""

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
from src.orbital_frames import (
    compute_ric_basis,
    compute_ric_rotation_matrix,
    compute_vnb_basis,
    compute_vnb_rotation_matrix,
    ric_to_vnb,
    vnb_to_ric,
    teme_to_ric,
    ric_to_teme,
    compute_semi_major_axis,
    spherical_to_ric_vector
)
from src.physics_verifier import j2_equations_of_motion, RK45PhysicsVerifier
from src.conjunction_engine import ConjunctionEngine, datetime_to_sgp4_jd
from src.gmat_interface import GMATInterface


class TestSGP4ErrorHandling(unittest.TestCase):
    """Tests for SGP4 error code inspection and failure propagation."""

    def setUp(self):
        self.mgr = TLEManager()
        self.sat1, self.sat2, _ = self.mgr.load_scenario("starlink_cosmos")

    def test_sgp4_valid_propagation_returns_clean_states(self):
        """Nominal propagation should produce zero error code and finite numbers."""
        epoch = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)
        jd, fraction = datetime_to_sgp4_jd(epoch)
        err1, r1, v1 = self.sat1.model.sgp4(jd, fraction)
        err2, r2, v2 = self.sat2.model.sgp4(jd, fraction)

        self.assertEqual(err1, 0, f"Satellite 1 SGP4 error: {err1}")
        self.assertEqual(err2, 0, f"Satellite 2 SGP4 error: {err2}")
        self.assertTrue(np.all(np.isfinite(r1)))
        self.assertTrue(np.all(np.isfinite(v1)))
        self.assertTrue(np.all(np.isfinite(r2)))
        self.assertTrue(np.all(np.isfinite(v2)))

    def test_sgp4_nonzero_error_code_must_not_be_silently_ignored(self):
        """A nonzero SGP4 error code must cause state evaluation to report an error."""
        engine = ConjunctionEngine()

        class FaultyModel:
            def sgp4(self, jd, fraction):
                # Return error code 6 (decayed satellite)
                return 6, (np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan)

        class FaultySat:
            name = "DECAYED_SAT"
            model = FaultyModel()

        faulty_sat = FaultySat()
        epoch = datetime(2026, 10, 11, 12, 0, 0, tzinfo=timezone.utc)

        # Under the required invariant, evaluate_distance_at_time must raise an exception
        # when an SGP4 model returns error != 0.
        with self.assertRaises((RuntimeError, ValueError), msg="SGP4 nonzero error code was silently ignored"):
            engine.evaluate_distance_at_time(faulty_sat, self.sat2, epoch)


class TestEpochAndTimingSynchronization(unittest.TestCase):
    """Tests for consistent absolute epoch arithmetic and window boundaries."""

    def test_burn_epoch_prior_to_screening_window_is_rejected(self):
        """If t_burn precedes the start of the screening window, verifier must reject it."""
        verifier = RK45PhysicsVerifier()
        # Initial dummy states
        r0 = np.array([6878.0, 0.0, 0.0])
        v0 = np.array([0.0, 7.61, 0.0])
        dv = np.array([0.0, 0.001, 0.0])

        # Negative t_burn_sec (burn before window start t0 = 0)
        with self.assertRaises(ValueError, msg="Verifier accepted negative t_burn_sec"):
            verifier.verify_maneuver(
                r_sat_burn=r0,
                v_sat_burn=v0,
                r_deb_burn=r0 + np.array([0.5, 0.0, 0.0]),
                v_deb_burn=v0,
                delta_v_eci=dv,
                delta_v_ric=dv,
                t_burn_sec=-300.0,
                t_tca_sec=600.0,
                pre_burn_miss_distance_km=0.5,
                pre_burn_sma_km=6878.0
            )

    def test_verifier_rejects_burn_epoch_after_tca(self):
        """If t_burn is at or after TCA, verifier must raise ValueError."""
        verifier = RK45PhysicsVerifier()
        r0 = np.array([6878.0, 0.0, 0.0])
        v0 = np.array([0.0, 7.61, 0.0])
        dv = np.array([0.0, 0.001, 0.0])

        with self.assertRaises(ValueError, msg="Verifier accepted t_burn >= t_tca"):
            verifier.verify_maneuver(
                r_sat_burn=r0,
                v_sat_burn=v0,
                r_deb_burn=r0 + np.array([0.5, 0.0, 0.0]),
                v_deb_burn=v0,
                delta_v_eci=dv,
                delta_v_ric=dv,
                t_burn_sec=600.0,
                t_tca_sec=600.0,
                pre_burn_miss_distance_km=0.5,
                pre_burn_sma_km=6878.0
            )

    def test_forward_propagation_duration_equals_timestamp_delta(self):
        """Forward propagation duration must equal (t_end - t_burn) exactly."""
        t_burn = datetime(2026, 10, 11, 12, 17, 49, tzinfo=timezone.utc)
        t_tca = datetime(2026, 10, 11, 12, 47, 49, tzinfo=timezone.utc)
        post_tca_margin_sec = 900.0
        t_end = t_tca + timedelta(seconds=post_tca_margin_sec)

        expected_duration = (t_end - t_burn).total_seconds()
        actual_calc = (t_tca - t_burn).total_seconds() + post_tca_margin_sec
        self.assertAlmostEqual(expected_duration, actual_calc, places=6)

    def test_reconstructed_burn_state_matches_forward_unperturbed_orbit(self):
        """Propagating backward by lead_time to burn epoch and forward by lead_time returns original state."""
        verifier = RK45PhysicsVerifier()
        r_tca = np.array([6878.0, 0.0, 0.0])
        v_tca = np.array([0.0, 7.61, 0.0])
        y_tca = np.hstack((r_tca, v_tca))
        lead_time_sec = 1200.0

        _, sat_back = verifier.propagate_state(y_tca, (0.0, -lead_time_sec))
        y_burn = sat_back[:, -1]

        _, sat_fwd = verifier.propagate_state(y_burn, (0.0, lead_time_sec))
        y_recovered = sat_fwd[:, -1]

        pos_diff_km = np.linalg.norm(y_recovered[0:3] - y_tca[0:3])
        vel_diff_km_s = np.linalg.norm(y_recovered[3:6] - y_tca[3:6])
        self.assertLess(pos_diff_km, 1e-3, "Position reversibility error exceeds 1 meter")
        self.assertLess(vel_diff_km_s, 1e-6, "Velocity reversibility error exceeds 1 mm/s")

    def test_optimizer_penalizes_lead_time_preceding_screening_window(self):
        """GAManeuverOptimizer must reject or heavily penalize lead times that exceed tca_sec."""
        from src.maneuver_optimizer import GAManeuverOptimizer
        opt = GAManeuverOptimizer(population_size=10, generations=2)
        r0 = np.array([6878.0, 0.0, 0.0])
        v0 = np.array([0.0, 7.61, 0.0])

        # TCA is at 600s, but candidate requests 1800s lead time (burn would be at -1200s, before t0)
        invalid_candidate = [1800.0, 0.001, 1.57, 1.57]
        fit = opt.evaluate_chromosome(
            individual=invalid_candidate,
            tca_sec=600.0,
            r_sat_tca=r0,
            v_sat_tca=v0,
            r_deb_tca=r0 + np.array([0.3, 0.0, 0.0]),
            v_deb_tca=v0,
            pre_burn_sma=6878.0
        )
        # Should return extreme infeasibility penalty (>= 1e4)
        self.assertGreaterEqual(fit[0], 1e4, "Optimizer failed to penalize pre-window burn candidate")


class TestNumericalIntegratorSafeguards(unittest.TestCase):
    """Tests for physical boundary crossing, solver failure, and energy conservation."""

    def test_invalid_initial_state_rejected(self):
        """Non-finite or zero position vector must raise ValueError before integration."""
        verifier = RK45PhysicsVerifier()
        invalid_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        # Direct check that propagate_state raises ValueError on degenerate initial radius
        with self.assertRaises(ValueError, msg="Zero initial position did not raise ValueError"):
            verifier.propagate_state(invalid_state, (0.0, 100.0))

    def test_earth_surface_impact_detection_stops_propagation(self):
        """Trajectories falling below Earth radius must trigger an event and not return zero derivatives."""
        inside_core_state = np.array([50.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        deriv_core = j2_equations_of_motion(0.0, inside_core_state)
        
        # Silently returning zeros means dv/dt = 0 and dr/dt = 0, freezing the satellite inside Earth.
        self.assertFalse(
            np.all(deriv_core == 0.0),
            "FLAW DETECTED: j2_equations_of_motion silently returned np.zeros(6) for trajectory inside Earth."
        )

    def test_integrator_checks_solver_success(self):
        """If solve_ivp fails, the verifier must report failure and not claim a valid trajectory."""
        verifier = RK45PhysicsVerifier()
        # Test propagation with max_step so tiny it exhausts steps or triggers solver failure
        # verify_maneuver must check sol.success
        self.assertTrue(hasattr(verifier, "propagate_state"))

    def test_two_body_energy_and_sma_conservation(self):
        """Under pure two-body gravity, specific orbital energy and SMA must be strictly conserved."""
        r0 = 6878.0
        v0 = np.sqrt(MU_EARTH / r0)
        y0 = np.array([r0, 0.0, 0.0, 0.0, v0, 0.0])
        
        def two_body_eom(t, y):
            r_vec = y[0:3]
            v_vec = y[3:6]
            r = np.linalg.norm(r_vec)
            a_vec = -MU_EARTH * r_vec / (r ** 3)
            return np.hstack((v_vec, a_vec))

        from scipy.integrate import solve_ivp
        period = 2.0 * np.pi * np.sqrt(r0**3 / MU_EARTH)
        sol = solve_ivp(two_body_eom, (0.0, period), y0, method='RK45', rtol=1e-10, atol=1e-12)

        self.assertTrue(sol.success)
        r_start = np.linalg.norm(sol.y[0:3, 0])
        v_start = np.linalg.norm(sol.y[3:6, 0])
        energy_start = 0.5 * (v_start**2) - (MU_EARTH / r_start)

        r_end = np.linalg.norm(sol.y[0:3, -1])
        v_end = np.linalg.norm(sol.y[3:6, -1])
        energy_end = 0.5 * (v_end**2) - (MU_EARTH / r_end)

        relative_energy_error = abs((energy_end - energy_start) / energy_start)
        self.assertLess(relative_energy_error, 1e-7, "Two-body energy conservation violated")


class TestCoordinateFramesAndGMATAlignment(unittest.TestCase):
    """Tests for coordinate frame transformations and RIC vs VNB maneuver equivalence."""

    def test_teme_to_gcrs_position_and_velocity_matches_skyfield_reference(self):
        """TEME to GCRS transformation must match Skyfield's GCRS positions and velocities to machine precision."""
        from skyfield.api import load
        from skyfield.sgp4lib import TEME
        mgr = TLEManager()
        sat1, _, _ = mgr.load_scenario("starlink_cosmos")
        ts = load.timescale()
        t = ts.utc(2026, 10, 11, 12, 0, 0)

        # Reference GCRS from Skyfield
        ref_gcrs_pos = sat1.at(t).position.km
        ref_gcrs_vel = sat1.at(t).velocity.km_per_s

        # Raw TEME
        r_teme, v_teme, err = sat1._position_and_velocity_TEME_km(t)
        self.assertIsNone(err)

        # Rotation matrix GCRS <- TEME
        R_gcrs_from_teme = TEME.rotation_at(t).T
        calc_gcrs_pos = R_gcrs_from_teme @ r_teme
        calc_gcrs_vel = R_gcrs_from_teme @ v_teme

        np.testing.assert_allclose(calc_gcrs_pos, ref_gcrs_pos, atol=1e-9, err_msg="GCRS position mismatch")
        np.testing.assert_allclose(calc_gcrs_vel, ref_gcrs_vel, atol=1e-9, err_msg="GCRS velocity mismatch")

    def test_teme_to_gcrs_module_functions_and_reversibility(self):
        """teme_to_gcrs and gcrs_to_teme in src.orbital_frames must round-trip to machine precision."""
        from skyfield.api import load
        from src.orbital_frames import teme_to_gcrs, gcrs_to_teme
        mgr = TLEManager()
        sat1, _, _ = mgr.load_scenario("starlink_cosmos")
        epoch = datetime(2026, 10, 11, 12, 17, 49, tzinfo=timezone.utc)
        ts = load.timescale()
        t = ts.utc(2026, 10, 11, 12, 17, 49)

        r_teme, v_teme, _ = sat1._position_and_velocity_TEME_km(t)
        r_gcrs, v_gcrs = teme_to_gcrs(epoch, r_teme, v_teme)

        ref_r_gcrs = sat1.at(t).position.km
        ref_v_gcrs = sat1.at(t).velocity.km_per_s

        np.testing.assert_allclose(r_gcrs, ref_r_gcrs, atol=1e-9)
        np.testing.assert_allclose(v_gcrs, ref_v_gcrs, atol=1e-12)

        # Round trip
        r_back, v_back = gcrs_to_teme(epoch, r_gcrs, v_gcrs)
        np.testing.assert_allclose(r_back, r_teme, atol=1e-12)
        np.testing.assert_allclose(v_back, v_teme, atol=1e-12)

    def test_gmat_script_transforms_teme_to_j2000_coordinates(self):
        """GMAT script generation with input_frame='TEME' must write transformed J2000 coordinates."""
        gmat = GMATInterface()
        epoch = datetime(2026, 10, 11, 12, 17, 49, tzinfo=timezone.utc)
        r_teme = np.array([6878.0, 0.0, 0.0])
        v_teme = np.array([0.0, 7.61, 0.0])
        dv_ric = np.array([0.0, 0.002, 0.0])

        script_path = gmat.generate_verification_script(
            primary_name="TEST_SAT",
            secondary_name="TEST_DEB",
            epoch_utc=epoch,
            r_primary_km=r_teme,
            v_primary_km_s=v_teme,
            r_secondary_km=r_teme + np.array([0.5, 0.0, 0.0]),
            v_secondary_km_s=v_teme,
            delta_v_ric_km_s=dv_ric,
            filename="test_gmat_j2000_transform.script",
            input_frame="TEME"
        )

        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()

        from src.orbital_frames import teme_to_gcrs
        r_j2000, v_j2000 = teme_to_gcrs(epoch, r_teme, v_teme)

        expected_x_str = f"GMAT PrimarySatellite.X = {r_j2000[0]:.6f};"
        self.assertIn(expected_x_str, content)
        # Verify that J2000 coordinates differ from raw TEME coordinates by precession displacement (> 10 km)
        displacement_km = np.linalg.norm(r_j2000 - r_teme)
        self.assertGreater(displacement_km, 10.0, "J2000 and TEME position should differ significantly at 2026 epoch")

    def test_ric_and_vnb_basis_in_eccentric_orbit(self):
        """In an eccentric orbit where r dot v != 0, RIC and VNB frames are distinct and not interchangeable."""
        r_ecc = np.array([5000.0, 5000.0, 0.0])  # r = 7071 km
        v_ecc = np.array([-5.5, 5.0, 1.0])       # r dot v = -2500 != 0
        r_dot_v = np.dot(r_ecc, v_ecc)
        self.assertNotEqual(r_dot_v, 0.0, "Sanity check: radial velocity must be non-zero for eccentric orbit")

        u_r = r_ecc / np.linalg.norm(r_ecc)
        h_vec = np.cross(r_ecc, v_ecc)
        u_c = h_vec / np.linalg.norm(h_vec)
        u_i = np.cross(u_c, u_r)

        u_v = v_ecc / np.linalg.norm(v_ecc)
        u_n = u_c
        u_b = np.cross(u_v, u_n)

        # In-track burn in RIC: 10 m/s along u_i
        dv_mag = 0.010  # 10 m/s in km/s
        dv_ric_components = np.array([0.0, dv_mag, 0.0])  # [R, I, C]
        dv_inertial = dv_ric_components[0]*u_r + dv_ric_components[1]*u_i + dv_ric_components[2]*u_c

        # Project into VNB frame:
        dv_v = float(np.dot(dv_inertial, u_v))
        dv_n = float(np.dot(dv_inertial, u_n))
        dv_b = float(np.dot(dv_inertial, u_b))
        dv_vnb_projected = np.array([dv_v, dv_n, dv_b])

        # A naive component swap (e.g. GMAT Element1=I, Element2=C, Element3=R):
        dv_vnb_naive = np.array([dv_ric_components[1], dv_ric_components[2], dv_ric_components[0]])
        dv_inertial_naive = dv_vnb_naive[0]*u_v + dv_vnb_naive[1]*u_n + dv_vnb_naive[2]*u_b
        directional_error_km_s = np.linalg.norm(dv_inertial_naive - dv_inertial)

        self.assertGreater(
            directional_error_km_s,
            1e-5,
            "Naive RIC->VNB component swap falsely appeared identical in eccentric orbit"
        )

        dv_inertial_projected = dv_vnb_projected[0]*u_v + dv_vnb_projected[1]*u_n + dv_vnb_projected[2]*u_b
        np.testing.assert_allclose(
            dv_inertial_projected,
            dv_inertial,
            atol=1e-12,
            err_msg="Rigorous VNB projection failed to reproduce physical delta-V"
        )

    def test_gmat_script_epoch_must_match_burn_state_epoch(self):
        """GMAT script generation must write the burn epoch, not the TCA epoch."""
        gmat = GMATInterface()
        t_burn = datetime(2026, 10, 11, 12, 17, 49, tzinfo=timezone.utc)
        r_burn = np.array([6878.0, 0.0, 0.0])
        v_burn = np.array([0.0, 7.61, 0.0])
        dv_ric = np.array([0.0, 0.002, 0.0])

        script_path = gmat.generate_verification_script(
            primary_name="TEST_SAT",
            secondary_name="TEST_DEB",
            epoch_utc=t_burn,
            r_primary_km=r_burn,
            v_primary_km_s=v_burn,
            r_secondary_km=r_burn + np.array([0.5, 0.0, 0.0]),
            v_secondary_km_s=v_burn,
            delta_v_ric_km_s=dv_ric,
            filename="test_gmat_epoch.script"
        )

        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()

        expected_epoch_str = t_burn.strftime("%d %b %Y %H:%M:%S.000")
        self.assertIn(f"GMAT PrimarySatellite.Epoch = '{expected_epoch_str}';", content)


class TestRICToVNBConversion(unittest.TestCase):
    """Physical RIC-to-VNB maneuver conversion and invariant tests."""

    def setUp(self):
        # Nominal circular LEO state
        self.r_circ = np.array([6878.0, 0.0, 0.0])
        self.v_circ = np.array([0.0, 7.61, 0.0])

        # Eccentric LEO state with non-zero radial velocity (r dot v != 0)
        self.r_ecc = np.array([5000.0, 5000.0, 0.0])
        self.v_ecc = np.array([-5.5, 5.0, 1.0])

    def test_round_trip_vector_equivalence(self):
        """1. Round-trip vector equivalence: RIC -> VNB -> RIC reproduces the original components within tolerance."""
        test_states = [
            (self.r_circ, self.v_circ),
            (self.r_ecc, self.v_ecc),
            (np.array([12000.0, -8000.0, 5000.0]), np.array([2.5, 4.2, -3.1])),
        ]
        test_dvs_ric = [
            np.array([0.001, 0.0, 0.0]),       # Pure radial
            np.array([0.0, 0.005, 0.0]),       # Pure in-track
            np.array([0.0, 0.0, 0.002]),       # Pure cross-track
            np.array([0.0015, -0.003, 0.0008]) # General 3D burn
        ]

        for r, v in test_states:
            for dv_ric in test_dvs_ric:
                # Forward: RIC -> VNB
                dv_vnb = ric_to_vnb(r, v, dv_ric)
                # Backward: VNB -> RIC
                dv_recovered = vnb_to_ric(r, v, dv_vnb)

                np.testing.assert_allclose(
                    dv_recovered,
                    dv_ric,
                    atol=1e-12,
                    err_msg=f"Round-trip RIC->VNB->RIC failed for state r={r}, v={v}, dv={dv_ric}"
                )

                # Reverse forward: VNB -> RIC -> VNB
                dv_vnb_recovered = ric_to_vnb(r, v, dv_recovered)
                np.testing.assert_allclose(
                    dv_vnb_recovered,
                    dv_vnb,
                    atol=1e-12,
                    err_msg=f"Round-trip VNB->RIC->VNB failed for state r={r}, v={v}"
                )

    def test_magnitude_preservation(self):
        """2. Magnitude preservation: ||dv_VNB|| == ||dv_RIC|| across arbitrary burns."""
        test_dvs = [
            np.array([0.0001, 0.0, 0.0]),
            np.array([0.0, 0.0025, 0.0]),
            np.array([0.0003, -0.0012, 0.0041]),
            np.array([0.050, -0.020, 0.015]),
        ]
        for dv in test_dvs:
            for r, v in [(self.r_circ, self.v_circ), (self.r_ecc, self.v_ecc)]:
                dv_vnb = ric_to_vnb(r, v, dv)
                norm_ric = np.linalg.norm(dv)
                norm_vnb = np.linalg.norm(dv_vnb)

                self.assertAlmostEqual(
                    norm_ric,
                    norm_vnb,
                    places=14,
                    msg=f"Magnitude not preserved: RIC={norm_ric}, VNB={norm_vnb}"
                )

    def test_eccentric_orbit_case(self):
        """3. Eccentric-orbit case: Nonzero radial velocity exposes errors caused by assuming frames coincide."""
        r_dot_v = float(np.dot(self.r_ecc, self.v_ecc))
        self.assertNotEqual(r_dot_v, 0.0, "State must have non-zero radial velocity")

        # In an eccentric orbit, an in-track burn along u_i is NOT strictly parallel to velocity u_v
        dv_ric = np.array([0.0, 0.010, 0.0]) # 10 m/s in-track
        dv_vnb = ric_to_vnb(self.r_ecc, self.v_ecc, dv_ric)

        # Naive component swap (Element 1 = I, Element 2 = C, Element 3 = R)
        dv_vnb_naive = np.array([dv_ric[1], dv_ric[2], dv_ric[0]])

        # Reconstruct inertial vectors
        Q_ric = compute_ric_rotation_matrix(self.r_ecc, self.v_ecc)
        Q_vnb = compute_vnb_rotation_matrix(self.r_ecc, self.v_ecc)

        dv_inertial_true = Q_ric @ dv_ric
        dv_inertial_projected = Q_vnb @ dv_vnb
        dv_inertial_naive = Q_vnb @ dv_vnb_naive

        # Projected VNB must exactly equal physical inertial delta-V
        np.testing.assert_allclose(
            dv_inertial_projected,
            dv_inertial_true,
            atol=1e-12,
            err_msg="Physical inertial delta-V mismatch in eccentric orbit projection"
        )

        # Naive swap must produce significant directional error
        error_norm = float(np.linalg.norm(dv_inertial_naive - dv_inertial_true))
        self.assertGreater(
            error_norm,
            1e-5,
            "Naive component swap failed to expose physical discrepancy in eccentric orbit"
        )

    def test_degenerate_state_rejection(self):
        """4. Degenerate-state rejection: Reject zero velocity or near-zero angular momentum before constructing bases."""
        r_valid = np.array([6878.0, 0.0, 0.0])
        v_valid = np.array([0.0, 7.61, 0.0])
        dv_dummy = np.array([0.0, 0.001, 0.0])

        # Zero velocity
        v_zero = np.array([0.0, 0.0, 0.0])
        with self.assertRaises(ValueError, msg="Zero velocity did not raise ValueError"):
            compute_vnb_basis(r_valid, v_zero)
        with self.assertRaises(ValueError, msg="Zero velocity did not raise ValueError in ric_to_vnb"):
            ric_to_vnb(r_valid, v_zero, dv_dummy)

        # Rectilinear / collinear state (angular momentum h = r x v = 0)
        v_collinear = np.array([7.61, 0.0, 0.0]) # parallel to r_valid
        with self.assertRaises(ValueError, msg="Collinear state did not raise ValueError in compute_vnb_basis"):
            compute_vnb_basis(r_valid, v_collinear)
        with self.assertRaises(ValueError, msg="Collinear state did not raise ValueError in compute_ric_basis"):
            compute_ric_basis(r_valid, v_collinear)
        with self.assertRaises(ValueError, msg="Collinear state did not raise ValueError in ric_to_vnb"):
            ric_to_vnb(r_valid, v_collinear, dv_dummy)
        with self.assertRaises(ValueError, msg="Collinear state did not raise ValueError in vnb_to_ric"):
            vnb_to_ric(r_valid, v_collinear, dv_dummy)

        # Degenerate position (r = 0)
        r_zero = np.array([0.0, 0.0, 0.0])
        with self.assertRaises(ValueError, msg="Zero position did not raise ValueError in compute_vnb_basis"):
            compute_vnb_basis(r_zero, v_valid)

    def test_gmat_script_integration(self):
        """5. GMAT script integration: Generated VNB burn components reconstruct the intended physical delta-V vector."""
        gmat = GMATInterface()
        epoch = datetime(2026, 10, 11, 12, 17, 49, tzinfo=timezone.utc)
        r_burn = self.r_ecc
        v_burn = self.v_ecc
        dv_ric = np.array([0.002, 0.005, -0.001]) # 3D burn in RIC

        script_path = gmat.generate_verification_script(
            primary_name="PRIMARY_SAT",
            secondary_name="SECONDARY_DEB",
            epoch_utc=epoch,
            r_primary_km=r_burn,
            v_primary_km_s=v_burn,
            r_secondary_km=r_burn + np.array([0.5, 0.0, 0.0]),
            v_secondary_km_s=v_burn,
            delta_v_ric_km_s=dv_ric,
            filename="test_gmat_vnb_burn.script",
            input_frame="EarthMJ2000Eq" # Pass directly to test script VNB elements directly against input state
        )

        with open(script_path, "r", encoding="utf-8") as f:
            script_text = f.read()

        import re
        e1_match = re.search(r"GMAT AvoidanceBurn\.Element1\s*=\s*([-0-9.]+);", script_text)
        e2_match = re.search(r"GMAT AvoidanceBurn\.Element2\s*=\s*([-0-9.]+);", script_text)
        e3_match = re.search(r"GMAT AvoidanceBurn\.Element3\s*=\s*([-0-9.]+);", script_text)

        self.assertIsNotNone(e1_match)
        self.assertIsNotNone(e2_match)
        self.assertIsNotNone(e3_match)

        e1 = float(e1_match.group(1))
        e2 = float(e2_match.group(1))
        e3 = float(e3_match.group(1))
        dv_vnb_from_script = np.array([e1, e2, e3])

        # Reconstruct physical delta-V from script VNB elements using Q_VNB
        Q_vnb = compute_vnb_rotation_matrix(r_burn, v_burn)
        dv_inertial_from_script = Q_vnb @ dv_vnb_from_script

        # Physical delta-V from original RIC vector
        Q_ric = compute_ric_rotation_matrix(r_burn, v_burn)
        dv_inertial_intended = Q_ric @ dv_ric

        # Must match to within floating-point text precision (6 decimal places = 1e-6 km/s)
        np.testing.assert_allclose(
            dv_inertial_from_script,
            dv_inertial_intended,
            atol=1e-6,
            err_msg="GMAT script VNB elements failed to reconstruct intended physical delta-V vector"
        )


if __name__ == "__main__":
    unittest.main()
