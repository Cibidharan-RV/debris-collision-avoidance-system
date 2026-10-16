"""End-to-end execution of conjunction screening, avoidance maneuver, and verification."""

import argparse
from datetime import datetime, timezone
import os
import sys
import time
import webbrowser
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.tle_manager import TLEManager, BENCHMARK_SCENARIOS
from src.conjunction_engine import ConjunctionEngine
from src.orbital_frames import (
    teme_to_gcrs,
    delta_v_to_eci,
    spherical_to_ric_vector,
    compute_semi_major_axis
)
from src.physics_verifier import RK45PhysicsVerifier
from src.gmat_interface import GMATInterface
from src.visualizer import OrbitVisualizer
from src.constants import MU_EARTH, SAFETY_DISTANCE_KM


def print_section(title: str):
    print(f"\n[{title}]")
    print("-" * 55)


def run_pipeline_demo(scenario_key: str = "starlink_cosmos", open_browser: bool = False):
    print("\n" + "=" * 60)
    print(" Space Debris Collision Avoidance System")
    print("=" * 60)
    print(f"Run time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")

    mgr = TLEManager()
    sat1, sat2, meta = mgr.load_scenario(scenario_key)

    print(f"\nScenario: {meta['name']}")
    print(f"  Primary Satellite : {sat1.name} (NORAD {sat1.model.satnum})")
    print(f"  Debris Hazard     : {sat2.name} (NORAD {sat2.model.satnum})")

    print_section("Conjunction Screening")
    t_start = time.time()
    engine = ConjunctionEngine(coarse_step_sec=20.0, safety_distance_km=SAFETY_DISTANCE_KM)
    event = engine.analyze_pair(sat1, sat2, duration_minutes=90.0)
    t_screen = time.time() - t_start

    print(f"Screening completed in {t_screen:.3f}s:")
    print(f"  Time of Closest Approach (TCA) : {event.tca_epoch.strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(f"  Miss Distance                  : {event.miss_distance_km:.3f} km ({event.miss_distance_km * 1000.0:.1f} m)")
    print(f"  Safety Threshold               : {event.risk_threshold_km:.1f} km")
    print(f"  Relative Velocity              : {event.relative_velocity_km_s * 1000.0:.2f} m/s ({event.relative_velocity_km_s:.4f} km/s)")
    print(f"  Heuristic Risk Proxy           : {event.heuristic_risk_proxy:.2e}")
    print("  Note: Collision risk indicator is a 2D isotropic proxy, not an operational covariance-based Pc.")

    if not event.is_risk:
        print(f"\nSafe distance (>= {SAFETY_DISTANCE_KM} km). No maneuver required.")
        return

    print_section("Avoidance Maneuver Planning")
    lead_time_min = 30.0
    lead_time_sec = lead_time_min * 60.0
    t_burn_sec = max(0.0, event.tca_seconds_from_t0 - lead_time_sec)

    r_gcrs_tca, v_gcrs_tca = teme_to_gcrs(event.tca_epoch, event.r_primary_tca, event.v_primary_tca)
    r_deb_gcrs_tca, v_deb_gcrs_tca = teme_to_gcrs(event.tca_epoch, event.r_secondary_tca, event.v_secondary_tca)

    verifier = RK45PhysicsVerifier(safety_threshold_km=SAFETY_DISTANCE_KM)
    y_sat_tca = np.hstack((r_gcrs_tca, v_gcrs_tca))
    y_deb_tca = np.hstack((r_deb_gcrs_tca, v_deb_gcrs_tca))

    _, sat_back = verifier.propagate_state(y_sat_tca, (0.0, -lead_time_sec))
    _, deb_back = verifier.propagate_state(y_deb_tca, (0.0, -lead_time_sec))
    r_sat_burn = sat_back[0:3, -1]
    v_sat_burn = sat_back[3:6, -1]
    r_deb_burn = deb_back[0:3, -1]
    v_deb_burn = deb_back[3:6, -1]

    dv_mag_m_s = 2.50
    dv_mag_km_s = dv_mag_m_s / 1000.0
    theta_rad = np.radians(90.0)
    phi_rad = np.radians(90.0)

    dv_ric = spherical_to_ric_vector(dv_mag_km_s, theta_rad, phi_rad)
    dv_eci = delta_v_to_eci(r_sat_burn, v_sat_burn, dv_mag_km_s, theta_rad, phi_rad)

    v_sat_post_burn = v_sat_burn + dv_eci
    post_sma = compute_semi_major_axis(r_sat_burn, v_sat_post_burn, MU_EARTH)
    orbit_dev_km = abs(post_sma - event.pre_burn_sma_km)

    print("Maneuver Parameters:")
    print(f"  Reference Frame        : GCRS / EarthMJ2000Eq")
    print(f"  Burn Timing            : {lead_time_min:.1f} min before TCA (t_burn = {t_burn_sec:.1f}s)")
    print(f"  Delta-V Magnitude      : {dv_mag_m_s:.2f} m/s ({dv_mag_km_s:.6f} km/s)")
    print(f"  RIC Vector             : R={dv_ric[0]*1000:+.2f} m/s, I={dv_ric[1]*1000:+.2f} m/s, C={dv_ric[2]*1000:+.2f} m/s")
    print(f"  Orbit Deviation (|da|) : {orbit_dev_km:.3f} km")

    print_section("Numerical Verification (RK45 + J2)")
    t_v_start = time.time()
    ver_res = verifier.verify_maneuver(
        r_sat_burn,
        v_sat_burn,
        r_deb_burn,
        v_deb_burn,
        dv_eci,
        dv_ric,
        t_burn_sec,
        event.tca_seconds_from_t0,
        event.miss_distance_km,
        event.pre_burn_sma_km,
        nominal_tca_epoch=event.tca_epoch
    )
    t_verify = time.time() - t_v_start

    print(f"Verification evaluated in {t_verify:.3f}s:")
    print(ver_res.summary())

    print_section("GMAT Mission Script Generation")
    gmat = GMATInterface(output_dir="output")
    gmat_script_path = gmat.generate_verification_script(
        event.primary_name,
        event.secondary_name,
        event.tca_epoch,
        r_sat_burn,
        v_sat_burn,
        r_deb_burn,
        v_deb_burn,
        dv_ric,
        propagation_duration_sec=lead_time_sec + 600.0
    )
    print(f"  GMAT script saved to : {gmat_script_path}")
    print("  Integrator model     : PrinceDormand78 (RK89) + JGM-2 4x4 Gravity")

    print_section("3D Orbit Visualization")
    vis = OrbitVisualizer(output_dir="output")
    html_path = vis.generate_3d_encounter_scene(
        event=event,
        sat_traj=sat_back[0:3, :],
        deb_traj=deb_back[0:3, :],
        post_burn_sat_traj=ver_res.sat_trajectory_history[0:3, :],
        burn_point=r_sat_burn,
        delta_v_eci=dv_eci,
        filename="encounter_3d_visualization.html"
    )
    print(f"  Visualization HTML saved to: {html_path}")

    print("\n" + "=" * 60)
    print(" Final Summary")
    print("=" * 60)
    print(f"  Primary Satellite       : {event.primary_name}")
    print(f"  Debris Object           : {event.secondary_name}")
    print(f"  Initial Miss Distance   : {event.miss_distance_km:.3f} km")
    print(f"  Avoidance Delta-V       : {dv_mag_m_s:.2f} m/s (along-track)")
    print(f"  Post-Burn Miss Distance : {ver_res.post_burn_miss_distance_km:.3f} km")
    print(f"  Verification Status     : {ver_res.status} ({'FEASIBLE_UNDER_MODEL' if ver_res.passed else 'INFEASIBLE'})")
    print(f"  NASA GMAT Script        : {gmat_script_path}")
    print(f"  3D Visualization        : {html_path}")
    print("=" * 60 + "\n")

    if open_browser:
        webbrowser.open(os.path.abspath(html_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collision avoidance demo")
    parser.add_argument(
        "--scenario",
        choices=list(BENCHMARK_SCENARIOS.keys()),
        default="starlink_cosmos",
        help="Conjunction scenario"
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open 3D visualization in browser"
    )
    args = parser.parse_args()
    run_pipeline_demo(args.scenario, open_browser=args.open)
