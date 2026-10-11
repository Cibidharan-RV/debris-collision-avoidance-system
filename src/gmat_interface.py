"""Generates NASA GMAT mission scripts for independent trajectory verification."""

import os
import subprocess
from datetime import datetime
import numpy as np

class GMATInterface:
    """Manages GMAT script generation and batch execution."""

    def __init__(self, output_dir: str = "output", gmat_bin_path: str | None = None):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.gmat_bin_path = gmat_bin_path or self._find_gmat_executable()

    def _find_gmat_executable(self) -> str | None:
        """Looks for GMAT.exe in common installation paths."""
        common_paths = [
            r"C:\Program Files\GMAT\bin\GMAT.exe",
            r"C:\GMAT\bin\GMAT.exe",
            r"D:\GMAT\bin\GMAT.exe",
            os.path.expanduser(r"~\AppData\Local\GMAT\bin\GMAT.exe")
        ]
        for p in common_paths:
            if os.path.exists(p):
                return p
        return None

    def generate_verification_script(
        self,
        primary_name: str,
        secondary_name: str,
        epoch_utc: datetime,
        r_primary_km: np.ndarray,
        v_primary_km_s: np.ndarray,
        r_secondary_km: np.ndarray,
        v_secondary_km_s: np.ndarray,
        delta_v_ric_km_s: np.ndarray,
        propagation_duration_sec: float = 3600.0,
        filename: str = "gmat_conjunction_verification.script"
    ) -> str:
        """Generates a GMAT script for post-burn trajectory propagation."""
        epoch_str = epoch_utc.strftime("%d %b %Y %H:%M:%S.000")
        output_filepath = os.path.join(self.output_dir, filename)

        # Map RIC to GMAT VNB axes: V = in-track, N = cross-track, B = radial
        dv_radial = float(delta_v_ric_km_s[0])
        dv_intrack = float(delta_v_ric_km_s[1])
        dv_crosstrack = float(delta_v_ric_km_s[2])

        script_content = f"""%--------------------------------------------------------------------------
% NASA GMAT Verification Script - Post-Maneuver Orbit Propagation
% Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
%--------------------------------------------------------------------------

%----------------------------------------
%---------- Spacecraft Definitions
%----------------------------------------
Create Spacecraft PrimarySatellite;
GMAT PrimarySatellite.DateFormat = UTCGregorian;
GMAT PrimarySatellite.Epoch = '{epoch_str}';
GMAT PrimarySatellite.CoordinateSystem = EarthMJ2000Eq;
GMAT PrimarySatellite.DisplayStateType = Cartesian;
GMAT PrimarySatellite.X = {r_primary_km[0]:.6f};
GMAT PrimarySatellite.Y = {r_primary_km[1]:.6f};
GMAT PrimarySatellite.Z = {r_primary_km[2]:.6f};
GMAT PrimarySatellite.VX = {v_primary_km_s[0]:.6f};
GMAT PrimarySatellite.VY = {v_primary_km_s[1]:.6f};
GMAT PrimarySatellite.VZ = {v_primary_km_s[2]:.6f};
GMAT PrimarySatellite.DryMass = 260; % Standard LEO SmallSat mass [kg]
GMAT PrimarySatellite.Cd = 2.2;
GMAT PrimarySatellite.DragArea = 2.0;

Create Spacecraft SpaceDebris;
GMAT SpaceDebris.DateFormat = UTCGregorian;
GMAT SpaceDebris.Epoch = '{epoch_str}';
GMAT SpaceDebris.CoordinateSystem = EarthMJ2000Eq;
GMAT SpaceDebris.DisplayStateType = Cartesian;
GMAT SpaceDebris.X = {r_secondary_km[0]:.6f};
GMAT SpaceDebris.Y = {r_secondary_km[1]:.6f};
GMAT SpaceDebris.Z = {r_secondary_km[2]:.6f};
GMAT SpaceDebris.VX = {v_secondary_km_s[0]:.6f};
GMAT SpaceDebris.VY = {v_secondary_km_s[1]:.6f};
GMAT SpaceDebris.VZ = {v_secondary_km_s[2]:.6f};
GMAT SpaceDebris.DryMass = 50; % Debris fragment mass [kg]

%----------------------------------------
%---------- Force Models & Integrator
%----------------------------------------
Create ForceModel LEOForceModel;
GMAT LEOForceModel.CentralBody = Earth;
GMAT LEOForceModel.PrimaryBodies = {{Earth}};
GMAT LEOForceModel.PointMasses = {{Sun, Luna}};
GMAT LEOForceModel.GravityField.Earth.Degree = 4;
GMAT LEOForceModel.GravityField.Earth.Order = 4;
GMAT LEOForceModel.GravityField.Earth.PotentialFile = 'JGM2.cof';
GMAT LEOForceModel.Drag.AtmosphereModel = JacchiaRoberts;

Create Propagator PrinceDormand78;
GMAT PrinceDormand78.FM = LEOForceModel;
GMAT PrinceDormand78.Type = RungeKutta89;
GMAT PrinceDormand78.InitialStepSize = 10;
GMAT PrinceDormand78.Accuracy = 1e-11;
GMAT PrinceDormand78.MinStep = 0.001;
GMAT PrinceDormand78.MaxStep = 60;

%----------------------------------------
%---------- Impulsive Burn (RIC Frame)
%----------------------------------------
Create ImpulsiveBurn AvoidanceBurn;
GMAT AvoidanceBurn.CoordinateSystem = Local;
GMAT AvoidanceBurn.Origin = Earth;
GMAT AvoidanceBurn.Axes = VNB;
GMAT AvoidanceBurn.Element1 = {dv_intrack:.6f};    % V: In-track [km/s]
GMAT AvoidanceBurn.Element2 = {dv_crosstrack:.6f}; % N: Cross-track [km/s]
GMAT AvoidanceBurn.Element3 = {dv_radial:.6f};     % B: Radial [km/s]

%----------------------------------------
%---------- Output Report
%----------------------------------------
Create ReportFile ConjunctionReport;
GMAT ConjunctionReport.Filename = 'conjunction_verification_report.txt';
GMAT ConjunctionReport.Precision = 16;
GMAT ConjunctionReport.Add = {{PrimarySatellite.UTCGregorian, PrimarySatellite.X, PrimarySatellite.Y, PrimarySatellite.Z, SpaceDebris.X, SpaceDebris.Y, SpaceDebris.Z}};

%----------------------------------------
%---------- Mission Sequence
%----------------------------------------
BeginMissionSequence;
% Apply impulsive collision avoidance maneuver to primary satellite
Maneuver AvoidanceBurn(PrimarySatellite);

% Numerically propagate both primary and debris through encounter window
Propagate PrinceDormand78(PrimarySatellite, SpaceDebris) {{PrimarySatellite.ElapsedSecs = {propagation_duration_sec:.1f}}};
"""
        with open(output_filepath, "w", encoding="utf-8") as f:
            f.write(script_content)

        return output_filepath

    def is_gmat_available(self) -> bool:
        """Returns True if NASA GMAT executable is available on the system."""
        return self.gmat_bin_path is not None and os.path.exists(self.gmat_bin_path)

    def execute_gmat_script(self, script_path: str) -> bool:
        """Executes a GMAT script in headless batch mode if GMAT is installed."""
        if not self.is_gmat_available():
            return False
        cmd = [self.gmat_bin_path, "-r", "-m", script_path]
        try:
            res = subprocess.run(cmd, capture_output=True, timeout=60)
            return res.returncode == 0
        except Exception:
            return False

if __name__ == "__main__":
    gmat = GMATInterface()
    sample_r = np.array([6878.0, 0.0, 0.0])
    sample_v = np.array([0.0, 7.61, 0.0])
    sample_dv = np.array([0.0001, 0.0005, 0.0])
    path = gmat.generate_verification_script(
        "STARLINK-1008",
        "COSMOS-1408 DEBRIS",
        datetime(2026, 10, 11, 12, 0, 0),
        sample_r,
        sample_v,
        sample_r + np.array([0.3, 0.2, 0.1]),
        sample_v + np.array([0.001, 0.0, 0.0]),
        sample_dv
    )
    print("NASA GMAT Script generated successfully at:", path)
    print("GMAT executable detected:", gmat.is_gmat_available())
