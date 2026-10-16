"""Coordinate frames and transformations (TEME, ECI, and local RIC)."""

import numpy as np
from datetime import datetime, timezone
from skyfield.api import load
from skyfield.sgp4lib import TEME

from src.constants import MU_EARTH

def teme_to_gcrs_matrix(epoch_dt: datetime) -> np.ndarray:
    """Computes the 3x3 rotation matrix from TEME to GCRS (EarthMJ2000Eq / J2000 ECI) at epoch_dt."""
    if epoch_dt.tzinfo is None:
        epoch_dt = epoch_dt.replace(tzinfo=timezone.utc)
    ts = load.timescale()
    sec = epoch_dt.second + (epoch_dt.microsecond / 1e6)
    t = ts.utc(epoch_dt.year, epoch_dt.month, epoch_dt.day, epoch_dt.hour, epoch_dt.minute, sec)
    # TEME.rotation_at(t) gives TEME <- GCRS; transpose gives GCRS <- TEME
    return TEME.rotation_at(t).T

def teme_to_gcrs(
    epoch_dt: datetime,
    r_teme: np.ndarray,
    v_teme: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Transforms position and optional velocity vectors from TEME to GCRS (EarthMJ2000Eq)."""
    R = teme_to_gcrs_matrix(epoch_dt)
    r_gcrs = R @ np.asarray(r_teme, dtype=float)
    if v_teme is not None:
        v_gcrs = R @ np.asarray(v_teme, dtype=float)
        return r_gcrs, v_gcrs
    return r_gcrs

def gcrs_to_teme(
    epoch_dt: datetime,
    r_gcrs: np.ndarray,
    v_gcrs: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
    """Transforms position and optional velocity vectors from GCRS (EarthMJ2000Eq) to TEME."""
    R = teme_to_gcrs_matrix(epoch_dt).T
    r_teme = R @ np.asarray(r_gcrs, dtype=float)
    if v_gcrs is not None:
        v_teme = R @ np.asarray(v_gcrs, dtype=float)
        return r_teme, v_teme
    return r_teme

def compute_ric_basis(r_vec: np.ndarray, v_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes orthonormal basis vectors (u_r, u_i, u_c) for the local RIC frame."""
    r_vec = np.asarray(r_vec, dtype=float)
    v_vec = np.asarray(v_vec, dtype=float)
    r_norm = np.linalg.norm(r_vec)
    if r_norm < 1.0:
        raise ValueError("Position vector norm too small for RIC computation.")
    u_r = r_vec / r_norm
    
    v_norm = np.linalg.norm(v_vec)
    if v_norm < 1e-6:
        raise ValueError("Velocity vector norm too small for RIC computation.")

    h_vec = np.cross(r_vec, v_vec)
    h_norm = np.linalg.norm(h_vec)
    if h_norm < 1e-9:
        raise ValueError("Angular momentum too small (degenerate or rectilinear orbit).")
    u_c = h_vec / h_norm
    
    u_i = np.cross(u_c, u_r)
    u_i = u_i / np.linalg.norm(u_i)
    return u_r, u_i, u_c

def compute_ric_rotation_matrix(r_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
    """Returns 3x3 rotation matrix Q_RIC = [u_r, u_i, u_c] from local RIC to inertial frame."""
    u_r, u_i, u_c = compute_ric_basis(r_vec, v_vec)
    return np.column_stack((u_r, u_i, u_c))

def compute_vnb_basis(r_vec: np.ndarray, v_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes orthonormal basis vectors (u_v, u_n, u_b) for the NASA GMAT VNB frame.
    
    NASA GMAT VNB Convention:
      - V (Element 1): Unit vector along instantaneous spacecraft velocity v.
      - N (Element 2): Unit vector along orbit normal h = r x v.
      - B (Element 3): Binormal unit vector completing the right-handed triad, B = V x N.
    """
    r_vec = np.asarray(r_vec, dtype=float)
    v_vec = np.asarray(v_vec, dtype=float)
    r_norm = np.linalg.norm(r_vec)
    if r_norm < 1.0:
        raise ValueError("Position vector norm too small for VNB computation.")
    
    v_norm = np.linalg.norm(v_vec)
    if v_norm < 1e-6:
        raise ValueError("Velocity vector norm too small for VNB computation.")
    
    h_vec = np.cross(r_vec, v_vec)
    h_norm = np.linalg.norm(h_vec)
    if h_norm < 1e-9:
        raise ValueError("Angular momentum too small (degenerate or rectilinear orbit).")
    
    u_v = v_vec / v_norm
    u_n = h_vec / h_norm
    u_b = np.cross(u_v, u_n)
    u_b = u_b / np.linalg.norm(u_b)
    return u_v, u_n, u_b

def compute_vnb_rotation_matrix(r_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
    """Returns 3x3 rotation matrix Q_VNB = [u_v, u_n, u_b] whose columns are VNB basis vectors."""
    u_v, u_n, u_b = compute_vnb_basis(r_vec, v_vec)
    return np.column_stack((u_v, u_n, u_b))

def ric_to_vnb(
    r_vec: np.ndarray,
    v_vec: np.ndarray,
    dv_ric: np.ndarray
) -> np.ndarray:
    """Converts a delta-V vector from local RIC frame to GMAT VNB frame via physical vector projection.
    
    Constructs the physical inertial delta-V vector dv = Q_RIC @ dv_RIC and projects
    onto the GMAT VNB basis: dv_VNB = Q_VNB.T @ dv.
    """
    Q_ric = compute_ric_rotation_matrix(r_vec, v_vec)
    Q_vnb = compute_vnb_rotation_matrix(r_vec, v_vec)
    dv_inertial = Q_ric @ np.asarray(dv_ric, dtype=float)
    return Q_vnb.T @ dv_inertial

def vnb_to_ric(
    r_vec: np.ndarray,
    v_vec: np.ndarray,
    dv_vnb: np.ndarray
) -> np.ndarray:
    """Converts a delta-V vector from GMAT VNB frame to local RIC frame via physical vector projection.
    
    Constructs the physical inertial delta-V vector dv = Q_VNB @ dv_VNB and projects
    onto the local RIC basis: dv_RIC = Q_RIC.T @ dv.
    """
    Q_ric = compute_ric_rotation_matrix(r_vec, v_vec)
    Q_vnb = compute_vnb_rotation_matrix(r_vec, v_vec)
    dv_inertial = Q_vnb @ np.asarray(dv_vnb, dtype=float)
    return Q_ric.T @ dv_inertial

def teme_to_ric_matrix(r_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
    """Returns 3x3 rotation matrix from inertial TEME to local RIC."""
    return compute_ric_rotation_matrix(r_vec, v_vec).T

def teme_to_ric(
    r_sat: np.ndarray,
    v_sat: np.ndarray,
    vec_teme: np.ndarray
) -> np.ndarray:
    """Transforms a vector from inertial TEME into the satellite's local RIC frame."""
    return teme_to_ric_matrix(r_sat, v_sat) @ np.asarray(vec_teme, dtype=float)

def ric_to_teme(
    r_sat: np.ndarray,
    v_sat: np.ndarray,
    vec_ric: np.ndarray
) -> np.ndarray:
    """Transforms a vector from local RIC into inertial TEME."""
    return compute_ric_rotation_matrix(r_sat, v_sat) @ np.asarray(vec_ric, dtype=float)

def spherical_to_ric_vector(delta_v_mag: float, theta: float, phi: float) -> np.ndarray:
    """Converts spherical angles (theta, phi) to a Cartesian RIC vector."""
    u_r = np.sin(phi) * np.cos(theta)
    u_i = np.sin(phi) * np.sin(theta)
    u_c = np.cos(phi)
    return delta_v_mag * np.array([u_r, u_i, u_c])

def delta_v_to_teme(
    r_vec: np.ndarray,
    v_vec: np.ndarray,
    delta_v_mag: float,
    theta: float,
    phi: float
) -> np.ndarray:
    """Converts spherical RIC burn angles into an inertial TEME delta-v vector."""
    delta_v_ric = spherical_to_ric_vector(delta_v_mag, theta, phi)
    return ric_to_teme(r_vec, v_vec, delta_v_ric)

delta_v_to_eci = delta_v_to_teme

def compute_semi_major_axis(r_vec: np.ndarray, v_vec: np.ndarray, mu: float = MU_EARTH) -> float:
    """Calculates orbital semi-major axis from vis-viva."""
    r_norm = np.linalg.norm(r_vec)
    v_norm = np.linalg.norm(v_vec)
    
    denom = (2.0 / r_norm) - (v_norm**2 / mu)
    if abs(denom) < 1e-9:
        return float('inf')
    return 1.0 / denom

def compute_relative_state(
    r_sat: np.ndarray,
    v_sat: np.ndarray,
    r_deb: np.ndarray,
    v_deb: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Computes relative position, velocity, scalar distance, and range rate."""
    r_rel = r_deb - r_sat
    v_rel = v_deb - v_sat
    dist = float(np.linalg.norm(r_rel))
    range_rate = float(np.dot(r_rel, v_rel) / dist) if dist > 1e-6 else 0.0
    return r_rel, v_rel, dist, range_rate

def compute_relative_state_ric(
    r_sat: np.ndarray,
    v_sat: np.ndarray,
    r_deb: np.ndarray,
    v_deb: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Computes relative position and velocity in the satellite's local RIC frame."""
    r_rel_teme, v_rel_teme, dist, range_rate = compute_relative_state(r_sat, v_sat, r_deb, v_deb)
    m_teme_to_ric = teme_to_ric_matrix(r_sat, v_sat)
    r_rel_ric = m_teme_to_ric @ r_rel_teme
    v_rel_ric = m_teme_to_ric @ v_rel_teme
    return r_rel_ric, v_rel_ric, dist, range_rate

