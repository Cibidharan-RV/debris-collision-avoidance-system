"""Coordinate frames and transformations (TEME, ECI, and local RIC)."""

import numpy as np
from src.constants import MU_EARTH

def compute_ric_basis(r_vec: np.ndarray, v_vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes orthonormal basis vectors (u_r, u_i, u_c) for the local RIC frame."""
    r_norm = np.linalg.norm(r_vec)
    if r_norm < 1e-6:
        raise ValueError("Position vector norm too small for RIC computation.")
    u_r = r_vec / r_norm
    
    h_vec = np.cross(r_vec, v_vec)
    h_norm = np.linalg.norm(h_vec)
    if h_norm < 1e-9:
        raise ValueError("Angular momentum too small (degenerate or rectilinear orbit).")
    u_c = h_vec / h_norm
    
    u_i = np.cross(u_c, u_r)
    u_i = u_i / np.linalg.norm(u_i)
    return u_r, u_i, u_c

def compute_ric_rotation_matrix(r_vec: np.ndarray, v_vec: np.ndarray) -> np.ndarray:
    """Returns 3x3 rotation matrix from local RIC to inertial TEME/ECI."""
    u_r, u_i, u_c = compute_ric_basis(r_vec, v_vec)
    return np.column_stack((u_r, u_i, u_c))

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

# Backward-compatible alias
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

