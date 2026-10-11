"""Earth and orbital mechanics constants (WGS-84)."""

# Earth gravitational parameter (km^3 / s^2)
MU_EARTH = 398600.4418

# Earth equatorial radius (km)
R_EARTH = 6378.137

# J2 oblateness harmonic coefficient
J2_EARTH = 1.08262668e-3

# Earth rotation rate (rad/s)
OMEGA_EARTH = 7.2921159e-5

# Safety threshold for avoidance (km)
SAFETY_DISTANCE_KM = 1.0

# Coarse search threshold (km)
BROAD_THRESHOLD_KM = 200.0

# GA normalization references
DELTA_V_REF_KM_S = 0.001   # 1 m/s in km/s
SMA_REF_KM = 5.0           # 5 km semi-major axis shift

# GA fitness weights: fuel, orbit change, safety margin
WEIGHT_FUEL = 1.0
WEIGHT_SMA = 0.5
WEIGHT_SAFETY = 100.0

