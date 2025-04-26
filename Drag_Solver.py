import numpy as np
import math
from Config import cfg


def f_for_k(k: float, x: float, v: float, target_apogee: float) -> float:
    """
    Evaluate the function whose root gives the drag constant k required
    to reach the target apogee from current altitude and vertical speed.

    f(k) = g * exp(2 * k * (target_apogee - x)) - (g + k * v^2)

    Args:
        k: Trial aerodynamic constant.
        x: Current altitude (m).
        v: Current vertical speed (m/s).
        target_apogee: Desired apogee altitude (m).

    Returns:
        Value of f(k). Returns np.inf if the exponential argument is too large.
    """
    arg = 2 * k * (target_apogee - x)
    if arg > 700:
        return np.inf
    return cfg.g * math.exp(arg) - (cfg.g + k * v**2)


def binary_search_k(
    x: float,
    v: float,
    target_apogee: float,
    k_low: float,
    k_high: float,
    tol: float = 1e-4,
    max_iter: int = 50
) -> float:
    """
    Find the drag constant k that zeros f_for_k using binary search.

    Args:
        x: Current altitude (m).
        v: Current vertical speed (m/s).
        target_apogee: Desired apogee altitude (m).
        k_low: Lower bound for initial k.
        k_high: Upper bound for initial k.
        tol: Tolerance for convergence.
        max_iter: Maximum number of iterations.

    Returns:
        Approximated k value within tolerance.
    """
    lo, hi = k_low, k_high
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = f_for_k(mid, x, v, target_apogee)
        if abs(f_mid) < tol:
            return mid
        if f_mid > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def compute_experienced_k(a_meas: np.ndarray, v_est: np.ndarray) -> float:
    """
    Estimate the aerodynamic constant k from measured acceleration and
    estimated velocity.

    k = - (a_drag · v_est) / |v_est|^3

    Args:
        a_meas: Noisy acceleration measurement [ax, ay, az] (m/s²).
        v_est: Estimated velocity vector [vx, vy, vz] (m/s).

    Returns:
        Estimated k. Returns 0 if velocity magnitude is negligible.
    """
    v_norm = np.linalg.norm(v_est)
    if v_norm < 1e-6:
        return 0.0

    # Remove gravity component
    gravity = np.array([0.0, 0.0, -cfg.g])
    a_drag = a_meas - gravity

    # Project drag acceleration onto velocity
    a_along = np.dot(a_drag, v_est) / v_norm
    k = -a_along / (v_norm**2)
    return k


def area_from_brake(b: float) -> float:
    """
    Map normalized brake deflection to frontal area A (m²) via linear interpolation.

    Args:
        b: Brake setting [0.0, 1.0].

    Returns:
        Frontal area between cfg.A_min and cfg.A_max.
    """
    return cfg.A_min + b * (cfg.A_max - cfg.A_min)


def cd_from_brake(b: float) -> float:
    """
    Map normalized brake deflection to drag coefficient Cd via linear interpolation.

    Args:
        b: Brake setting [0.0, 1.0].

    Returns:
        Drag coefficient between cfg.Cd_min and cfg.Cd_max.
    """
    return cfg.Cd_min + b * (cfg.Cd_max - cfg.Cd_min)


def compute_k(Cd: float, A: float) -> float:
    """
    Compute aerodynamic constant k from drag coefficient and area.

    k = ½ * rho * Cd * A / mass

    Args:
        Cd: Drag coefficient.
        A: Frontal area (m²).

    Returns:
        Aerodynamic constant k.
    """
    return 0.5 * cfg.rho * Cd * A / cfg.mass
