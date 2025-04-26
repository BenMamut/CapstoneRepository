import numpy as np
from Config import cfg

# Gravitational acceleration (m/s²)
g = cfg.g


def f_process_aug(x: np.ndarray, dt: float, k: float) -> np.ndarray:
    """
    Discrete‐time process model for position, velocity, and wind bias.

    Args:
        x: State vector [px, py, pz, vx, vy, vz, wx, wy, wz].
        dt: Time step (s).
        k: Aerodynamic drag constant.

    Returns:
        Predicted next‐step state vector.
    """
    # Unpack state
    p, v, w = x[:3], x[3:6], x[6:9]

    # Relative velocity and drag acceleration
    v_rel = v - w
    speed = np.linalg.norm(v_rel)
    drag = -k * speed * v_rel if speed > 1e-6 else np.zeros(3)

    # Total acceleration (gravity + drag)
    a = np.array([0.0, 0.0, -g]) + drag

    # Euler integration for position and velocity; wind bias unchanged
    p_next = p + v * dt
    v_next = v + a * dt
    w_next = w

    return np.hstack((p_next, v_next, w_next))


def compute_F_jacobian(x: np.ndarray, dt: float, k: float) -> np.ndarray:
    """
    Compute the Jacobian of the process model ∂f/∂x for linearization.

    Args:
        x: Current state vector.
        dt: Time step (s).
        k: Drag constant.

    Returns:
        9×9 state transition Jacobian matrix.
    """
    F = np.eye(9)
    # ∂p/∂v terms
    F[0:3, 3:6] = np.eye(3) * dt

    # Unpack velocities and wind
    v, w = x[3:6], x[6:9]
    v_rel = v - w
    speed = np.linalg.norm(v_rel)

    # If moving, add drag partials
    if speed > 1e-6:
        I3 = np.eye(3)
        outer = np.outer(v_rel, v_rel)
        dAd = -k * (I3 * speed + outer / speed)
        # ∂v_next/∂v and ∂v_next/∂w
        F[3:6, 3:6] += dt * dAd
        F[3:6, 6:9] -= dt * dAd

    return F


def h_measurement_aug(x: np.ndarray, k: float) -> np.ndarray:
    """
    Measurement model mapping state to predicted sensor outputs.

    Args:
        x: State vector.
        k: Drag constant.

    Returns:
        Measurement vector [pz, ax, ay, az, vz].
    """
    p, v, w = x[:3], x[3:6], x[6:9]
    # Relative velocity and drag
    v_rel = v - w
    speed = np.linalg.norm(v_rel)
    drag = -k * speed * v_rel if speed > 1e-6 else np.zeros(3)

    # Predict barometric altitude, inertial accel, and vertical velocity
    pz = p[2]
    a_pred = np.array([0.0, 0.0, -g]) + drag
    vz = v[2]

    return np.hstack((pz, a_pred, vz))


def compute_H_jacobian(x: np.ndarray, k: float) -> np.ndarray:
    """
    Compute the Jacobian of the measurement model ∂h/∂x.

    Args:
        x: Predicted state vector (after f_process_aug).
        k: Drag constant.

    Returns:
        5×9 measurement Jacobian matrix.
    """
    H = np.zeros((5, 9))
    # ∂pz/∂pz
    H[0, 2] = 1.0

    v, w = x[3:6], x[6:9]
    v_rel = v - w
    speed = np.linalg.norm(v_rel)

    # If moving, add drag partials for accel measurements
    if speed > 1e-6:
        I3 = np.eye(3)
        outer = np.outer(v_rel, v_rel)
        dAd = -k * (I3 * speed + outer / speed)
        H[1:4, 3:6] = dAd      # ∂a/∂v
        H[1:4, 6:9] = -dAd     # ∂a/∂w

    # ∂vz/∂vz
    H[4, 5] = 1.0

    return H


def ekf_update_aug(
    x: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    dt: float,
    k: float,
    Q: np.ndarray,
    R: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform one predict‐update cycle of the augmented-state EKF.

    Args:
        x: Prior state estimate.
        P: Prior covariance matrix.
        z: Measurement vector.
        dt: Time step.
        k: Drag constant.
        Q: Process noise covariance.
        R: Measurement noise covariance.

    Returns:
        Tuple of (updated state estimate, updated covariance).
    """
    # Predict
    x_pred = f_process_aug(x, dt, k)
    F = compute_F_jacobian(x, dt, k)
    P_pred = F @ P @ F.T + Q

    # Measurement update
    z_pred = h_measurement_aug(x_pred, k)
    H = compute_H_jacobian(x_pred, k)
    y = z - z_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)

    x_upd = x_pred + K @ y
    P_upd = (np.eye(9) - K @ H) @ P_pred

    return x_upd, P_upd
