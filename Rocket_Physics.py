import numpy as np
from Quaternion_Utils import quaternion_from_two_vectors
from Config import cfg

# Gravitational acceleration from configuration (m/s²)
g = cfg.g


class RocketState:
    """
    Represents the full state of the rocket at a given time.

    Attributes:
        position (np.ndarray): 3D position vector [x, y, z] in meters.
        velocity (np.ndarray): 3D velocity vector [vx, vy, vz] in m/s.
        quat (np.ndarray): Orientation quaternion [w, x, y, z].
    """
    def __init__(
        self,
        position: np.ndarray,
        velocity: np.ndarray,
        quat: np.ndarray = None
    ) -> None:
        """
        Initialize the rocket state.

        Args:
            position: Iterable of length 3 for initial position [x, y, z].
            velocity: Iterable of length 3 for initial velocity [vx, vy, vz].
            quat: Optional iterable of length 4 for orientation quaternion;
                  defaults to [1, 0, 0, 0] if not provided.
        """
        self.position = np.array(position, dtype=float)
        self.velocity = np.array(velocity, dtype=float)
        self.quat = np.array(
            quat if quat is not None else [1.0, 0.0, 0.0, 0.0],
            dtype=float
        )


class PhysicsModel:
    """
    Physics engine for the coasting phase of rocket flight, including gravity,
    aerodynamic drag, and wind effects.

    Attributes:
        k (float): Aerodynamic drag constant (½·ρ·Cd·A / mass).
        wind (np.ndarray): Constant 3D wind velocity vector [wx, wy, wz] in m/s.
    """
    def __init__(self, k: float, wind: np.ndarray = None) -> None:
        """
        Args:
            k: Initial aerodynamic drag constant.
            wind: Optional wind vector [wx, wy, wz]; defaults to [0,0,0].
        """
        self.k = k
        self.wind = np.array(wind if wind is not None else np.zeros(3), dtype=float)

    def set_k(self, new_k: float) -> None:
        """
        Update the aerodynamic drag constant at runtime.

        Args:
            new_k: New drag constant value.
        """
        self.k = new_k

    def acceleration(self, state: RocketState) -> np.ndarray:
        """
        Compute the instantaneous acceleration on the rocket.

        Args:
            state: Current RocketState (provides velocity).

        Returns:
            np.ndarray: 3D acceleration [ax, ay, az] in m/s².
                        Includes gravity and aerodynamic drag.
        """
        # Relative velocity to the wind
        v_rel = state.velocity - self.wind
        speed = np.linalg.norm(v_rel)

        # Compute drag: F_D = -k * |v_rel| * v_rel
        if speed > 1e-6:
            drag = -self.k * speed * v_rel
        else:
            drag = np.zeros(3)

        # Gravity acts in -z
        gravity = np.array([0.0, 0.0, -g])

        return gravity + drag

    def step(self, state: RocketState, dt: float) -> RocketState:
        """
        Advance the rocket state by one time step using semi-explicit Euler integration.

        Args:
            state: Current RocketState.
            dt: Time step in seconds.

        Returns:
            RocketState: New state after dt seconds.
        """
        # Compute acceleration
        a = self.acceleration(state)

        # Integrate velocity and position
        new_velocity = state.velocity + a * dt
        new_position = state.position + state.velocity * dt

        # Update orientation to align body axis [0,0,1] with new velocity
        if np.linalg.norm(new_velocity) > 1e-6:
            new_quat = quaternion_from_two_vectors(np.array([0.0, 0.0, 1.0]), new_velocity)
        else:
            new_quat = state.quat.copy()

        return RocketState(new_position, new_velocity, new_quat)
