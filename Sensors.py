import numpy as np


class BarometricAltimeter:
    """
    Simulates a barometric altimeter sensor that measures altitude (z-position)
    with Gaussian noise.
    """
    def __init__(self, noise_std: float):
        """
        Args:
            noise_std (float): Standard deviation of measurement noise.
        """
        self.noise_std = noise_std

    def measure(self, state) -> float:
        """
        Measures the altitude (z-axis position) with noise.

        Args:
            state: An object with a 'position' attribute (expected to be a 3D vector).

        Returns:
            float: Noisy altitude measurement.
        """
        true_altitude = state.position[2]
        noisy_altitude = true_altitude + np.random.normal(0.0, self.noise_std)
        return noisy_altitude


class InertialAccelerometer:
    """
    Simulates a 3-axis inertial accelerometer sensor that measures acceleration
    with Gaussian noise.
    """
    def __init__(self, noise_std: float):
        """
        Args:
            noise_std (float): Standard deviation of measurement noise per axis.
        """
        self.noise_std = noise_std

    def measure(self, state, model) -> np.ndarray:
        """
        Measures the 3D acceleration with noise.

        Args:
            state: An object representing the system's state.
            model: A model providing a method 'acceleration(state)' that returns the true acceleration.

        Returns:
            np.ndarray: Noisy 3D acceleration measurement.
        """
        true_acceleration = model.acceleration(state)
        noisy_acceleration = true_acceleration + np.random.normal(0.0, self.noise_std, 3)
        return noisy_acceleration
