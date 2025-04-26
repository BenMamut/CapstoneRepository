class PIDController:
    """
    Proportional–Integral–Derivative (PID) controller for air-brake actuation.

    Attributes:
        Kp (float): Proportional gain.
        Ki (float): Integral gain.
        Kd (float): Derivative gain.
        dt (float): Control loop time step (seconds).
        setpoint (float): Desired target value for the controlled variable.
        integral (float): Accumulated integral of the error.
        prev_error (float): Error value from the previous update, for derivative term.
    """
    def __init__(
        self,
        Kp: float,
        Ki: float,
        Kd: float,
        dt: float,
        setpoint: float = 0.0
    ) -> None:
        """
        Initialize the PID controller with specified gains and time step.

        Args:
            Kp: Proportional gain.
            Ki: Integral gain.
            Kd: Derivative gain.
            dt: Time step for each control update (seconds).
            setpoint: Initial desired setpoint (default 0.0).
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, measured_value: float) -> float:
        """
        Compute the PID control output based on the measured process variable.

        Args:
            measured_value: Current measured value of the variable being controlled.

        Returns:
            float: Control output (e.g., brake adjustment command).
        """
        # Calculate the error term
        error = self.setpoint - measured_value

        # Update the integral term
        self.integral += error * self.dt

        # Compute the derivative term
        derivative = (error - self.prev_error) / self.dt

        # Store error for next derivative calculation
        self.prev_error = error

        # Compute and return the PID output
        return self.Kp * error + self.Ki * self.integral + self.Kd * derivative
