import csv
import numpy as np
import matplotlib.pyplot as plt

from Config import cfg
import Rocket_Physics
import PID
import Sensors
import Quaternion_Utils
import Drag_Solver
import EKF


def main() -> None:
    """
    Run the Monte Carlo suite of N = cfg.N simulations and report performance.
    """
    # Initialization
    # Compute baseline drag constant (no brakes deployed)
    k_no_brake = Drag_Solver.compute_k(
        Drag_Solver.cd_from_brake(0.0),
        Drag_Solver.area_from_brake(0.0)
    )

    # EKF noise covariances
    Q = np.eye(9) * 1e-4
    Q[6:9, 6:9] = np.eye(3) * 10.0  # higher process noise for wind bias
    R = np.eye(5) * 0.5             # measurement noise covariance

    successes = []
    failures = []
    success_count = 0
    all_true_trajectories = []

    # Prepare CSV header
    csv_rows = [[
        "sim", "success",
        "wind_x", "wind_y", "wind_z",
        "p0_x", "p0_y", "p0_z",
        "v0_x", "v0_y", "v0_z",
        "true_ap", "ekf_ap",
        "rmse", "rmsex", "rmsey", "rmsez",
        "target"
    ]]

    # Monte Carlo Trials
    for i in range(1, cfg.N + 1):
        print(f"Run {i}/{cfg.N}...")

        # Select target and initial conditions
        if cfg.vary_target:
            target = np.random.uniform(cfg.target_min, cfg.target_max)
            p0_z = np.random.uniform(100, 200)
            v0_z = np.random.uniform(30, 80)
        else:
            target = cfg.target_apogee
            p0_z = np.random.uniform(150, 155)
            v0_z = np.random.uniform(50, 55)

        wind_xy = np.random.uniform(-5, 5, 2)
        wind_z = np.random.uniform(-0.5, 0.5)
        true_wind = np.array([*wind_xy, wind_z])

        p0 = np.array([0.0, 0.0, p0_z])
        v0_xy = np.random.uniform(-5, 5, 2)
        v0 = np.array([*v0_xy, v0_z])

        # Instantiate physics, sensors, filter, and controller
        model = Rocket_Physics.PhysicsModel(k_no_brake, wind=true_wind)
        altimeter = Sensors.BarometricAltimeter(noise_std=1.0)
        accelerometer = Sensors.InertialAccelerometer(noise_std=0.2)

        # Initial state and EKF estimate
        q0 = Quaternion_Utils.quaternion_from_two_vectors(
            np.array([0.0, 0.0, 1.0]), v0
        )
        state = Rocket_Physics.RocketState(p0, v0, quat=q0)

        x_est = np.hstack((p0, v0, [0.0, 0.0, 0.0]))  # initial EKF state
        P_est = np.eye(9) * 0.1                       # initial EKF covariance

        pid = PID.PIDController(10000, 100, 10, cfg.dt)
        airbrake_setting = 0.0

        true_traj = []
        est_traj = []
        vz_integrated = x_est[5]  # pseudo-measurement of vertical velocity

        # Flight Loop
        for t in np.arange(0.0, cfg.t_max, cfg.dt):
            # Propagate true state
            state = model.step(state, cfg.dt)
            true_traj.append(state.position.copy())

            # Sensor measurements
            z_baro = altimeter.measure(state)
            a_meas = accelerometer.measure(state, model)
            vz_integrated += a_meas[2] * cfg.dt
            z_vec = np.hstack((z_baro, a_meas, vz_integrated))

            # EKF prediction & update
            x_est, P_est = EKF.ekf_update_aug(
                x_est, P_est, z_vec, cfg.dt, model.k, Q, R
            )
            est_traj.append(x_est[:3].copy())

            # Compute required and experienced drag constants
            z_est = x_est[2]
            vz_est = x_est[5]
            k_req = Drag_Solver.binary_search_k(
                z_est, vz_est, target,
                Drag_Solver.compute_k(
                    Drag_Solver.cd_from_brake(0.0),
                    Drag_Solver.area_from_brake(0.0)
                ),
                Drag_Solver.compute_k(
                    Drag_Solver.cd_from_brake(1.0),
                    Drag_Solver.area_from_brake(1.0)
                )
            )
            k_exp = Drag_Solver.compute_experienced_k(a_meas, x_est[3:6])

            # PID control and update brake deployment
            pid.setpoint = k_req
            cmd = pid.update(k_exp) * cfg.dt
            airbrake_setting = np.clip(airbrake_setting + cmd, 0.0, 1.0)

            Cd = Drag_Solver.cd_from_brake(airbrake_setting)
            A = Drag_Solver.area_from_brake(airbrake_setting)
            model.set_k(Drag_Solver.compute_k(Cd, A))

            # Termination condition (ground impact)
            if state.position[2] < 0 or state.velocity[2] < -10:
                break

        # Convert to arrays
        true_traj = np.array(true_traj)
        est_traj = np.array(est_traj)
        all_true_trajectories.append(true_traj)

        # Post-Flight Metrics
        n = min(len(true_traj), len(est_traj))
        diffs = true_traj[:n] - est_traj[:n]
        rmse_all = np.sqrt(np.mean(np.sum(diffs**2, axis=1)))
        rmse_xyz = np.sqrt(np.mean(diffs**2, axis=0))
        true_ap = float(np.nanmax(true_traj[:, 2])) if true_traj.size else np.nan
        ekf_ap = float(np.nanmax(est_traj[:, 2])) if est_traj.size else np.nan

        success = (
            abs(true_ap - target) <= cfg.tol
            and abs(ekf_ap - target) <= cfg.tol
        )
        success_count += int(success)

        print(
            f"  → {'Success' if success else 'Fail'} "
            f"(true_ap={true_ap:.1f}, ekf_ap={ekf_ap:.1f})"
        )

        # Record inputs for possible input-space plotting
        entry = (p0_z, v0_z, target)
        (successes if success else failures).append(entry)

        # Append row to CSV data
        csv_rows.append([
            i, int(success),
            *true_wind, *p0, *v0,
            true_ap, ekf_ap,
            rmse_all, *rmse_xyz,
            target
        ])

        # Optional per-run plotting
        if cfg.plot_each_run:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.plot(*true_traj.T, label='True', linewidth=1.0)
            ax.plot(*est_traj.T, label='EKF est.', linewidth=1.0)
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_zlabel('Z (m)')
            ax.set_title(f'Run {i}: True vs EKF Trajectory')
            ax.legend()
            plt.show()

    # Summary & Outputs
    overall_pct = 100.0 * success_count / cfg.N
    print(f"\n{success_count}/{cfg.N} runs within ±{cfg.tol} m → {overall_pct:.1f}% success")

    # Optional input-space scatter
    if cfg.plot_input_space and cfg.vary_target:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        data = np.array(successes)
        ax.scatter(*data.T, alpha=0.6, s=20, label='Success')
        fail_data = np.array(failures)
        ax.scatter(*fail_data.T, alpha=0.1, s=5, color='gray', label='Fail')
        ax.set_xlabel('Init Altitude (m)')
        ax.set_ylabel('Init Vertical Speed (m/s)')
        ax.set_zlabel('Target Apogee (m)')
        ax.set_title(f'{len(successes)}/{cfg.N} Viable Input Combinations')
        ax.legend()
        plt.tight_layout()
        plt.show()

    # Optional save CSV results
    if cfg.save_results:
        filename = cfg.filename
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(csv_rows)
        print(f"Saved to {filename}")

    # Optional overplot of all true trajectories
    if cfg.plot_all_true:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        for traj in all_true_trajectories:
            ax.plot(*traj.T, alpha=0.3, linewidth=0.5, color='red')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(f'{cfg.N} Monte Carlo True Trajectories')
        plt.show()


if __name__ == "__main__":
    main()
