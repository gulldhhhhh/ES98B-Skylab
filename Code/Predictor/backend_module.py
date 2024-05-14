
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

## Relative Position Estimation
def estimate_position_from_radars_2D(radar_positions, radar_readings):
    """
    Estimates the satellite position from multiple radar readings using nonlinear least squares.
    
    Parameters:
        radar_positions (DataFrame): DataFrame containing the x, y, z coordinates of each radar.
        radar_readings (DataFrame): DataFrame containing the distances from each radar to the satellite at various timesteps.

    Returns:
        estimated_positions (np.array): An array of estimated x, y, z positions for each timestep.
    """
    num_radars = len(radar_positions)
    num_timesteps = len(radar_readings) // num_radars
    estimated_positions = np.zeros((num_timesteps, 2))  # To store x, y for each timestep

    for i in range(num_timesteps):
        timestep_readings = radar_readings.iloc[i*num_radars:(i+1)*num_radars]

        def residuals(pos):
            """
            Calculate the difference between observed distances and the distances to the guess positions

            Args:
              pos: Guessed posiiton

            Returns: 
              Distance: Difference between observed distances and the distances to the guess positions

            """
            dists = np.sqrt((timestep_readings['x'] - pos[0])**2 +
                            (timestep_readings['y'] - pos[1])**2 )
            return dists

        initial_guess = np.mean(timestep_readings, axis=0)

        # Least squares optimization
        result = least_squares(residuals, initial_guess)
        estimated_positions[i] = result.x

    return estimated_positions

def estimate_position_from_radars_3D(radar_positions, radar_readings):
    """
    Estimates the satellite position from multiple radar readings using nonlinear least squares.
    
    Parameters:
        radar_positions (DataFrame): DataFrame containing the x, y, z coordinates of each radar.
        radar_readings (DataFrame): DataFrame containing the distances from each radar to the satellite at various timesteps.

    Returns:
        estimated_positions (np.array): An array of estimated x, y, z positions for each timestep.
    """
    num_radars = len(radar_positions)
    num_timesteps = len(radar_readings) // num_radars
    estimated_positions = np.zeros((num_timesteps, 3))  # To store x, y, z for each timestep

    for i in range(num_timesteps):
        timestep_readings = radar_readings.iloc[i*num_radars:(i+1)*num_radars]

        def residuals(pos):
            """
            Calculate the difference between observed distances and the distances to the guess positions

            Args:
              pos: Guessed posiiton

            Returns: 
              Distance: Difference between observed distances and the distances to the guess positions

            """
            dists = np.sqrt((timestep_readings['x'] - pos[0])**2 +
                            (timestep_readings['y'] - pos[1])**2 +
                            (timestep_readings['z'] - pos[2])**2)
            return dists

        initial_guess = np.mean(timestep_readings, axis=0)

        # Least squares optimization
        result = least_squares(residuals, initial_guess)
        estimated_positions[i] = result.x

    return estimated_positions


## EKF 2D

def f(x, dt):
    """ Simplified motion model """
    return np.array([x[0] + dt * x[1], x[1], x[2] + dt * x[3], x[3]])

def h(x):
    """ Measurement function, only measures position """
    return np.array([x[0], x[2]])

def jacobian_f(x, dt):
    """ Jacobian of f, the state transition function """
    return np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]])

def jacobian_h(x):
    """ Jacobian of h, the measurement function """
    return np.array([[1, 0, 0, 0], [0, 0, 1, 0]])

def extended_kalman_filter(x_est, P, z, Q, R, dt):
    """
    Performs the extended Kalman filter algorithm to estimate the state of a system.

    Parameters:
    - x_est (numpy.ndarray): The estimated state vector at the previous time step.
    - P (numpy.ndarray): The estimated error covariance matrix at the previous time step.
    - z (numpy.ndarray): The measurement vector at the current time step.
    - Q (numpy.ndarray): The process noise covariance matrix.
    - R (numpy.ndarray): The measurement noise covariance matrix.
    - dt (float): The time step duration.

    Returns:
    - x_est (numpy.ndarray): The estimated state vector at the current time step.
    - P (numpy.ndarray): The estimated error covariance matrix at the current time step.
    """
    # Predict
    F = jacobian_f(x_est, dt)
    x_pred = f(x_est, dt)
    P_pred = F @ P @ F.T + Q

    # Update
    H = jacobian_h(x_pred)
    z_pred = h(x_pred)
    y = z - z_pred
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_est = x_pred + K @ y
    P = (np.eye(4) - K @ H) @ P_pred

    return x_est, P


## LKF 2D
def kalman_filter(data_test, F, H, Q, R, P, dt):
    """
    Kalman filter implementation for satellite state prediction.

    Args:
        data_test (pandas.DataFrame): Input data containing measurements of the object's position.
        F (numpy.ndarray): State transition matrix.
        H (numpy.ndarray): Measurement matrix.
        Q (numpy.ndarray): Process noise covariance matrix.
        R (numpy.ndarray): Measurement noise covariance matrix.
        P (numpy.ndarray): Error covariance matrix.
        dt (float): Time step.

    Returns:
        x_est (numpy.ndarray): Estimated state vector [x, vx, y, vy] at each time step.
        cov_est (numpy.ndarray): Estimated covariance matrix at each time step.
    """
    num_steps = len(data_test)
    x_est = np.zeros((4, num_steps))  # State vector [x, vx, y, vy]

    # Initialise
    x_est[:, 0] = [data_test.iloc[0]['x'], 0, data_test.iloc[0]['y'], 0]
    cov_est = []

    # Kalman Filter implementation
    for i in range(1, num_steps):
        # Predict
        x_pred = F @ x_est[:, i-1]
        P_pred = F @ P @ F.T + Q

        # Update
        z = data_test.iloc[i][['x', 'y']].values
        y = z - H @ x_pred  # Measurement residual
        S = H @ P_pred @ H.T + R  # Residual covariance
        K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
        x_est[:, i] = x_pred + K @ y
        P = (np.eye(4) - K @ H) @ P_pred
        cov_est.append(P)
    return np.array(x_est), np.asarray(cov_est)


## LKF 3D

def kalman_filter_3d(data_test, F, H, Q, R, P):
    """
    Applies a 3D Kalman filter to estimate the state of a system based on measurements.

    Args:
        data_test (pandas.DataFrame): The input data containing measurements.
        F (numpy.ndarray): The state transition matrix.
        H (numpy.ndarray): The measurement matrix.
        Q (numpy.ndarray): The process noise covariance matrix.
        R (numpy.ndarray): The measurement noise covariance matrix.
        P (numpy.ndarray): The initial state covariance matrix.

    Returns:
        x_est (numpy.ndarray): The estimated state vector at each time step.

    """
    num_steps = len(data_test)
    state_dimension = 6  # [x, vx, y, vy, z, vz]
    x_est = np.zeros((state_dimension, num_steps))  # State vector initialization
    cov_est = []
    # Initialize state with the first measurement and assume starting velocity is zero
    x_est[:, 0] = [data_test.iloc[0]['x'], 0, data_test.iloc[0]['y'], 0, data_test.iloc[0]['z'], 0]

    # Kalman Filter Loop
    for i in range(1, num_steps):
        # Predict
        x_pred = F @ x_est[:, i-1]
        P_pred = F @ P @ F.T + Q

        # Measurement update
        z = data_test.iloc[i][['x', 'y', 'z']].values
        y = z - H @ x_pred  # Residual
        S = H @ P_pred @ H.T + R  # Residual covariance
        K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
        x_est[:, i] = x_pred + K @ y
        P = (np.eye(state_dimension) - K @ H) @ P_pred
        cov_est.append(P)
    return np.array(x_est), np.asarray(cov_est)

## UKF 3D
def state_transition_function(state, dt):
    # Simple motion model: x = vt + x0
    # State vector: [x, vx, y, vy, z, vz]
    x, vx, y, vy, z, vz = state
    new_x = x + vx * dt
    new_y = y + vy * dt
    new_z = z + vz * dt
    return [new_x, vx, new_y, vy, new_z, vz]

def measurement_function(state):
    # We only measure position, not velocity as it's unavailable
    x, vx, y, vy, z, vz = state
    return [x, y, z]

def ukf_3d(data_test, dt):
    """
    Performs Unscented Kalman Filtering (UKF) for 3D data.

    Args:
        data_test (pandas.DataFrame): Input data containing measurements for x, y, and z coordinates.
        dt (float): Time step between measurements.

    Returns:
        tuple: A tuple containing two numpy arrays: xs and Ps.
            - xs (numpy.ndarray): Array of estimated states at each time step.
            - Ps (numpy.ndarray): Array of estimated state covariance matrices at each time step.
    """
    points = MerweScaledSigmaPoints(6, alpha=0.1, beta=2., kappa=1.)
    ukf = UKF(dim_x=6, dim_z=3, fx=state_transition_function, hx=measurement_function, dt=dt, points=points)
    ukf.x = np.array([data_test.iloc[0]['x'], 0, data_test.iloc[0]['y'], 0, data_test.iloc[0]['z'], 0])  # initial state
    ukf.R = np.diag([85, 85, 85])  # Measurement noise: assuming measurement noise is large
    ukf.Q = np.eye(6) * 0.001  # Process noise: assuming process noise is small

    xs, Ps = [], []
    for index, row in data_test.iterrows():
        z = [row['x'], row['y'], row['z']]
        ukf.predict()
        ukf.update(z)
        xs.append(ukf.x)
        Ps.append(ukf.P)

    xs = np.array(xs)
    Ps = np.array(Ps)
    return xs, Ps


def plot_trajectory_3D(data_real, estimated_states, Ps):
    """
    Plots the 3D satellite trajectory and a heat map of the estimated positions.

    Parameters:
    - data_real (DataFrame): DataFrame containing the measured positions of the satellite.
    - estimated_states (ndarray): Array of estimated states from the UKF predictions.
    - Ps (ndarray): Array of covariance matrices for each estimated state.

    Returns:
    None
    """

    # Create a new figure with 3D projection
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plotting measured positions
    ax.scatter(data_real['x'], data_real['y'], data_real['z'], c='b', label='Measured Positions', alpha=0.2)

    # Plotting UKF predictions
    ax.plot(estimated_states[:, 0], estimated_states[:, 2], estimated_states[:, 4], 'r-', label='UKF Predictions')

    # Extract the last state and covariance for the heat map
    x_last = estimated_states[-1, [0, 2]]
    P_last = Ps[-1, np.ix_([0, 2], [0, 2])]

    # Generate grid of points for heat map in the XY plane at the last Z position
    z_last = estimated_states[-1, 4]
    x_grid, y_grid = np.mgrid[x_last[0]-50:x_last[0]+50:100j, x_last[1]-50:x_last[1]+50:100j]
    pos = np.dstack((x_grid, y_grid))
    rv = multivariate_normal(x_last, P_last)
    # Create a flattened array of Z values
    z_values = np.full(pos.shape[:-1], z_last)
    # Use contourf to plot heat map at the last Z position
    ax.contourf(x_grid, y_grid, z_values, rv.pdf(pos), levels=50, cmap='viridis', offset=z_last)

    # Setting labels and title
    ax.set_title('3D Satellite Trajectory Prediction Using UKF')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')

    # Legend
    ax.legend()

    # Show the plot
    plt.show()

def plot_trajectory_2D(data_test, estimated_states, covariances):
    # Extract positions and their uncertainties from the estimated states and covariances
    x_positions = estimated_states[0, :]
    y_positions = estimated_states[2, :]
    x_errors = np.sqrt([cov[0, 0] for cov in covariances])  # Square root of variance for x
    y_errors = np.sqrt([cov[2, 2] for cov in covariances])  # Square root of variance for y

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    # Plot measured positions
    ax.scatter(data_test['x'], data_test['y'], color='blue', label='Measured Positions', alpha=0.6)
    # Plot estimated positions
    ax.errorbar(x_positions, y_positions, xerr=x_errors, yerr=y_errors, fmt='o', color='red', ecolor='lightgray', elinewidth=3, capsize=0, label='Estimated Positions')
    
    # Add titles and labels
    ax.set_title('Estimated Satellite Trajectory with Error Bars')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()

    # Display the plot
    plt.show()

# Main function to run chosen filter on input data
def run_filter(filter_type, dimension, visualize=False, dt=1.0):
    if filter_type == 'ekf' and dimension == '2d':
        reading_columns = ['x', 'y']
        position_columns = ['x', 'y']
        radar_data_path = 'Radar_Readings.csv'
        radar_positions_path = 'Radar_Positions.csv'
        radar_data = pd.read_csv(radar_data_path, names=reading_columns)
        radar_positions = pd.read_csv(radar_positions_path, names=position_columns)
        data_test = estimate_position_from_radars_2D(radar_positions, radar_data)

        num_steps = len(data_test)
        dt = 1.0
        x_est = np.zeros((4, num_steps))
        measurements = np.zeros((2, num_steps))

        x_est[:, 0] = [data_test.iloc[0]['x'], 0, data_test.iloc[0]['y'], 0]

        P = np.diag([100, 10, 100, 10])
        cov_est = []
        Q = np.diag([0.1, 0.1, 0.1, 0.1]) * 0.001
        R = np.diag([10, 10])
        for i in range(1, num_steps):
            measurements[:, i] = data_test.iloc[i][['x', 'y']].values
            x_est[:, i], P = extended_kalman_filter(x_est[:, i-1], P, measurements[:, i], Q, R, dt)
            # cov_est.append(np.sqrt(np.diag(P)))
            cov_est.append(P)
        predicted_positions = np.array(x_est)
        predicted_cov = np.asarray(cov_est)

    elif filter_type == 'kalman' and dimension == '2d':
        # Prepare and run Kalman Filter 2D
        reading_columns = ['x', 'y']
        position_columns = ['x', 'y']
        radar_data_path = 'Radar_Readings.csv'
        radar_positions_path = 'Radar_Positions.csv'
        radar_data = pd.read_csv(radar_data_path, names=reading_columns)
        radar_positions = pd.read_csv(radar_positions_path, names=position_columns)
        data_test = estimate_position_from_radars_2D(radar_positions, radar_data)

        # Constants
        F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]]) # State transition matrix
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]]) # Measurement matrix
        R = np.diag([8, 8]) # Assuming measurement noise is large
        Q = np.diag([0.1, 0.1, 0.1, 0.1]) * 0.001 # Assuming process noise is small
        P = np.diag([10, 1, 10, 1]) # Initial state covariance

        predicted_positions, predicted_cov = kalman_filter(data_test, F, H, Q, R, P, dt)


    elif filter_type == 'kalman' and dimension == '3d':
        reading_columns = ['x', 'y', 'z']
        position_columns = ['x', 'y', 'z']
        radar_data_path = 'Radar_Readings.csv'
        radar_positions_path = 'Radar_Positions.csv'
        radar_data = pd.read_csv(radar_data_path, names=reading_columns)
        radar_positions = pd.read_csv(radar_positions_path, names=position_columns)
        data_test = estimate_position_from_radars_3D(radar_positions, radar_data)

        # Constants
        M_earth = 5.972e24
        drag = 1

        # Settings
        F = np.array([
            [1, dt, 0,  0,  0,  0],
            [0,  1 - drag * dt / M_earth, 0,  0,  0,  0],
            [0,  0, 1, dt,  0,  0],
            [0,  0, 0,  1 - drag * dt / M_earth,  0,  0],
            [0,  0, 0,  0,  1, dt],
            [0,  0, 0,  0,  0,  1- drag * dt / M_earth]
        ])
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0]
        ])
        R = np.diag([8.5, 8.5, 8.5])  # Measurement noise covariance
        Q = np.diag([0.1, 0, 0.1, 0, 0.1, 0]) * 0.001  # Process noise covariance
        P = np.diag([10, 1, 10, 1, 10, 1])  # Initial state covariance

        predicted_positions, predicted_cov = kalman_filter_3d(data_test, F, H, Q, R, P)


    elif filter_type == 'ukf' and dimension == '3d':
        reading_columns = ['x', 'y', 'z']
        position_columns = ['x', 'y', 'z']
        radar_data_path = 'Radar_Readings.csv'
        radar_positions_path = 'Radar_Positions.csv'
        radar_data = pd.read_csv(radar_data_path, names=reading_columns)
        radar_positions = pd.read_csv(radar_positions_path, names=position_columns)
        data_test = estimate_position_from_radars_3D(radar_positions, radar_data)
        
        predicted_positions, predicted_cov = ukf_3d(data_test, dt)

    if visualize:
        # Visualization logic based on filter type and dimension
        if dimension == '2d':
            plot_trajectory_2D(data_test, predicted_positions, predicted_cov)
       
        if dimension == '3d':
            plot_trajectory_3D(data_test, predicted_positions, predicted_cov)

        
# Example of how to use the backend
if __name__ == '__main__':
    
    run_filter('ekf', '2d', visualize=True, dt=1.0)
