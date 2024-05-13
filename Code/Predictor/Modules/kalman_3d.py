from scipy.optimize import least_squares
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

# Constants
M_earth = 5.972e24
dt = 10  # Time step in seconds
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

    return x_est

# # Sample Usage
# data_test = pd.DataFrame(positions_estimated[:], columns=['x', 'y', 'z'])


# # Run the Kalman Filter
# estimated_states = kalman_filter_3d(data_test, F, H, Q, R, P)

# # Plot the results
# plt.figure(figsize=(12, 8))
# ax = plt.axes(projection='3d')
# ax.scatter(data_test['x'], data_test['y'], data_test['z'], c='b', label='Measured Positions', alpha=0.2)
# ax.plot(estimated_states[0, :], estimated_states[2, :], estimated_states[4, :], 'r-', label='Kalman Filter Predictions')
# ax.set_title('3D Satellite Trajectory Prediction Using Kalman Filter')
# ax.set_xlabel('X Position')
# ax.set_ylabel('Y Position')
# ax.set_zlabel('Z Position')
# ax.legend()
# plt.show()
