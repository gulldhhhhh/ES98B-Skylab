import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# # Load the data
# data_path = 'data_test.csv'
# data_test = pd.read_csv(data_path, names=['x', 'y'])

# Constants
# dt = 1
F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]]) # State transition matrix
H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]]) # Measurement matrix
R = np.diag([8, 8]) # Assuming measurement noise is large
Q = np.diag([0.1, 0.1, 0.1, 0.1]) * 0.001 # Assuming process noise is small
P = np.diag([10, 1, 10, 1]) # Initial state covariance

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
    """
    num_steps = len(data_test)
    x_est = np.zeros((4, num_steps))  # State vector [x, vx, y, vy]

    # Initialise
    x_est[:, 0] = [data_test.iloc[0]['x'], 0, data_test.iloc[0]['y'], 0]

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

    return x_est

# # Example usage
# estimated_states = kalman_filter(data_test, F, H, Q, R, P, dt)

# # Plot results
# plt.figure(figsize=(12, 6))
# plt.plot(data_test['x'], data_test['y'], 'o', label='Measured')
# plt.plot(estimated_states[0, :], estimated_states[2, :], 'r-', label='Kalman Filter Predictions')
# plt.title('Satellite Trajectory Prediction Using Only Position Measurements')
# plt.xlabel('X Position')
# plt.ylabel('Y Position')
# plt.legend()
# plt.grid(True)
# plt.show()