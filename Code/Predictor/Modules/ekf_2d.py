import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# # Load the data
# data_path = 'data_test.csv'
# data_test = pd.read_csv(data_path, names=['x', 'y'])
# Initial conditions
num_steps = len(data_test)
dt = 1.0
x_est = np.zeros((4, num_steps))
measurements = np.zeros((2, num_steps))

x_est[:, 0] = [data_test.iloc[0]['x'], 0, data_test.iloc[0]['y'], 0]

P = np.diag([100, 10, 100, 10])
Q = np.diag([0.1, 0.1, 0.1, 0.1]) * 0.001
R = np.diag([100, 100])

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


# # Run the extended Kalman filter
# for i in range(1, num_steps):
#     x_est[:, i], P = extended_kalman_filter(x_est[:, i-1], P, measurements[:, i], Q, R, dt)
    
# # Plotting
# plt.figure(figsize=(10, 5))
# plt.plot(data_test['x'], data_test['y'], 'o', label='True Trajectory (Data Test)')
# plt.plot(measurements[0, :], measurements[1, :],color='red',alpha = 0.3,label='Noisy Measurements')
# plt.plot(x_est[0, :], x_est[2, :], label='EKF Estimated Trajectory')
# plt.title('EKF with Real Data Test Trajectory and Simulated Measurements')
# plt.xlabel('Position X')
# plt.ylabel('Position Y')
# plt.legend()
# plt.grid(True)
# plt.show()