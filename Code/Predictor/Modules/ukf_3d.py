import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import least_squares



# Input 
# # Constants
# radius_polar = 6356752
# radius_equatorial = 6378137
# earth_eccentricity_squared = 6.694379e-3
# M_earth = 5.972e24
# G = 6.673e-11
# mu = G * M_earth
# dt = 10  # Time step in seconds





# State transition function: Basic assumption is that the object is moving at a constant velocity
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
    points = MerweScaledSigmaPoints(6, alpha=0.1, beta=2., kappa=1.)
    ukf = UKF(dim_x=6, dim_z=3, fx=state_transition_function, hx=measurement_function, dt=dt, points=points)
    ukf.x = np.array([data_test.iloc[0]['x'], 0, data_test.iloc[0]['y'], 0, data_test.iloc[0]['z'], 0])  # initial state
    ukf.R = np.diag([100, 100, 100])  # Measurement noise: assuming measurement noise is large
    ukf.Q = np.eye(6) * 0.001  # Process noise: assuming process noise is small

    xs, zs = [], []
    for index, row in data_test.iterrows():
        z = [row['x'], row['y'], row['z']]
        ukf.predict()
        ukf.update(z)
        xs.append(ukf.x)
        zs.append(z)

    xs = np.array(xs)
    return xs

# # Sample Usage
# data_test = pd.DataFrame(positions_estimated[:], columns=['x', 'y', 'z'])

# estimated_states = ukf_3d(data_test, dt)

# # Plot the results

# plt.figure(figsize=(12, 8))
# ax = plt.axes(projection='3d')
# ax.scatter(data_test['x'], data_test['y'], data_test['z'], c='b', label='Measured Positions',alpha=0.2)
# ax.plot(estimated_states[:, 0], estimated_states[:, 2], estimated_states[:, 4], 'r-', label='UKF Predictions')
# ax.set_title('3D Satellite Trajectory Prediction Using UKF')
# ax.set_xlabel('X Position')
# ax.set_ylabel('Y Position')
# ax.set_zlabel('Z Position')
# ax.legend()
# plt.show()