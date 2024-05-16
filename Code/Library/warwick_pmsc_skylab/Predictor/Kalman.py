
import numpy as np
import pandas as pd
import warwick_pmsc_skylab
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF, MerweScaledSigmaPoints
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal

## Relative Position Estimation
def get_realtime(initial_radar_positions, initial_time, elapsed_time):
        """
        Adjusts radar positions to account for earth's rotation

        Parameters:
            initial_radar_positions (np.array): An array of x, y, z coordinates for each radar's position at midnight.
            initial_time (DateTime): A DateTime object corresponding to the time of first measurement of the radar
            elapsed_time (float): A float corresponding to how many seconds have elapsed since radars started taking readings.
        
        Returns:
            realtime_positions (np.array): A numpy array containing the x, y, z coordinates of the radar after rotation of the earth.
        """
        midnight_time = initial_time.replace(hour=0,minute=0,second=0,microsecond=0)
        init_elapsed_time = (initial_time - midnight_time).total_seconds()
        timefrac = np.array(init_elapsed_time + elapsed_time) / 86164
        spherpos = warwick_pmsc_skylab.Simulator.Cart2Spher(initial_radar_positions)
        azdiff = (timefrac * 2 * np.pi) % (2*np.pi) 
        updated_spherpos = spherpos + np.array([0, 0, azdiff])
        realtime_positions = warwick_pmsc_skylab.Simulator.Spher2Cart(updated_spherpos)
        return realtime_positions


def convert_distalt_to_xyz(radar_positions, radar_readings, reading_interval, sat_initpos, initial_time, multilateration_number=3, fixed_earth=True):
     """
     Converts radar readings of the form (distance, altitude) to absolute satellite readings of the form (x, y, z)

     Parameters:
        radar_positions (DataFrame): DataFrame containing the x, y, z coordinates of each reading
        radar_readings (DataFrame): DataFrame containing the (distance, altitude) readings of each satellite at various timesteps
        reading_interval (float): represents how much time passes between each radar reading
        sat_initpos (list): list containing the initial position of the satellite in x, y, z coordinates
        initial_time (DateTime): A DateTime corresponding to when the first reading was taken
        multilateration_number (int): Defines how many radar we use for the multilateration algorithm. Must be >=3.
        fixed_earth (Bool): Setting this to false will account for earth's rotation when analyzing radar data.
    
     Returns:
        sat_data_xyz (np.array): An array of estimated x, y, z for the satellite's position at each timestep.

     """
     radar_positions = radar_positions[['x','y','z']].values
     num_radars = radar_positions.shape[0]
     if num_radars < multilateration_number:
        multilateration_number = num_radars
     radar_readings = radar_readings[['distance','altitude']].values
     sat_data_xyz = np.zeros((int(radar_readings.shape[0]/num_radars),3))
     for i in range(0, radar_readings.shape[0], num_radars):
        data_batch = radar_readings[i:i+num_radars,:]
        #Select 3 Best Radar Stations
        sorted_indices = np.argsort(data_batch[:,0])[::-1]
        top_n_indices = sorted_indices[:multilateration_number]
        topn_batch = data_batch[top_n_indices]
        distances = topn_batch[:,0]
        if fixed_earth == True:
            topn_radars = radar_positions[top_n_indices]
        else:
            rot_radars = get_realtime(radar_positions, initial_time, i*reading_interval)
            topn_radars = rot_radars[top_n_indices]

        distance_func = lambda x: np.abs(np.sqrt(np.sum((topn_radars - x)**2, axis=1)) - distances)
        initial_guess = sat_initpos
        #print(initial_guess)
        result = least_squares(distance_func, initial_guess)

        sat_data_xyz[i//num_radars,:] = result.x
     return sat_data_xyz

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

    return pd.DataFrame(estimated_positions, columns=["x","y"])

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
        print("Initial Guess:" initial_guess)
        print(residuals(initial_guess))

        # Least squares optimization
        result = least_squares(residuals, initial_guess)
        estimated_positions[i, :] = result.x

    return pd.DataFrame(estimated_positions, columns=["x","y","z"])


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
    x_est = np.zeros((num_steps,state_dimension))  # State vector initialization
    cov_est = []
    # Initialize state with the first measurement and assume starting velocity is zero
    x_est[0, :] = [data_test.iloc[0]['x'], 0, data_test.iloc[0]['y'], 0, data_test.iloc[0]['z'], 0]

    # Kalman Filter Loop
    for i in range(1, num_steps):
        # Predict
        x_pred = F @ x_est[i-1, :]
        P_pred = F @ P @ F.T + Q

        # Measurement update
        z = data_test.iloc[i][['x', 'y', 'z']].values
        y = z - H @ x_pred  # Residual
        S = H @ P_pred @ H.T + R  # Residual covariance
        K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
        x_est[i, :] = x_pred + K @ y
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

def ukf_3d(data_test, dt, radar_noise, process_noise):
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
    ukf.R = np.diag([radar_noise, radar_noise, radar_noise])  # Measurement noise: assuming measurement noise is large
    ukf.Q = np.eye(6) * process_noise  # Process noise: assuming process noise is small

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


def plot_trajectory_3D(data_real, filter_type, estimated_states, Ps):
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

    # Plotting predictions
    ax.plot(estimated_states[:, 0], estimated_states[:, 2], estimated_states[:, 4], 'r-', label='Predictions')

    # Extract the last state and covariance for the heat map
    x_last = estimated_states[-1][ [0, 2]]
    P_last = Ps[-1][np.ix_([0, 2], [0, 2])]

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
    ax.set_title('%f Estimated Satellite Trajectory with Heat Map' % filter_type)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')

    # Legend
    ax.legend()

    # Show the plot
    plt.show()

def plot_trajectory_2D(data_test, filter_type, estimated_states, covariances):
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
    ax.errorbar(x_positions[1:], y_positions[1:], xerr=x_errors, yerr=y_errors, color='red', ecolor='lightgray', elinewidth=3, capsize=0, label='Estimated Positions')
    
    # Add titles and labels
    ax.set_title('%f Estimated Satellite Trajectory with Error Bars'% filter_type)
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.legend()

    # Display the plot
    plt.show()

# Main function to run chosen filter on input data
def run_filter(filter_type, dimension, visualize=False, dt=10.0, reading_type='XYZ', reading_interval=10, sat_initpos=[0,0,0], initial_time=None, multilateration_number=3, fixed_earth=True, radar_noise=5, process_noise=0.001):
    if filter_type == 'ekf' and dimension == '2d':
        reading_columns = ['x', 'y']
        position_columns = ['x', 'y']
        radar_data_path = 'Radar_Readings_2D.csv'
        radar_positions_path = 'Radar_Positions_2D.csv'
        radar_data = pd.read_csv(radar_data_path, names=reading_columns)
        radar_positions = pd.read_csv(radar_positions_path, names=position_columns)
        data_test = estimate_position_from_radars_2D(radar_positions, radar_data)

        num_steps = len(data_test)
        dt = 1.0
        x_est = np.zeros((4, num_steps))
        measurements = np.zeros((2, num_steps))

        x_est[:, 0] = [data_test.iloc[0]['x'], 0, data_test.iloc[0]['y'], 0]

        num_radars = len(radar_positions)
        measurement_noise = radar_noise
        P = np.diag([10, 1, 10, 1])
        cov_est = []
        Q = np.eye(4) * process_noise
        R = np.diag([measurement_noise, measurement_noise])
        for i in range(1, num_steps):
            measurements[:, i] = data_test.iloc[i][['x', 'y']].values
            x_est[:, i], P = extended_kalman_filter(x_est[:, i-1], P, measurements[:, i], Q, R, dt)
            
            cov_est.append(P)
        predicted_positions = np.array(x_est)
        predicted_cov = np.asarray(cov_est)
        if visualize:
            # Visualization logic based on filter type and dimension
            plot_trajectory_2D(data_test, filter_type, predicted_positions, predicted_cov)
       
            
        return predicted_positions, predicted_cov

    elif filter_type == 'kalman' and dimension == '2d':
        # Prepare and run Kalman Filter 2D
        reading_columns = ['x', 'y']
        position_columns = ['x', 'y']
        radar_data_path = 'Radar_Readings_2D.csv'
        radar_positions_path = 'Radar_Positions_2D.csv'
        radar_data = pd.read_csv(radar_data_path, names=reading_columns)
        radar_positions = pd.read_csv(radar_positions_path, names=position_columns)
        data_test = estimate_position_from_radars_2D(radar_positions, radar_data)

        # Constants
        num_radars = len(radar_positions)
        measurement_noise = radar_noise
        F = np.array([[1, dt, 0, 0], [0, 1, 0, 0], [0, 0, 1, dt], [0, 0, 0, 1]]) # State transition matrix
        H = np.array([[1, 0, 0, 0], [0, 0, 1, 0]]) # Measurement matrix
        R = np.diag([measurement_noise, measurement_noise]) # measurement noise 
        Q = np.diag([process_noise, process_noise, process_noise, process_noise]) # process noise
        P = np.diag([10, 0, 10, 0]) # Initial state covariance

        predicted_positions, predicted_cov = kalman_filter(data_test, F, H, Q, R, P, dt)
        if visualize:
            # Visualization logic based on filter type and dimension
            plot_trajectory_2D(data_test, filter_type, predicted_positions, predicted_cov)
       
            
        return predicted_positions, predicted_cov


    elif filter_type == 'kalman' and dimension == '3d':
        if reading_type == 'XYZ':
            reading_columns = ['x', 'y', 'z']
        else:
            reading_columns = ['distance','altitude']
        position_columns = ['x', 'y', 'z']
        radar_data_path = 'Radar_Readings.csv'
        radar_positions_path = 'Radar_Positions.csv'
        radar_data = pd.read_csv(radar_data_path, names=reading_columns)
        radar_positions = pd.read_csv(radar_positions_path, names=position_columns)
        if reading_type == 'XYZ':
            num_radars = radar_positions.to_numpy().shape[0]
            measurement_noise = radar_noise
            if fixed_earth:
                data_test = estimate_position_from_radars_3D(radar_positions, radar_data)
            else:
                num_radars = radar_positions.to_numpy().shape[0]
                num_readings = radar_data.to_numpy().shape[0]
                data_test = data_test = np.zeros((num_readings//num_radars,3))
                moving_radar_positions = np.zeros((num_radars,3))
                print(moving_radar_positions.shape)
                radar_positions = radar_positions[['x','y','z']].values
                for i in range(0, len(num_readings), num_radars):
                    moving_radar_positions[i:i+num_radars, :] = get_realtime(radar_positions, initial_time, i//num_radars *reading_interval)
                    some_pd = estimate_position_from_radars_3D(moving_radar_positions[i:i+num_radars], radar_data[i:i+num_radars])
                    data_test[i//num_radars, :] = some_pd.to_numpy()
                data_test = pd.DataFrame(data_test, columns=["x","y","z"])
        else:
            measurement_noise = radar_noise
            data_test = convert_distalt_to_xyz(radar_positions, radar_data, reading_interval, sat_initpos, initial_time, multilateration_number, fixed_earth)
            data_test = pd.DataFrame(data_test, columns=['x','y','z'])

        # Constants
        M_earth = 5.972e24
        drag = 1

        F = np.array([
            [1, dt, 0,  0,  0,  0],
            [0,  1 , 0,  0,  0,  0],
            [0,  0, 1, dt,  0,  0],
            [0,  0, 0,  1 ,  0,  0],
            [0,  0, 0,  0,  1, dt],
            [0,  0, 0,  0,  0,  1]
        ])
        H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0]
        ])
        R = np.diag([measurement_noise, measurement_noise, measurement_noise])  # Measurement noise covariance
        Q = np.diag([process_noise, 0, process_noise, 0, process_noise, 0])  # Process noise covariance
        P = np.diag([10, 0, 10, 0, 10, 0])  # Initial state covariance
        predicted_positions, predicted_cov = kalman_filter_3d(data_test, F, H, Q, R, P)
        
        if visualize:
            # Visualization logic based on filter type and dimension
            plot_trajectory_3D(data_test, filter_type, predicted_positions, predicted_cov)
        return predicted_positions, predicted_cov


    elif filter_type == 'ukf' and dimension == '3d':
        if reading_type == 'XYZ':
            reading_columns = ['x', 'y', 'z']
        else:
            reading_columns = ['distance','altitude']
        position_columns = ['x', 'y', 'z']
        radar_data_path = 'Radar_Readings.csv'
        radar_positions_path = 'Radar_Positions.csv'
        radar_data = pd.read_csv(radar_data_path, names=reading_columns)
        radar_positions = pd.read_csv(radar_positions_path, names=position_columns)
        if reading_type == 'XYZ':
            num_radars = radar_positions.to_numpy().shape[0]
            measurement_noise = radar_noise
            if fixed_earth:
                data_test = estimate_position_from_radars_3D(radar_positions, radar_data)
            else:
                num_readings = radar_data.to_numpy().shape[0]
                data_test = np.zeros((num_readings//num_radars,3))
                moving_radar_positions = np.zeros((num_radars,3))
                radar_positions = radar_positions[['x','y','z']].values
                for i in range(0, len(num_readings), num_radars):
                    moving_radar_positions[i:i+num_radars, :] = get_realtime(radar_positions, initial_time, i//num_radars *reading_interval)
                    some_pd = estimate_position_from_radars_3D(moving_radar_positions[i:i+num_radars], radar_data[i:i+num_radars])
                    data_test[i//num_radars,:] = some_pd.to_numpy()
                data_test = pd.DataFrame(np.array(data_test), columns=["x","y","z"])
                print(data_test)
        else:
            measurement_noise = radar_noise
            data_test = convert_distalt_to_xyz(radar_positions, radar_data, reading_interval, sat_initpos, initial_time, multilateration_number, fixed_earth)
            data_test = pd.DataFrame(data_test, columns=['x','y','z'])

        
        predicted_positions, predicted_cov = ukf_3d(data_test, dt, measurement_noise, process_noise)
        if visualize:
            # Visualization logic based on filter type and dimension
            plot_trajectory_3D(data_test, filter_type, predicted_positions, predicted_cov)
        return predicted_positions, predicted_cov

    

        
# # Example of how to use the backend
# if __name__ == '__main__':
    
#     run_filter('kalman', '3d', visualize=True, dt=1.0)
# Sample output:
# predicted position and covariance matrix, image output
