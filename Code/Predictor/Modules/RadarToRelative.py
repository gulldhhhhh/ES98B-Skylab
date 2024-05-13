import numpy as np
import pandas as pd
from scipy.optimize import least_squares

# # Load radar data
# reading_columns = ['x', 'y', 'z']
# position_columns = ['x', 'y', 'z']
# radar_data_path = '/content/Radar_Readings_XYZ.csv'
# radar_positions_path = '/content/Radar_Positions_XYZ.csv'
# radar_data = pd.read_csv(radar_data_path, names=reading_columns)
# radar_positions = pd.read_csv(radar_positions_path, names=position_columns)

def estimate_position_from_radars(radar_positions, radar_readings):
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

# # Sample Usage
# positions_estimated = estimate_position_from_radars(radar_positions, radar_data)
# len(positions_estimated)
