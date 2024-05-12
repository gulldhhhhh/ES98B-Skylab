#!/usr/bin/env python
# coding: utf-8

# # Simulator Frontend

# ### IMPORTANT: run cell below first!

# In[30]:


get_ipython().run_line_magic('run', 'Simulator_Backend.ipynb')


# ### 2D Simulator

# In[27]:


"""
ellipse_parameters: contains the centre, width, height and angle of slant for the ellipse orbited around

satellite_parameters: must contain the mass, drag coefficient, initial position, initial velocity, initial time, and tangential_velocity flag of the satellite.
    -  if the flag tangential_velocity = True, then initial velocity must be a scalar. Otherwise, initial velocity must be a list of 2 elements (XY velocity).

radar_parameters: contains some radar information in 'radar parameter', and some noise level
    - if passing the simple_radar = True flag, first input must contain an integer corresponding to the number of equidistributed radars on the ellipse
    - if passing the simple_radar = False flag, first input must contain an array of xy coordinates corresponding to radar positions on the ellipse
    - No matter the flag, must pass a noise level corresponding to a percentage of noise we expect in the reading
    - reading_interval specifies how many seconds there are between radar readings

optional parameters/flags: 
    - dt: contains the value for the timestepping algorithm; dt = 0.1 has pretty ok performance.
    - maxIter: how many iterations the forward step runs. default is 1,000,000. Consider reducing this if you're going to model stable orbits!
    - solver: what solver to use for forward stepping the satellite's position
    - simple_solver: Whether or not to use forward Euler as a method for forward stepping. Recommended to be set to False
    - simple_radar: A flag whether or not to use equally spaced radar arrays or not (see radar_parameters for required inputs for each flag value)

Outputs:
    - Explicit outputs are poshist and althist (histories of the position in XY and altitude of the satellite)
    - Implicit outputs are two CSV files and a JSON file:
        -- Radar_Positions_2D.csv: Contains the xy coordinates of the positions of the radar stations
        -- Radar_Readings_2D.csv: Contains the readings of the radars at each time step. For n radar stations, every n rows corresponds to one timestep
        -- Satellite_Information_2D.json: Contains a bunch of important information about initial and final positions and parameters
"""

ellipse_parameters = {
    'centre': (0,0),
    'width': 2 * radius_equatorial,
    'height': 2 * radius_equatorial,
    'angle': 0
}

satellite_parameters = {
    'mass': 3000,
    'drag coefficient': 2.2,
    'initial position': [0, radius_equatorial + 1000000],
    'initial velocity': 7000,
    'time': datetime.datetime(2024, 5, 8, 19, 33, 20),
    'tangential_velocity': True
}

radar_parameters = {
    'radar parameter': 6,
    'noise level (%)': 0.05,
    'reading_interval': 10
}

poshist, althist = Simulator_2D(ellipse_parameters, satellite_parameters, radar_parameters)


# ### 3D Simulator

# In[51]:


get_ipython().run_line_magic('run', 'Simulator_Backend.ipynb')

"""
satellite_parameters: must contain the mass, drag coefficient, initial position, initial velocity, and initial time of the satellite.

radar_parameters:
    - if passing the simple_radar = True flag, must contain an integer corresponding to the number of equidistributed radars on the equator
    - if passing the simple_radar = False flag, must contain an array of xyz coordinates corresponding to radar positions on the earth
    - 'reading type': set this value to 'XYZ' to get radar readings representing the noisy XYZ distance from a radar. Anything else gives (distance, altitude) pairs as outputs
    - No matter the flag, must pass a noise level corresponding to a percentage of noise we expect in the reading
    - 'reading interval' specifies how many seconds can be 

dt: contains the value for the timestepping algorithm. The default dt = 0.1 is pretty ok at performance
maxIter: how many iterations the forward step runs. default is 1,000,000. Consider reducing this if you're going to model stable orbits!
solver: what solver to use for forward stepping the satellite's position. Is Runge-Kutta 4-5 by default
kilometers: A flag to use km or m for values (recommended to use km). Is km by default
simple_solver: Whether or not to use forward Euler as a method for forward stepping. Recommended to be set to False, and is by default.
drag_type: can be "simple" or "complex": What kind of atmospheric model to use. "complex", chosen by default uses nmrlsise00.
simple_radar: A flag whether or not to use Equatorial radar arrays or not (see radar_parameters for required inputs for each flag value)
rotating_earth: A flag wheter or not to rotate the earth for the purposes of radar readings and final crash position. Is false by defaut.

Outputs:
    - Explicit outputs are poshist and althist (histories of the position in XYZ and altitude of the satellite)
    - Implicit outputs are two CSV files and a JSON file:
        -- Radar_Positions.csv: Contains the xyz coordinates of the positions of the radar stations
        -- Radar_Readings.csv: Contains the readings of the radars at each time step. For n radar stations, every n rows corresponds to one timestep
        -- Satellite_Information.json: Contains a bunch of important information about initial and final positions and parameters
"""

#Temporary Fix: Define here the speed in km/s, and change the 408.0 value to whatever "altitude" you want to start at.
#Helps generate some random initial tangential velocity
speed = 6
input_pos = np.sqrt(np.array(random_split()) * (radius_equatorial/1000 + 408.0)**2)
input_veloc = speed * (random_normal(input_pos))

#input_pos = np.sqrt(np.array([1,0,0]) * (radius_equatorial/1000 + 408.0)**2)
#input_veloc = speed * np.array([0,1,0])

satellite_parameters = {
    'mass': 3000,
    'drag coefficient': 2.2,
    'initial position': input_pos.tolist(),
    'initial velocity': input_veloc.tolist(),
    'initial time': datetime.datetime(2024, 5, 8, 19, 33, 20)
}

radar_parameters = {
    'radar parameter': 8,
    'reading type': 'Not XYZ',
    'noise level (%)': 0.05,
    'reading interval': 10
}

poshist, althist = Simulator(satellite_parameters, radar_parameters, maxIter=100000, rotating_earth=True)


# ### 3D Orbit GIF creator

# In[52]:


from matplotlib import animation

poshist = np.array(poshist)
init_elev = Cart2Spher(np.array([poshist[0]]))[0][1]
final_azmth = Cart2Spher(np.array([poshist[-1]]))[0][2]

fig = plt.figure()
plt.axis('off')
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot_surface(earth_ellipsoid[0],earth_ellipsoid[1],earth_ellipsoid[2], alpha = 0.3)
line, = ax1.plot(poshist[:,0],poshist[:,1],poshist[:,2])

#ax1.view_init(elev = init_elev + np.pi/2, azim = final_azmth + np.pi/4)
ax1.view_init(elev = 0, azim = 0)

def update(num, poshist, line):
    line.set_data(np.array([poshist[:num,0], poshist[:num,1]]))
    line.set_3d_properties(np.array(poshist[:num,2]))

N = np.arange(0,len(poshist),100).tolist()
N.append(len(poshist)-1)
N = iter(tuple(N))
plt.axis('off')
ani = animation.FuncAnimation(fig, update, N, fargs = (poshist, line), cache_frame_data=False, interval = 10, blit=False)
ani.save('SatelliteCrash.gif', writer='pillow')
plt.subplots_adjust(wspace = 0.9)
plt.close()


# In[ ]:




