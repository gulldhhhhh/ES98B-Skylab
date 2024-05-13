# Description
For the predictor, here is the current plan and steps for implementations: https://docs.google.com/document/d/1zQnEGZjM29txwxAWLzHeEAaYB8sSWSuNGp2hKgga_oc/edit?usp=sharing

Currently, the backend_module.py in predictor folder is our semifinal version, this backend module includes all predictor stuff, currently not appending any functions in simulating earth but includes:
estimate the relative location of the satellite
- lkf 2d
- lkf 3d
- ukf 3d
- ekf 2d
- plotting for 2d: including error bar derived from covariance matrix
- plotting for 3d: including contourf mapping for final landing position

What needs to be improved:
- plotting: Earth simulation where easy for visual indication in crashing point

Sample usage of module:

run_filter('ekf', '2d', visualize=True, dt=1.0)
