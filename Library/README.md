TITLE: Library


DESCRIPTION: This project was designed by the Skylab team for the ES98B Group Project module on the Predictive Modelling and Scientific Computing course.


CREDITS: created by Simon Coolsaet, Leixin Xu, Tanya Sanjeev, Gullveig Liang and Kyle Berwick with additional support from Prof. James Kermode


LICENSE: See Library/warwick_pmsc_skylab/LICENSE

USAGE: 
The easiest way to use this library is by reading the user guide called User_Guide.pdf.

Furthermore, docstrings have been outlined for all major functions, which can be retrieved by running help() on the function.


The main function of this library is run_GUI(), which opens an interactive window allowing the user to easily specify input parameters.

The other main functions are Simulator.Simulate(), Simulator.Simulate_2D() and Predictor.Kalman.run_filter(). The first of these two functions create .csv files with noisy simulated values.
The run_filter function runs a specified Kalman filter over the noisy input data from the simulator. Use help() to get specific information regarding each of these functions!