#!/usr/bin/env python
# coding: utf-8

# In[9]:


import sys
import datetime
import warwick_pmsc_skylab.Simulator
import warwick_pmsc_skylab.Predictor.Kalman
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
from PyQt5.QtWidgets import QMessageBox, QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QRadioButton, QGroupBox, QHBoxLayout, QDateTimeEdit, QMainWindow, QCheckBox, QButtonGroup, QTextEdit, QDialog, QSlider
from PyQt5.QtCore import Qt, QEvent

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.image import imread
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas, NavigationToolbar2QT as NavigationToolbar
import matplotlib.patches as patches
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.stats import multivariate_normal


# In[10]:

class LoadingScreen(QDialog):
    """
    A dialog window that displays a loading message.
    
    Args:
        message (str): The message to be displayed on the loading screen.
    """
    
    def __init__(self, message="Loading..."):
        super().__init__()
        self.setWindowTitle("Please Wait")
        layout = QVBoxLayout()
        self.label = QLabel(message)
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.setFixedSize(200, 100)


class Window_2D(QWidget):
    def __init__(self):
        """
        Initializes the 2D Model Input GUI.

        Sets the window title and creates the layout for the GUI.
        Defines the sections for ellipse parameters, satellite parameters, radar parameters,
        simulator parameters, and predictor parameters.
        Adds the necessary widgets and layouts to each section.
        Connects signals and slots for button clicks and checkbox state changes.
        """
        
        super().__init__()
        self.setWindowTitle("2D Model Input")
        layout = QVBoxLayout()

        toplevel_1 = QHBoxLayout()

        #Define Ellipse Sectoin
        ellipsebox = QGroupBox("Ellipse Parameters")
        ellipse_layout = QVBoxLayout()

        self.centre_layout = QHBoxLayout()
        self.centre_layout.addWidget(QLabel("Centre (x,y):"))
        self.centre_text = QLineEdit("(0,0)")
        self.centre_layout.addWidget(self.centre_text)

        self.width_layout = QHBoxLayout()
        self.width_layout.addWidget(QLabel("Width:"))
        self.width_text = QLineEdit("12756274")
        self.width_layout.addWidget(self.width_text)

        self.height_layout = QHBoxLayout()
        self.height_layout.addWidget(QLabel("Height:"))
        self.height_text = QLineEdit("12756274")
        self.height_layout.addWidget(self.height_text)

        self.angle_layout = QHBoxLayout()
        self.angle_layout.addWidget(QLabel("Angle:"))
        self.angle_text = QLineEdit("0")
        self.angle_layout.addWidget(self.angle_text)

        ellipse_layout.addLayout(self.centre_layout)
        ellipse_layout.addLayout(self.width_layout)
        ellipse_layout.addLayout(self.height_layout)
        ellipse_layout.addLayout(self.angle_layout)
        ellipsebox.setLayout(ellipse_layout)


        #Satellite Section
        satellitebox = QGroupBox("Satellite Parameters")
        satellite_layout = QVBoxLayout()

        self.mass_layout = QHBoxLayout()
        self.mass_layout.addWidget(QLabel("Mass (kg):"))
        self.mass_text = QLineEdit("3000")
        self.mass_layout.addWidget(self.mass_text)

        self.drag_layout = QHBoxLayout()
        self.drag_layout.addWidget(QLabel("Drag Coefficient:"))
        self.drag_text = QLineEdit("2.2")
        self.drag_layout.addWidget(self.drag_text)

        self.initpos_layout = QHBoxLayout()
        self.initpos_layout.addWidget(QLabel("Initial Position:"))
        self.initpos_text = QLineEdit()
        self.initpos_layout.addWidget(self.initpos_text)
        self.default_initpos = QPushButton("Sample")
        self.default_initpos.clicked.connect(self.set_initpos)
        self.initpos_layout.addWidget(self.default_initpos)

        self.initspeed_layout = QHBoxLayout()
        self.initspeed_layout.addWidget(QLabel("Initial Speed (Clockwise):"))
        self.initspeed_text = QLineEdit()
        self.initspeed_layout.addWidget(self.initspeed_text)

        self.initveloc_norm_layout = QHBoxLayout()
        self.initveloc_norm_layout.addWidget(QLabel("Initial Direction:"))
        self.initveloc_norm_text = QLineEdit()
        self.initveloc_norm_layout.addWidget(self.initveloc_norm_text)
        self.tangent_velocity_flag = QCheckBox("Auto Tangent")
        self.tangent_velocity_flag.stateChanged.connect(self.tangent_velocity)
        self.initveloc_norm_layout.addWidget(self.tangent_velocity_flag)

        self.inittime_layout = QHBoxLayout()
        self.inittime_layout.addWidget(QLabel("Initial Time:"))
        self.inittime_text = QDateTimeEdit(calendarPopup=True)
        self.inittime_layout.addWidget(self.inittime_text)

        satellite_layout.addLayout(self.mass_layout)
        satellite_layout.addLayout(self.drag_layout)
        satellite_layout.addLayout(self.initpos_layout)
        satellite_layout.addLayout(self.initspeed_layout)
        satellite_layout.addLayout(self.initveloc_norm_layout)
        satellite_layout.addLayout(self.inittime_layout)
        satellitebox.setLayout(satellite_layout)

        toplevel_1.addWidget(ellipsebox)
        toplevel_1.addWidget(satellitebox)
        layout.addLayout(toplevel_1)

        #Radar Section
        radarbox = QGroupBox("Radar Parameters")
        radar_layout = QVBoxLayout()

        self.radarcomplexity = QHBoxLayout()
        self.radiobbox = QButtonGroup()
        self.radiosimple = QRadioButton("Simple Radar")
        self.radiocomplex = QRadioButton("Complex Radar")
        self.radiobbox.addButton(self.radiosimple)
        self.radiobbox.addButton(self.radiocomplex)
        self.radiosimple.setChecked(True)
        self.radarcomplexity.addWidget(self.radiosimple)
        self.radarcomplexity.addWidget(self.radiocomplex)

        self.radarparam_layout = QHBoxLayout()
        self.radarparam_layout.addWidget(QLabel("Radar Numbers (simple)/Locations (complex):"))
        self.radarparam_text = QLineEdit("8")
        self.radarparam_layout.addWidget(self.radarparam_text)

        self.noiselevel_layout = QHBoxLayout()
        self.noiselevel_layout.addWidget(QLabel("Radar Noise (%):"))
        self.noiselevel_text = QLineEdit("0.05")
        self.noiselevel_layout.addWidget(self.noiselevel_text)

        self.readint_layout = QHBoxLayout()
        self.readint_layout.addWidget(QLabel("Readings Interval (s):"))
        self.readint_text = QLineEdit("10")
        self.readint_layout.addWidget(self.readint_text)

        radar_layout.addLayout(self.radarcomplexity)
        radar_layout.addLayout(self.radarparam_layout)
        radar_layout.addLayout(self.noiselevel_layout)
        radar_layout.addLayout(self.readint_layout)
        radarbox.setLayout(radar_layout)

        layout.addWidget(radarbox)

        toplevel_2 = QHBoxLayout()

        #Simulator Parameters
        simulatorbox = QGroupBox("Simulator Parameters")
        simulator_layout = QVBoxLayout()

        self.steptime_layout = QHBoxLayout()
        self.steptime_buttons = QButtonGroup()
        self.steptime_small = QRadioButton("High Accuracy")
        self.steptime_large = QRadioButton("Low Accuracy")
        self.steptime_custom = QRadioButton("Custom Timestep:")
        self.steptime_text = QLineEdit("0.1")
        self.steptime_text.setEnabled(False)
        self.steptime_buttons.addButton(self.steptime_small)
        self.steptime_buttons.addButton(self.steptime_large)
        self.steptime_buttons.addButton(self.steptime_custom)
        self.steptime_small.setChecked(True)
        self.steptime_layout.addWidget(self.steptime_small)
        self.steptime_layout.addWidget(self.steptime_large)
        self.steptime_layout.addWidget(self.steptime_custom)
        self.steptime_layout.addWidget(self.steptime_text)

        self.steptime_small.clicked.connect(self.change_steptime)
        self.steptime_large.clicked.connect(self.change_steptime)
        self.steptime_custom.clicked.connect(self.change_steptime)

        self.maxiter_layout = QHBoxLayout()
        self.maxiter_buttons = QButtonGroup()
        self.maxiter_small = QRadioButton("Satellite Orbiting")
        self.maxiter_big = QRadioButton("Satellite De-orbiting")
        self.maxiter_custom = QRadioButton("Custom maxIter:")
        self.maxiter_text = QLineEdit("1000000")
        self.maxiter_text.setEnabled(False)
        self.maxiter_buttons.addButton(self.maxiter_small)
        self.maxiter_buttons.addButton(self.maxiter_big)
        self.maxiter_buttons.addButton(self.maxiter_custom)
        self.maxiter_big.setChecked(True)
        self.maxiter_layout.addWidget(self.maxiter_small)
        self.maxiter_layout.addWidget(self.maxiter_big)
        self.maxiter_layout.addWidget(self.maxiter_custom)
        self.maxiter_layout.addWidget(self.maxiter_text)

        self.maxiter_small.clicked.connect(self.change_maxiter)
        self.maxiter_big.clicked.connect(self.change_maxiter)
        self.maxiter_custom.clicked.connect(self.change_maxiter)

        self.simple_solver_layout = QHBoxLayout()
        self.simple_solver = QCheckBox("Simple Forward Euler Solver (Not Recommended)")
        self.simple_solver_layout.addWidget(self.simple_solver)

        simulator_layout.addLayout(self.steptime_layout)
        simulator_layout.addLayout(self.maxiter_layout)
        simulator_layout.addLayout(self.simple_solver_layout)
        simulatorbox.setLayout(simulator_layout)
        
        #layout.addWidget(simulatorbox)

        # Predictor Parameters

        predictorbox = QGroupBox("Predictor Parameters")
        predictor_layout = QVBoxLayout()

        self.filtertype_layout = QHBoxLayout()
        self.filtertype_bbox = QButtonGroup()
        self.filtertype_ekf = QRadioButton("Extended Kalman Filter")
        self.filtertype_kalman = QRadioButton("Linear Kalman Filter")
        self.filtertype_ekf.setChecked(True)
        self.filtertype_bbox.addButton(self.filtertype_ekf)
        self.filtertype_bbox.addButton(self.filtertype_kalman)
        self.filtertype_layout.addWidget(self.filtertype_ekf)
        self.filtertype_layout.addWidget(self.filtertype_kalman)

        self.pred_dt_layout = QHBoxLayout()
        self.pred_dt_layout.addWidget(QLabel("Kalman dt:"))
        self.pred_dt_text = QLineEdit("10.0")
        self.pred_dt_layout.addWidget(self.pred_dt_text)

        self.process_noise_layout = QHBoxLayout()
        self.process_noise_layout.addWidget(QLabel("Process Noise:"))
        self.process_noise_text = QLineEdit("0.01")
        self.process_noise_layout.addWidget(self.process_noise_text)

        predictor_layout.addLayout(self.process_noise_layout)
        predictor_layout.addLayout(self.filtertype_layout)
        predictor_layout.addLayout(self.pred_dt_layout)
        predictorbox.setLayout(predictor_layout)


        #layout.addWidget(predictorbox)
        toplevel_2.addWidget(simulatorbox)
        toplevel_2.addWidget(predictorbox)

        layout.addLayout(toplevel_2)


        button_layout = QHBoxLayout()
        self.button_back = QPushButton("Back")
        self.button_back.clicked.connect(self.go_back)
        self.button_confirm = QPushButton("Confirm")
        self.button_confirm.clicked.connect(self.run_simulator)

        button_layout.addWidget(self.button_back)
        button_layout.addWidget(self.button_confirm)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def go_back(self):
        """
        Go back to the main window.

        Creates a new instance of the MainWindow class, shows it, and closes the current window.
        """
        self.w = MainWindow()
        self.w.show()
        self.close()
    
    def set_initpos(self):
        """
        Sets the initial position of the GUI window based on the provided width, height, and center coordinates.

        If the width, height, and center coordinates are provided, the initial position is calculated as follows:
        - The x-coordinate is calculated as (width - center_x) + 0.1 * width
        - The y-coordinate is calculated as (height - center_y) + 0.1 * height

        If any of the width, height, or center coordinates are not provided, the default initial position is set to [100, 100].
        """
        if self.width_text.text() != "" and self.height_text != "" and self.centre_text != "":
            width = eval(self.width_text.text())
            height = eval(self.height_text.text())
            maxdim = max(width, height)
            distaway = 1.06 * (maxdim/2)
            randang = np.random.uniform(0,2*np.pi)
            self.initpos_text.setText(f"[{(np.cos(randang) * distaway):.2f},{(np.sin(randang) * distaway):.2f}]")

            #self.initpos_text.setText(f"[{((eval(self.width_text.text()) - eval(self.centre_text.text())[0])) + 0.1 * eval(self.width_text.text())}, {((eval(self.height_text.text()) - eval(self.centre_text.text())[1])) + 0.1 * eval(self.height_text.text())}]")
        else:
            self.initpos_text.setText("[100,100]")

    def tangent_velocity(self):
        if self.tangent_velocity_flag.isChecked():
            self.initveloc_norm_text.setText("")
            self.initveloc_norm_text.setEnabled(False)
        else:
            self.initveloc_norm_text.setEnabled(True)

    def change_steptime(self):
        if self.steptime_small.isChecked():
            self.steptime_text.setText("0.1")
            self.steptime_text.setEnabled(False)
        elif self.steptime_large.isChecked():
            self.steptime_text.setText("1.0")
            self.steptime_text.setEnabled(False)
        else:
            self.steptime_text.setText("")
            self.steptime_text.setEnabled(True)
    
    def change_maxiter(self):
        if self.maxiter_small.isChecked():
            self.maxiter_text.setText("100000")
            self.maxiter_text.setEnabled(False)
        elif self.maxiter_big.isChecked():
            self.maxiter_text.setText("1000000")
            self.maxiter_text.setEnabled(False)
        else:
            self.maxiter_text.setText("")
            self.maxiter_text.setEnabled(True)

    def show_error_message(self, message):
            """
            Display an error message dialog box.

            Args:
                message (str): The error message to be displayed.

            Returns:
                None
            """
            error_dialog = QMessageBox()
            error_dialog.setIcon(QMessageBox.Critical)
            error_dialog.setText(message)
            error_dialog.setWindowTitle("Error")
            error_dialog.exec_()

    def run_simulator(self):
        """
        Runs the simulator with the provided parameters.

        This method checks if all the required parameters are filled in and displays an error message if any parameter is missing.
        It then initializes the loading screen, shows it, and processes any pending events in the application.
        The method then retrieves the ellipse, satellite, and radar parameters from the GUI inputs.
        It sets the initial velocity of the satellite based on the user's selection of tangential velocity or custom vector.
        The method then prints the satellite parameters and sets the radar parameters.
        Finally, it calls the `Simulator_2D` method to run the simulator and stores the position and altitude history.
        It then passes to the `run_predictor` method.

        Returns:
            None
        """
        if not self.centre_text.text() or not self.width_text.text() or not self.height_text.text() or not self.angle_text.text():
            self.show_error_message("Please fill in all ellipse parameters.")
            return

        if not self.mass_text.text() or not self.drag_text.text() or not self.initpos_text.text() or not self.initspeed_text.text():
            self.show_error_message("Please fill in all satellite parameters.")
            return

        if not self.radarparam_text.text() or not self.noiselevel_text.text() or not self.readint_text.text():
            self.show_error_message("Please fill in all radar parameters.")
            return

        if not self.steptime_text.text() or not self.maxiter_text.text():
            self.show_error_message("Please fill in all simulator parameters.")
            return
        
        self.loading_screen = LoadingScreen("Running simulator...")
        self.loading_screen.show()
        QApplication.processEvents()

        ellipse_parameters = {
            'centre': eval(self.centre_text.text()),
            'width': eval(self.width_text.text()),
            'height': eval(self.height_text.text()),
            'angle': eval(self.angle_text.text())
        }

        satellite_parameters = {
            'mass': eval(self.mass_text.text()),
            'drag coefficient': eval(self.drag_text.text()),
            'initial position': eval(self.initpos_text.text()),
            'initial velocity': None,
            'time': self.inittime_text.dateTime().toPyDateTime(),
            'tangential_velocity': self.tangent_velocity_flag.isChecked()
        }

        if self.tangent_velocity_flag.isChecked():
            satellite_parameters["initial velocity"] = eval(self.initspeed_text.text())
        else:
            tempvec = eval(self.initveloc_norm_text.text())
            satellite_parameters["initial velocity"] = [eval(self.initspeed_text.text()) * tempvec[0], eval(self.initspeed_text.text()) * tempvec[1]]

        print(satellite_parameters)
        radar_parameters = {
            'radar parameter': eval(self.radarparam_text.text()),
            'noise level (%)': eval(self.noiselevel_text.text()),
            'reading_interval': eval(self.readint_text.text())
        }
        
        self.dt = eval(self.steptime_text.text())
        maxIter = eval(self.maxiter_text.text())
        simple_solver = self.simple_solver.isChecked()
        simple_radar = self.radiosimple.isChecked()

        self.poshist, self.althist = warwick_pmsc_skylab.Simulator.Simulator_2D(ellipse_parameters, satellite_parameters, radar_parameters, dee_t = self.dt, maxIter = maxIter, solver = 'RK45', simple_solver = simple_solver, simple_radar = simple_radar)

        #self.Handoff_2D(self.poshist, self.althist, None, None)
        self.run_predictor()
    
    def run_predictor(self):
        """
        Runs the predictor based on the selected filter type and updates the predicted positions and covariance.

        This method sets the loading screen label to indicate that the predictor is running. It then checks the selected
        filter type and assigns the corresponding value to the `filter_type` variable. The predictor is then executed
        using the selected filter type, 2D mode, and the provided parameters for time step, radar noise, and process noise.
        The predicted positions and covariance are stored in the `self.predicted_positions` and `self.predicted_cov`
        variables, respectively.

        Finally, the loading screen is closed and the `Handoff_2D` method is called with the historical positions,
        historical altitudes, predicted positions, and predicted covariance as arguments.

        Returns:
            None
        """
        self.loading_screen.label.setText("Running predictor...")
        QApplication.processEvents()

        if self.filtertype_ekf.isChecked():
            filter_type = 'ekf'
        else:
            filter_type = 'kalman'
        

        self.predicted_positions, self.predicted_cov = warwick_pmsc_skylab.Predictor.Kalman.run_filter(filter_type, '2d', dt = eval(self.pred_dt_text.text()), radar_noise = eval(self.noiselevel_text.text()), process_noise = eval(self.process_noise_text.text()))
        
        self.loading_screen.close()
        self.Handoff_2D(self.poshist, self.althist, self.predicted_positions, self.predicted_cov)

    def Handoff_2D(self, poshist, althist, predicted_positions, predicted_cov):
        self.w = VisualizationWindow('2D', poshist, althist, predicted_positions, predicted_cov, self.inittime_text.dateTime().toPyDateTime(), self.dt, eval(self.pred_dt_text.text()))
        self.w.show()
        self.close()




class Window_3D(QWidget):
    def __init__(self):
        """
        Initializes the 3D Model Input GUI.

        This method sets up the window title and creates the layout for the GUI.
        It also creates the sections for Satellite Parameters, Radar Parameters,
        Simulator Parameters, and Predictor Parameters. Each section contains
        various input fields and buttons for user interaction.

        Args:
            None

        Returns:
            None
        """
        
        super().__init__()
        self.setWindowTitle("3D Model Input")
        layout = QVBoxLayout()

        #Satellite Section
        toplevel_1 = QHBoxLayout()
        satellitebox = QGroupBox("Satellite Parameters")
        satellite_layout = QVBoxLayout()

        self.mass_layout = QHBoxLayout()
        self.mass_layout.addWidget(QLabel("Mass (kg):"))
        self.mass_text = QLineEdit("3000")
        self.mass_layout.addWidget(self.mass_text)

        self.drag_layout = QHBoxLayout()
        self.drag_layout.addWidget(QLabel("Drag Coefficient:"))
        self.drag_text = QLineEdit("2.2")
        self.drag_layout.addWidget(self.drag_text)

        self.initpos_layout = QHBoxLayout()
        self.initpos_layout.addWidget(QLabel("Initial Position (x, y, z) (km):"))
        self.initpos_text = QLineEdit()
        self.initpos_layout.addWidget(self.initpos_text)
        self.default_initpos = QPushButton("Sample")
        self.default_initpos.clicked.connect(self.set_initpos)
        self.initpos_layout.addWidget(self.default_initpos)

        self.initspeed_layout = QHBoxLayout()
        self.initspeed_layout.addWidget(QLabel("Initial Speed (km/s):"))
        self.initspeed_text = QLineEdit()
        self.initspeed_layout.addWidget(self.initspeed_text)

        self.initveloc_norm_layout = QHBoxLayout()
        self.initveloc_norm_layout.addWidget(QLabel("Initial Velocity Direction:"))
        self.initveloc_norm_text = QLineEdit()
        self.initveloc_norm_layout.addWidget(self.initveloc_norm_text)
        self.tangent_velocity_flag = QCheckBox("Auto Tangent")
        self.tangent_velocity_flag.stateChanged.connect(self.tangent_velocity)
        self.initveloc_norm_layout.addWidget(self.tangent_velocity_flag)

        self.inittime_layout = QHBoxLayout()
        self.inittime_layout.addWidget(QLabel("Initial Time:"))
        self.inittime_text = QDateTimeEdit(calendarPopup=True)
        self.inittime_layout.addWidget(self.inittime_text)

        satellite_layout.addLayout(self.mass_layout)
        satellite_layout.addLayout(self.drag_layout)
        satellite_layout.addLayout(self.initpos_layout)
        satellite_layout.addLayout(self.initspeed_layout)
        satellite_layout.addLayout(self.initveloc_norm_layout)
        satellite_layout.addLayout(self.inittime_layout)
        satellitebox.setLayout(satellite_layout)

        #layout.addWidget(satellitebox)


        #Radar Section
        radarbox = QGroupBox("Radar Parameters")
        radar_layout = QVBoxLayout()

        self.radarcomplexity = QHBoxLayout()
        self.complexity_buttongroup = QButtonGroup()
        self.radiosimple = QRadioButton("Simple Radar")
        self.radiocomplex = QRadioButton("Complex Radar")
        self.radiosimple.setChecked(True)
        self.complexity_buttongroup.addButton(self.radiosimple)
        self.complexity_buttongroup.addButton(self.radiocomplex)
        self.radarcomplexity.addWidget(self.radiosimple)
        self.radarcomplexity.addWidget(self.radiocomplex)

        self.radarparam_layout = QHBoxLayout()
        self.radarparam_layout.addWidget(QLabel("Radar Numbers (simple)/Locations (complex):"))
        self.radarparam_text = QLineEdit("8")
        self.radarparam_layout.addWidget(self.radarparam_text)

        self.readingtype = QHBoxLayout()
        self.read_buttongroup = QButtonGroup()
        self.readingxyz = QRadioButton("Relative XYZ Measurements")
        self.readingdistalt = QRadioButton("Distance-Altitude Measurements")
        self.readingxyz.setChecked(True)
        self.read_buttongroup.addButton(self.readingxyz)
        self.read_buttongroup.addButton(self.readingdistalt)
        self.readingtype.addWidget(self.readingxyz)
        self.readingtype.addWidget(self.readingdistalt)

        self.noiselevel_layout = QHBoxLayout()
        self.noiselevel_layout.addWidget(QLabel("Radar Noise (%):"))
        self.noiselevel_text = QLineEdit("0.05")
        self.noiselevel_layout.addWidget(self.noiselevel_text)

        self.readint_layout = QHBoxLayout()
        self.readint_layout.addWidget(QLabel("Readings Interval (s):"))
        self.readint_text = QLineEdit("10")
        self.readint_layout.addWidget(self.readint_text)

        radar_layout.addLayout(self.radarcomplexity)
        radar_layout.addLayout(self.radarparam_layout)
        radar_layout.addLayout(self.readingtype)
        radar_layout.addLayout(self.noiselevel_layout)
        radar_layout.addLayout(self.readint_layout)
        radarbox.setLayout(radar_layout)

        #layout.addWidget(radarbox)
        toplevel_1.addWidget(satellitebox)
        toplevel_1.addWidget(radarbox)
        layout.addLayout(toplevel_1)


        toplevel_2 = QHBoxLayout()
        # Simulator Parameters
        simulatorbox = QGroupBox("Simulator Parameters")
        simulator_layout = QVBoxLayout()

        self.steptime_layout = QHBoxLayout()
        self.steptime_buttons = QButtonGroup()
        self.steptime_small = QRadioButton("High Accuracy")
        self.steptime_large = QRadioButton("Low Accuracy")
        self.steptime_custom = QRadioButton("Custom Timestep:")
        self.steptime_text = QLineEdit("0.1")
        self.steptime_text.setEnabled(False)
        self.steptime_buttons.addButton(self.steptime_small)
        self.steptime_buttons.addButton(self.steptime_large)
        self.steptime_buttons.addButton(self.steptime_custom)
        self.steptime_small.setChecked(True)
        self.steptime_layout.addWidget(self.steptime_small)
        self.steptime_layout.addWidget(self.steptime_large)
        self.steptime_layout.addWidget(self.steptime_custom)
        self.steptime_layout.addWidget(self.steptime_text)

        self.steptime_small.clicked.connect(self.change_steptime)
        self.steptime_large.clicked.connect(self.change_steptime)
        self.steptime_custom.clicked.connect(self.change_steptime)

        self.maxiter_layout = QHBoxLayout()
        self.maxiter_buttons = QButtonGroup()
        self.maxiter_small = QRadioButton("Satellite Orbiting")
        self.maxiter_big = QRadioButton("Satellite De-orbiting")
        self.maxiter_custom = QRadioButton("Custom maxIter:")
        self.maxiter_text = QLineEdit("1000000")
        self.maxiter_text.setEnabled(False)
        self.maxiter_buttons.addButton(self.maxiter_small)
        self.maxiter_buttons.addButton(self.maxiter_big)
        self.maxiter_buttons.addButton(self.maxiter_custom)
        self.maxiter_big.setChecked(True)
        self.maxiter_layout.addWidget(self.maxiter_small)
        self.maxiter_layout.addWidget(self.maxiter_big)
        self.maxiter_layout.addWidget(self.maxiter_custom)
        self.maxiter_layout.addWidget(self.maxiter_text)

        self.maxiter_small.clicked.connect(self.change_maxiter)
        self.maxiter_big.clicked.connect(self.change_maxiter)
        self.maxiter_custom.clicked.connect(self.change_maxiter)

        self.dragtype = QHBoxLayout()
        self.dragtype_buttons = QButtonGroup()
        self.dragtype_simple = QRadioButton("Simple Atmospheric Model")
        self.dragtype_complex = QRadioButton("Complex Atmospheric Model")
        self.dragtype_complex.setChecked(True)
        self.dragtype_buttons.addButton(self.dragtype_simple)
        self.dragtype_buttons.addButton(self.dragtype_complex)
        self.dragtype.addWidget(self.dragtype_simple)
        self.dragtype.addWidget(self.dragtype_complex)

        self.rot_earth = QHBoxLayout()
        self.rot_earth_flag = QCheckBox("Rotating Earth")
        self.rot_earth.addWidget(self.rot_earth_flag)

        simulator_layout.addLayout(self.steptime_layout)
        simulator_layout.addLayout(self.maxiter_layout)
        simulator_layout.addLayout(self.dragtype)
        simulator_layout.addLayout(self.rot_earth)

        simulatorbox.setLayout(simulator_layout)
        
        #layout.addWidget(simulatorbox)

        predictorbox = QGroupBox("Predictor Parameters")
        predictor_layout = QVBoxLayout()

        self.filtertype_layout = QHBoxLayout()
        self.filtertype_bbox = QButtonGroup()
        self.filtertype_ukf = QRadioButton("Unscented Kalman Filter")
        self.filtertype_kalman = QRadioButton("Linear Kalman Filter")
        self.filtertype_ukf.setChecked(True)
        self.filtertype_bbox.addButton(self.filtertype_ukf)
        self.filtertype_bbox.addButton(self.filtertype_kalman)
        self.filtertype_layout.addWidget(self.filtertype_ukf)
        self.filtertype_layout.addWidget(self.filtertype_kalman)

        self.pred_dt_layout = QHBoxLayout()
        self.pred_dt_layout.addWidget(QLabel("Kalman dt:"))
        self.pred_dt_text = QLineEdit("10.0")
        self.pred_dt_layout.addWidget(self.pred_dt_text)

        self.process_noise_layout = QHBoxLayout()
        self.process_noise_layout.addWidget(QLabel("Process noise:"))
        self.process_noise_text = QLineEdit("0.01")
        self.process_noise_layout.addWidget(self.process_noise_text)

        predictor_layout.addLayout(self.process_noise_layout)
        predictor_layout.addLayout(self.filtertype_layout)
        predictor_layout.addLayout(self.pred_dt_layout)
        predictorbox.setLayout(predictor_layout)


        #layout.addWidget(predictorbox)

        toplevel_2.addWidget(simulatorbox)
        toplevel_2.addWidget(predictorbox)
        layout.addLayout(toplevel_2)

        button_layout = QHBoxLayout()
        self.button_back = QPushButton("Back")
        self.button_back.clicked.connect(self.go_back)
        self.button_confirm = QPushButton("Confirm")
        self.button_confirm.clicked.connect(self.run_simulator)

        button_layout.addWidget(self.button_back)
        button_layout.addWidget(self.button_confirm)
        layout.addLayout(button_layout)
        
        self.setLayout(layout)
    
    def go_back(self):
        self.w = MainWindow()
        self.w.show()
        self.close()
    
    def set_initpos(self):
        input_pos = np.round(np.sqrt(np.array(warwick_pmsc_skylab.Simulator.random_split()) * (warwick_pmsc_skylab.radius_equatorial/1000 + 408.0)**2), decimals=3)
        self.initpos_text.setText(f"{input_pos.tolist()}")

    def tangent_velocity(self):
        if self.tangent_velocity_flag.isChecked():
            self.initveloc_norm_text.setText(f"{warwick_pmsc_skylab.Simulator.random_normal(eval(self.initpos_text.text())).tolist()}")
        else:
            self.initveloc_norm_text.setText("")
    
    def change_steptime(self):
        if self.steptime_small.isChecked():
            self.steptime_text.setText("0.1")
            self.steptime_text.setEnabled(False)
        elif self.steptime_large.isChecked():
            self.steptime_text.setText("1.0")
            self.steptime_text.setEnabled(False)
        else:
            self.steptime_text.setText("")
            self.steptime_text.setEnabled(True)
    
    def change_maxiter(self):
        if self.maxiter_small.isChecked():
            self.maxiter_text.setText("100000")
            self.maxiter_text.setEnabled(False)
        elif self.maxiter_big.isChecked():
            self.maxiter_text.setText("1000000")
            self.maxiter_text.setEnabled(False)
        else:
            self.maxiter_text.setText("")
            self.maxiter_text.setEnabled(True)

    def show_error_message(self, message):
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setText(message)
        error_dialog.setWindowTitle("Error")
        error_dialog.exec_()

    def run_simulator(self):
        """
        Runs the simulator with the provided satellite and radar parameters.

        This method checks if all the required parameters are filled in the GUI.
        If any parameter is missing, it displays an error message and returns.
        Otherwise, it initializes the simulator with the provided parameters and runs it.
        The simulator calculates the position and altitude history of the satellite.
        Finally, it calls the `run_predictor` method to perform prediction based on the calculated history.
        """

        if not self.mass_text.text() or not self.drag_text.text() or not self.initpos_text.text() or not self.initspeed_text.text():
            self.show_error_message("Please fill in all satellite parameters.")
            return

        if not self.radarparam_text.text() or not self.noiselevel_text.text() or not self.readint_text.text():
            self.show_error_message("Please fill in all radar parameters.")
            return

        if not self.steptime_text.text() or not self.maxiter_text.text():
            self.show_error_message("Please fill in all simulator parameters.")
            return
        

        self.loading_screen = LoadingScreen("Running simulator...")
        self.loading_screen.show()
        QApplication.processEvents()

        self.satellite_parameters = {
            'mass': eval(self.mass_text.text()),
            'drag coefficient': eval(self.drag_text.text()),
            'initial position': np.array(eval(self.initpos_text.text())).tolist(),
            'initial velocity': (eval(self.initspeed_text.text()) * np.array(eval(self.initveloc_norm_text.text()))).tolist(),
            'time': self.inittime_text.dateTime().toPyDateTime()
        }


        if self.readingxyz.isChecked():
            rtype = 'XYZ'
        else:
            rtype = 'distalt'
        self.radar_parameters = {
            'radar parameter': eval(self.radarparam_text.text()),
            'reading type': rtype,
            'noise level (%)': eval(self.noiselevel_text.text()),
            'reading_interval': eval(self.readint_text.text())
        }
        
        dt = eval(self.steptime_text.text())
        maxIter = eval(self.maxiter_text.text())
        simple_radar = self.radiosimple.isChecked()

        self.poshist, self.althist = warwick_pmsc_skylab.Simulator.Simulator(self.satellite_parameters, self.radar_parameters, dt = dt, maxIter = maxIter, solver = 'RK45', simple_solver = False, simple_radar = simple_radar, rotating_earth=self.rot_earth_flag.isChecked())

        self.run_predictor()
    
    def run_predictor(self):
        """
        Runs the predictor algorithm.

        This method runs the predictor algorithm based on the selected filter type and other parameters.
        It updates the predicted positions and covariance matrices and then calls the Handoff_3D method.

        Returns:
            None
        """
        self.loading_screen.label.setText("Running predictor...")
        QApplication.processEvents()

        #print("reached!")
        if self.filtertype_ukf.isChecked():
            filter_type = 'ukf'
        else:
            filter_type = 'kalman'

        fixed_earth = not self.rot_earth_flag.isChecked()
        self.predicted_positions, self.predicted_cov = warwick_pmsc_skylab.Predictor.Kalman.run_filter(filter_type, '3d', dt=eval(self.pred_dt_text.text()), reading_type=self.radar_parameters['reading type'], sat_initpos=self.satellite_parameters['initial position'], initial_time=self.satellite_parameters['time'], multilateration_number=3, fixed_earth=fixed_earth, radar_noise=eval(self.noiselevel_text.text()), process_noise=eval(self.process_noise_text.text()))

        self.loading_screen.close()
        self.Handoff_3D(self.poshist, self.althist, self.predicted_positions, self.predicted_cov)

    def Handoff_3D(self, poshist, althist, predicted_positions, predicted_cov):
        self.w = VisualizationWindow('3D', poshist, althist, predicted_positions, predicted_cov)
        self.w.show()
        self.close()


class MainWindow(QMainWindow):
    """
    The main window of the Skylab Orbit Predictor GUI.

    This window allows the user to choose between the 2D and 3D versions of the Skylab Orbit Predictor.
    """

    def __init__(self):
        super().__init__()
        self.w = None

        self.setWindowTitle("Skylab Orbit Predictor GUI")
        
        layout = QVBoxLayout()

        self.intro = QLabel("Welcome to the Warwick Predictive Modelling and Scientific Computing ES98B Skylab Group Project! \nPlease choose which model type you'd like to use:")
        layout.addWidget(self.intro)

        first_layout = QHBoxLayout()
        self.button_2D = QPushButton("2D Version")
        self.button_2D.clicked.connect(self.show_2D_window)
        self.button_3D = QPushButton("3D Version")
        self.button_3D.clicked.connect(self.show_3D_window)

        first_layout.addWidget(self.button_2D)
        first_layout.addWidget(self.button_3D)

        second_layout = QHBoxLayout()
        self.exitButton = QPushButton("Exit")
        self.exitButton.clicked.connect(self.exit)
        second_layout.addWidget(self.exitButton)

        layout.addLayout(first_layout)
        layout.addLayout(second_layout)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
    
    def exit(self):
        """
        Exit the application and close the main window.
        """
        QApplication.quit()
        self.hide()
    
    def show_2D_window(self, checked):
        """
        Show the 2D window and close the main window.
        """
        self.w = Window_2D()
        self.w.show()
        self.close()
    
    def show_3D_window(self, checked):
        """
        Show the 3D window and close the main window.
        """
        self.w = Window_3D()
        self.w.show()
        self.close()

class VisualizationWindow(QMainWindow):
    """
    A class representing a visualization window for simulation and prediction.

    Parameters:
    - model (str): The model used for prediction (either "2D" or "3D").
    - poshist (list): The history of positions during simulation.
    - althist (list): The history of altitudes during simulation.
    - predicted_positions (numpy.ndarray): The predicted positions.
    - predicted_cov (list): The predicted covariance matrices.

    Attributes:
    - title (str): The title of the visualization window.
    - model (str): The model used for prediction.
    - poshist (list): The history of positions during simulation.
    - althist (list): The history of altitudes during simulation.
    - predicted_positions (numpy.ndarray): The predicted positions.
    - predicted_cov (list): The predicted covariance matrices.
    - figure (matplotlib.figure.Figure): The figure object for the plot.
    - canvas (matplotlib.backends.backend_qt5agg.FigureCanvasQTAgg): The canvas for the plot.
    - toolbar (matplotlib.backends.backend_qt5agg.NavigationToolbar2QT): The toolbar for the plot.
    - ax (matplotlib.axes._subplots.AxesSubplot): The axes object for the plot.
    - back_button (PyQt5.QtWidgets.QPushButton): The button to go back to the main window.

    Methods:
    - initUI(): Initializes the user interface of the visualization window.
    - eventFilter(source, event): Filters events for the visualization window.
    - zoom_2d_plot(delta): Zooms the 2D plot.
    - zoom_3d_plot(delta): Zooms the 3D plot.
    - go_back(): Closes the visualization window and goes back to the main window.
    - get_error_ellipse_2d(cov, pos, nsig=1): Calculates the error ellipse for a 2D position.
    - get_error_ellipsoid_3d(cov, pos, nsig=1): Calculates the error ellipsoid for a 3D position.
    - draw_2d_plot(): Draws the 2D plot.
    - draw_3d_plot(): Draws the 3D plot.
    """
    def __init__(self, model, poshist, althist, predicted_positions, predicted_cov, init_time, sim_timestep, pred_timestep):
        super().__init__()
        self.title = 'Simulation and Prediction Visualization'
        self.model = model
        self.poshist = poshist
        self.althist = althist
        self.predicted_positions = predicted_positions
        self.predicted_cov = predicted_cov
        self.init_time = init_time
        self.sim_runtime = sim_timestep * (len(poshist)-1)
        self.pred_runtime = pred_timestep * (len(poshist)-1)
        self.initUI()

    def initUI(self):
        """
        Initializes the user interface for the application.

        This method sets up the window title, geometry, and layout. It creates a figure and canvas for plotting,
        adds a toolbar for navigation, and displays crash predictions from the simulator and predictor models.

        Args:
            None

        Returns:
            None
        """
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 800, 600)

        widget = QWidget(self)
        self.setCentralWidget(widget)
        layout = QVBoxLayout()
        widget.setLayout(layout)

        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        self.toolbar = NavigationToolbar(self.canvas, self)
        layout.addWidget(self.toolbar)

        val_layout = QVBoxLayout()

        simulator_vals = QHBoxLayout()
        simulator_crash_pos = QLabel()
        simulator_crash_pos.setText(f"Simulator predicted crash at location: {self.poshist[-1]}")
        simulator_crash_time = QLabel()
        simulator_crash_time.setText(f"Simulator predicted crash at time: {self.init_time + datetime.timedelta(self.sim_runtime)}")
        simulator_vals.addWidget(simulator_crash_pos)
        simulator_vals.addWidget(simulator_crash_time)

        predictor_vals = QHBoxLayout()
        predictor_crash_pos = QLabel()
        if self.model == "3D":
            crash_pos = self.predicted_positions[-1, [0, 2, 4]]
        else:
            crash_pos = self.predicted_positions[[0, 2], -1]
        predictor_crash_pos.setText(f"Predictor predicted crash at location: {crash_pos}")
        predictor_crash_time = QLabel()
        predictor_crash_time.setText(f"Predictor predicted crash at time: {self.init_time + datetime.timedelta(self.pred_runtime)}")
        predictor_vals.addWidget(predictor_crash_pos)
        predictor_vals.addWidget(predictor_crash_time)

        val_layout.addLayout(simulator_vals)
        val_layout.addLayout(predictor_vals)

        layout.addLayout(val_layout)

        self.back_button = QPushButton("Home", self)
        self.back_button.setFixedSize(100, 30)
        self.back_button.clicked.connect(self.go_back)
        layout.addWidget(self.back_button)

        if self.model == "3D":
            self.ax = self.figure.add_subplot(111, projection='3d')
            self.draw_3d_plot()
        else:
            self.ax = self.figure.add_subplot(111)
            self.draw_2d_plot()

        plt.close()

        self.installEventFilter(self)

    def eventFilter(self, source, event):
        if event.type() == QEvent.Wheel:
            delta = event.angleDelta().y()
            if self.model == "3D":
                self.zoom_3d_plot(delta)
            else:
                self.zoom_2d_plot(delta)
            return True
        return super().eventFilter(source, event)

    def zoom_2d_plot(self, delta):
        base_scale = 1.1
        if delta > 0:
            scale_factor = base_scale
        else:
            scale_factor = 1 / base_scale
        self.ax.set_xlim(self.ax.get_xlim()[0] * scale_factor, self.ax.get_xlim()[1] * scale_factor)
        self.ax.set_ylim(self.ax.get_ylim()[0] * scale_factor, self.ax.get_ylim()[1] * scale_factor)
        self.canvas.draw_idle()

    def zoom_3d_plot(self, delta):
        base_scale = 1.1
        if delta > 0:
            scale_factor = base_scale
        else:
            scale_factor = 1 / base_scale
        self.ax.set_xlim(self.ax.get_xlim()[0] * scale_factor, self.ax.get_xlim()[1] * scale_factor)
        self.ax.set_ylim(self.ax.get_ylim()[0] * scale_factor, self.ax.get_ylim()[1] * scale_factor)
        self.ax.set_zlim(self.ax.get_zlim()[0] * scale_factor, self.ax.get_zlim()[1] * scale_factor)
        self.canvas.draw_idle()

    def go_back(self):
        self.w = MainWindow()
        self.w.show()
        self.close()

    def get_error_ellipse_2d(self, cov, pos, nsig=1):
        """
        Returns an Ellipse object representing the error ellipse for a 2D Gaussian distribution.

        Parameters:
            cov (ndarray): The covariance matrix of the Gaussian distribution.
            pos (tuple): The center position of the ellipse.
            nsig (float): The number of standard deviations to include in the ellipse (default: 1).

        Returns:
            Ellipse: An Ellipse object representing the error ellipse.

        """
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height = 2 * nsig * np.sqrt(eigvals)
        return patches.Ellipse(xy=pos, width=width, height=height, angle=angle, edgecolor='r', fc='None', lw=2)

    def get_error_ellipsoid_3d(self, cov, pos, nsig=1):
        """
        Calculate the points of an error ellipsoid in 3D space.

        Parameters:
        - cov: numpy.ndarray
            The covariance matrix of the ellipsoid.
        - pos: numpy.ndarray
            The position of the ellipsoid in 3D space.
        - nsig: int, optional
            The number of standard deviations to use for scaling the ellipsoid.
            Default is 1.

        Returns:
        - x_ellip: numpy.ndarray
            The x-coordinates of the ellipsoid points.
        - y_ellip: numpy.ndarray
            The y-coordinates of the ellipsoid points.
        - z_ellip: numpy.ndarray
            The z-coordinates of the ellipsoid points.
        """
        eigvals, eigvecs = np.linalg.eigh(np.array(cov))
        order = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]

        # Calculate the ellipsoid points
        phi = np.linspace(0, 2 * np.pi, 100)
        theta = np.linspace(0, np.pi, 50)
        phi, theta = np.meshgrid(phi, theta)

        # Cartesian coordinates for the ellipsoid
        x = nsig * np.sqrt(eigvals[0]) * np.sin(theta) * np.cos(phi)
        y = nsig * np.sqrt(eigvals[1]) * np.sin(theta) * np.sin(phi)
        z = nsig * np.sqrt(eigvals[2]) * np.cos(theta)

        # Rotate the ellipsoid to align with the eigenvectors
        ellipsoid_points = np.stack((x.flatten(), y.flatten(), z.flatten()), axis=1)
        ellipsoid_points = ellipsoid_points @ eigvecs.T

        # Translate to the specified position
        ellipsoid_points = ellipsoid_points + pos

        # Reshape to the meshgrid shape
        x_ellip = ellipsoid_points[:, 0].reshape(theta.shape)
        y_ellip = ellipsoid_points[:, 1].reshape(theta.shape)
        z_ellip = ellipsoid_points[:, 2].reshape(theta.shape)

        return x_ellip, y_ellip, z_ellip

    def draw_2d_plot(self):
        poshist = np.array(self.poshist)
        pred_positions = np.array(self.predicted_positions)
        print(pred_positions.shape)
        covariances = self.predicted_cov

        self.ax.plot(poshist[:, 0], poshist[:, 1], 'b-', label='Simulated Path')

        x_errors = np.sqrt([cov[0, 0] for cov in covariances])  # Square root of variance for x
        y_errors = np.sqrt([cov[2, 2] for cov in covariances])  # Square root of variance for y

        x_positions = pred_positions[0, :]
        y_positions = pred_positions[2, :]
        self.ax.errorbar(x_positions[1:], y_positions[1:], xerr=x_errors, yerr=y_errors, color='red', linestyle='--', ecolor='lightgray', elinewidth=9, capsize=0, label='Predicted Positions')

        earth = plt.Circle((0, 0), 6371, color='blue', label='Earth')
        self.ax.add_patch(earth)

        # for i in range(len(pred_positions)):
        #     cov = covariances[np.ix_([0, 2], [0, 2])][i]
        #     ellipse = self.get_error_ellipse_2d(cov, pred_positions[:,i])
        #     self.ax.add_patch(ellipse)

        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_title('Satellite Trajectory and Prediction')
        self.canvas.draw()

    def draw_3d_plot(self):
        print("Drawing Plot")
        poshist = np.array(self.poshist)
        pred_positions = np.array(self.predicted_positions[:, [0, 2, 4]])
        covariances = self.predicted_cov
        img = imread('world_map.jpg')
        img = img[::2, ::2]  # Reduces the resolution by a factor of 2
        x_ellipsoid, y_ellipsoid, z_ellipsoid = warwick_pmsc_skylab.Simulator.earth_ellipsoid

        fig = plt.figure()
        plt.axis('off')
        
        # Map the texture to the ellipsoid
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 50)
        u, v = np.meshgrid(u, v)
        img = img / 255.0  # Normalize the image to the range [0, 1]

        # Create the ellipsoid with texture
        self.ax.plot_surface(x_ellipsoid, y_ellipsoid, z_ellipsoid, rstride=5, cstride=5, facecolors=img, linewidth=0, antialiased=False)

        line, = self.ax.plot(poshist[:, 0], poshist[:, 1], poshist[:, 2])

        self.ax.view_init(elev=0, azim=0)

        def update(num, poshist, line):
            line.set_data(np.array([poshist[:num, 0], poshist[:num, 1]]))
            line.set_3d_properties(np.array(poshist[:num, 2]))

        N = np.arange(0, len(poshist), 100).tolist()
        N.append(len(poshist) - 1)
        N = iter(tuple(N))

        plt.axis('off')
        #ani = animation.FuncAnimation(fig, update, N, fargs=(poshist, line), cache_frame_data=False, interval=100, blit=False)
        #ani.save('SatelliteCrash.gif', writer='pillow')
        plt.subplots_adjust(wspace=0.9)
        plt.axis('off')

        self.ax.plot(poshist[:, 0], poshist[:, 1], poshist[:, 2], 'b-', label='Simulated Path')
        self.ax.plot(pred_positions[:, 0], pred_positions[:, 1], pred_positions[:, 2], 'r--', label='Predicted Path')

        # Extract the last state and covariance for the heat map
        x_last = self.predicted_positions[-1][[0, 2]]
        P_last = covariances[-1][np.ix_([0, 2], [0, 2])]

        # Generate grid of points for heat map in the XY plane at the last Z position
        z_last = self.predicted_positions[-1, 4]
        x_grid, y_grid = np.mgrid[x_last[0] - 50:x_last[0] + 50:100j, x_last[1] - 50:x_last[1] + 50:100j]
        pos = np.dstack((x_grid, y_grid))
        rv = multivariate_normal(x_last, P_last)
        # Create a flattened array of Z values
        z_values = np.full(pos.shape[:-1], z_last)
        # Use contourf to plot heat map at the last Z position
        self.ax.contourf(x_grid, y_grid, z_values, rv.pdf(pos), levels=50, cmap='viridis', offset=z_last)

        for i in range(len(pred_positions)):
            cov = covariances[i][np.ix_([0, 2, 4], [0, 2, 4])]
            x_ellip, y_ellip, z_ellip = self.get_error_ellipsoid_3d(cov, pred_positions[i])
            self.ax.plot_wireframe(x_ellip, y_ellip, z_ellip, color='r', alpha=0.3)

        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_zlabel('Z Position')
        self.ax.set_title('3D Satellite Trajectory and Prediction')
        self.ax.legend()
        self.ax.set_axis_off()
        self.canvas.draw()

        plt.close()


def run_GUI(window_class=MainWindow):
    app = 0
    app = QApplication(sys.argv)
    w = window_class()
    w.show()

    sys.exit(app.exec_())


# In[11]:


#run_GUI()


# In[ ]:




