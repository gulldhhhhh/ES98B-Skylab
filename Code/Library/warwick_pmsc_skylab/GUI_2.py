#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys
import warwick_pmsc_skylab.Simulator
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QLineEdit, QRadioButton, QGroupBox, QHBoxLayout, QDateTimeEdit, QMainWindow, QCheckBox, QButtonGroup
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


# In[3]:


class Window_2D(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D Model Input")
        layout = QVBoxLayout()

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

        layout.addWidget(ellipsebox)


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

        layout.addWidget(satellitebox)


        #Radar Section
        radarbox = QGroupBox("Radar Parameters")
        radar_layout = QVBoxLayout()

        self.radarcomplexity = QHBoxLayout()
        self.radiosimple = QRadioButton("Simple Radar")
        self.radiocomplex = QRadioButton("Complex Radar")
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


        #Simulator Parameters
        simulatorbox = QGroupBox("Simulator Parameters")
        simulator_layout = QVBoxLayout()

        self.steptime_layout = QHBoxLayout()
        self.steptime_layout.addWidget(QLabel("Time step dt (s):"))
        self.steptime_text = QLineEdit("0.1")
        self.steptime_layout.addWidget(self.steptime_text)

        self.maxiter_layout = QHBoxLayout()
        self.maxiter_layout.addWidget(QLabel("Maximum Iterations:"))
        self.maxiter_text = QLineEdit("1000000")
        self.maxiter_layout.addWidget(self.maxiter_text)

        self.simple_solver_layout = QHBoxLayout()
        self.simple_solver = QCheckBox("Simple Forward Euler Solver (Not Recommended)")
        self.simple_solver_layout.addWidget(self.simple_solver)

        simulator_layout.addLayout(self.steptime_layout)
        simulator_layout.addLayout(self.maxiter_layout)
        simulator_layout.addLayout(self.simple_solver_layout)
        simulatorbox.setLayout(simulator_layout)
        
        layout.addWidget(simulatorbox)

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
        if self.width_text.text() != "" and self.height_text != "" and self.centre_text != "":
            self.initpos_text.setText(f"[{((eval(self.width_text.text()) - eval(self.centre_text.text())[0])) + 0.1 * eval(self.width_text.text())}, {((eval(self.height_text.text()) - eval(self.centre_text.text())[1])) + 0.1 * eval(self.height_text.text())}]")
        else:
            self.initpos_text.setText("[100,100]")

    def tangent_velocity(self):
        if self.tangent_velocity_flag.isChecked():
            self.initveloc_norm_text.setText("")
            self.initveloc_norm_text.setEnabled(False)
        else:
            self.initveloc_norm_text.setEnabled(True)
    
    def run_simulator(self):
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
        
        dt = eval(self.steptime_text.text())
        maxIter = eval(self.maxiter_text.text())
        simple_solver = self.simple_solver.isChecked()
        simple_radar = self.radiosimple.isChecked()

        poshist, althist = warwick_pmsc_skylab.Simulator.Simulator_2D(ellipse_parameters, satellite_parameters, radar_parameters, dt = dt, maxIter = maxIter, solver = 'RK45', simple_solver = simple_solver, simple_radar = simple_radar)

        self.Handoff_2D(poshist, althist)

    def Handoff_2D(self, poshist, althist):
        self.w = VisualizationWindow('2D', poshist, althist)
        self.w.show()
        self.close()




class Window_3D(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D Model Input")
        layout = QVBoxLayout()

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

        layout.addWidget(satellitebox)


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

        layout.addWidget(radarbox)


        #Simulator Parameters
        simulatorbox = QGroupBox("Simulator Parameters")
        simulator_layout = QVBoxLayout()

        self.steptime_layout = QHBoxLayout()
        self.steptime_layout.addWidget(QLabel("Time step dt (s):"))
        self.steptime_text = QLineEdit("0.1")
        self.steptime_layout.addWidget(self.steptime_text)

        self.maxiter_layout = QHBoxLayout()
        self.maxiter_layout.addWidget(QLabel("Maximum Iterations:"))
        self.maxiter_text = QLineEdit("1000000")
        self.maxiter_layout.addWidget(self.maxiter_text)

        self.dragtype = QHBoxLayout()
        self.dragtype_simple = QRadioButton("Simple Atmospheric Model")
        self.dragtype_complex = QRadioButton("Complex Atmospheric Model")
        self.dragtype_complex.setChecked(True)
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
        
        layout.addWidget(simulatorbox)

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
    
    def run_simulator(self):

        satellite_parameters = {
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
        radar_parameters = {
            'radar parameter': eval(self.radarparam_text.text()),
            'reading type': rtype,
            'noise level (%)': eval(self.noiselevel_text.text()),
            'reading_interval': eval(self.readint_text.text())
        }
        
        dt = eval(self.steptime_text.text())
        maxIter = eval(self.maxiter_text.text())
        simple_radar = self.radiosimple.isChecked()

        poshist, althist = warwick_pmsc_skylab.Simulator.Simulator(satellite_parameters, radar_parameters, dt = dt, maxIter = maxIter, solver = 'RK45', simple_solver = False, simple_radar = simple_radar)

        #PREDICTOR TO BE ADDED HERE

        self.Handoff_3D(poshist, althist)
    
    def Handoff_3D(self, poshist, althist):
        self.w = VisualizationWindow('3D', poshist, althist)
        self.w.show()
        self.close()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.w = None

        self.setWindowTitle("Skylab Orbit Predictor GUI")
        
        layout = QVBoxLayout()

        self.button_2D = QPushButton("2D Version")
        self.button_2D.clicked.connect(self.show_2D_window)
        self.button_3D = QPushButton("3D Version")
        self.button_3D.clicked.connect(self.show_3D_window)
        
        layout.addWidget(self.button_2D)
        layout.addWidget(self.button_3D)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)
    
    def show_2D_window(self, checked):
        self.w = Window_2D()
        self.w.show()
        self.close()
    
    def show_3D_window(self, checked):
        self.w = Window_3D()
        self.w.show()
        self.close()

class VisualizationWindow(QMainWindow):
    def __init__(self, model, poshist, althist):
        super().__init__()
        self.title = 'Simulation and Prediction Visualization'
        self.model = model
        self.poshist = poshist
        self.althist = althist
        self.initUI()

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 800, 600)
        
        widget = QWidget(self)
        self.setCentralWidget(widget)
        layout = QVBoxLayout()
        widget.setLayout(layout)
        
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)

        if self.model == "3D":
            self.ax = self.figure.add_subplot(111, projection='3d')
            self.draw_3d_plot()
        else:
            self.ax = self.figure.add_subplot(111)
            self.draw_2d_plot()

    def draw_plot_2d(self):
        sim_data = pd.read_csv('simulator_2D.csv')
        pred_data = pd.read_csv('predictor_2D.csv')

        earth = plt.Circle((0, 0), 1, color='blue', label='Earth')
        self.ax.add_patch(earth)

        self.ax.plot(sim_data['x'], sim_data['y'], 'ro-', label='Satellite Path')
        
        self.ax.plot(pred_data['x'], pred_data['y'], 'go--', label='Predicted Path')

        self.ax.legend()
        self.ax.grid(True)
        self.ax.set_xlabel('X Coordinate')
        self.ax.set_ylabel('Y Coordinate')
        self.ax.set_title('Satellite Trajectory and Prediction')
        self.canvas.draw()

    def draw_3d_plot(self):

        poshist = np.array(self.poshist)

        fig = plt.figure()
        plt.axis('off')
        self.ax.plot_surface(warwick_pmsc_skylab.Simulator.earth_ellipsoid[0],warwick_pmsc_skylab.Simulator.earth_ellipsoid[1],warwick_pmsc_skylab.Simulator.earth_ellipsoid[2], alpha = 0.3)
        line, = self.ax.plot(poshist[:,0],poshist[:,1],poshist[:,2])

        #ax1.view_init(elev = init_elev + np.pi/2, azim = final_azmth + np.pi/4)
        self.ax.view_init(elev = 0, azim = 0)

        def update(num, poshist, line):
            line.set_data(np.array([poshist[:num,0], poshist[:num,1]]))
            line.set_3d_properties(np.array(poshist[:num,2]))

        N = np.arange(0,len(poshist),100).tolist()
        N.append(len(poshist)-1)
        N = iter(tuple(N))
        plt.axis('off')
        ani = animation.FuncAnimation(fig, update, N, fargs = (poshist, line), cache_frame_data=True, interval = 100, blit=False)
        ani.save('SatelliteCrash.gif', writer='pillow')
        plt.subplots_adjust(wspace = 0.9)
        plt.axis('off')
        #plt.close()

def run_GUI(window_class=MainWindow):
    app = 0
    print("Test!")
    app = QApplication(sys.argv)
    print(QApplication.instance())
    w = window_class()
    w.show()
    print(QApplication.instance())
    sys.exit(app.exec_())


# In[4]:


# In[ ]:



