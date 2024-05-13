#Universal Constants
from warwick_pmsc_skylab.GUI_2 import MainWindow
import sys
from PyQt5.QtWidgets import QApplication

radius_polar = 6356752
radius_equatorial = 6378137
earth_eccentricity_squared = 6.694379e-3
M_e = 5.972e24
G = 6.673e-11
rho_0 = 1.225
H = 8400.0
stellar_day  = 86164.1
earth_rotation_s = 1/360 * stellar_day

def run_GUI(window_class=MainWindow):
    #app = 0
    app = QApplication(sys.argv)
    w = window_class()
    w.show()
    sys.exit(app.exec_())