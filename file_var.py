from PyQt5.QtWidgets import QApplication, QMainWindow, QFileSystemModel, QFileDialog
from PyQt5.QtCore import QDir, QModelIndex, Qt, QStandardPaths
import pandas as pd
import os
import sys
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np


class VarData:
    def __init__(self):

        self.y_raw = np.array([])    # init data
        self.y_plot = np.array([])   # plot data

        self.y_analyzed = np.array([])     # analyzed data
        self.nor_TSMA = np.array([])       # output of first step
        self.mb_sos = np.array([])     # output of second step
        self.baseline = np.array([])   # output of second step
        self.absorption_profile = np.array([])  # output of 3-th step
        self.average_peak_data = np.array([])   # output of 4-th step
        self.area_under_curve = 0               # output of 5-th step
        self.amplitude_ab = 0                   # output of 5-th step
        self.time_centroi = 0  # output of 5-th step
        self.fre_centroi = 0  # output of 5-th step

        self.x = np.array([])
        # =============================================================== frequencies
        self.frequencies_plot = np.ndarray([])  # plot data
        self.abs_spectrum_plot = np.ndarray([])

        # ===============================================================

        self.step_number = None
        # ===============================================================
        self.f0 = 10000
        self.fs = 320000
        self.maxM = None
        self.maxK = int(self.fs // (2*self.f0))
        self.max_analyzed_length = None
        # ===============================================================
        self.time = np.zeros(10)

        self.max_width_imf = None






