from PyQt5.QtWidgets import QApplication, QMainWindow, QFileSystemModel, QFileDialog, QMessageBox, QWidget
from PyQt5.QtCore import QDir, QModelIndex, Qt, QStandardPaths
import pandas as pd
import os
import sys
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray

from New_window import Ui_MainWindow
from file_var import *

import funtion_pulse as pu_f


class FigurePlot(QMainWindow):
    def __init__(self, self_instance):
        super(FigurePlot, self).__init__()
        self.ui: Ui_MainWindow = self_instance.ui
        self.var: VarData = self_instance.var

        self.ui.horizontalScrollBar_x.valueChanged.connect(
            lambda: self.callback_update_time_plot(self.var.x, self.var.y_plot))
        self.ui.spinBox_x.valueChanged.connect(
            lambda: self.callback_update_time_plot(self.var.x, self.var.y_plot))

        self.ui.horizontalScrollBar_f.valueChanged.connect(
            lambda: self.callback_update_frequency_plot(self.var.frequencies_plot, self.var.abs_spectrum_plot))
        self.ui.spinBox_f.valueChanged.connect(
            lambda: self.callback_update_frequency_plot(self.var.frequencies_plot, self.var.abs_spectrum_plot))

        self.ui.doubleSpinBox_f0.valueChanged.connect(lambda: self.callback_update_f0())
        self.ui.doubleSpinBox_fs.valueChanged.connect(lambda: self.callback_update_fs())
        self.ui.spinBox_M.valueChanged.connect(lambda: self.callback_update_m())
        self.ui.spinBox_window_length.valueChanged.connect(lambda: self.callback_update_window_length())
        self.ui.spinBox_K_subband.valueChanged.connect(lambda: self.callback_update_k())
        self.ui.spinBox_order.valueChanged.connect(lambda: self.callback_order())
        self.ui.doubleSpinBox_bwidth.valueChanged.connect(lambda: self.callback_bwidth_filter())
        self.ui.spinBox_bwidth_pulse.valueChanged.connect(lambda: self.callback_bwidth_pulse())
        self.ui.spinBox_samp_period.valueChanged.connect(lambda: self.callback_samp_period())

        self.ui.pushButton_information.clicked.connect(self.show_information)
        self.ui.pushButton_save_graph.clicked.connect(self.save_graph)
        self.ui.pushButton_save_signal.clicked.connect(self.save_signal)

        self.var.max_width_imf = self.ui.frame_information.maximumWidth()
        self.max_oder = 15

        # ============================================================ local var
        # Initialize a Figure and Axes for Matplotlib
        self.fig_time, self.axes_time = plt.subplots(nrows=1, ncols=1)
        # Create a canvas to integrate Matplotlib with PyQt
        self.canvas_time = FigureCanvas(self.fig_time)
        # Initialize a Figure and Axes for Matplotlib
        self.fig_fre, self.axes_fre = plt.subplots(nrows=1, ncols=1)
        # Create a canvas to integrate Matplotlib with PyQt
        self.canvas_fre = FigureCanvas(self.fig_fre)

        self.maxspinBox_x = self.ui.spinBox_x.value()
        self.maxspinBox_f = self.ui.spinBox_f.value()
        # ============================================================
        self.init_window(self_instance)

    def init_window(self, self_instance):

        self.openfile_window()
        self.ui.label_name_signal_2.setText("Name of signal")
        self.ui.label_length_signal.setText("None")
        self.ui.label_length_sample.setText("None")

        # Change the text size for MainWindow
        font = self_instance.font()
        min_point_size = 10
        if font.pointSize() < min_point_size:
            font.setPointSize(min_point_size)
        self_instance.setFont(font)

        self.ui.label_f0.setText(f"{self.var.f0}")
        self.ui.label_fs.setText(f"{self.var.fs}")

        # update window
        self.ui.pushButton_next.setEnabled(False)

        # add to time graph
        self.ui.time_graph.addWidget(self.canvas_time)
        self.axes_time.margins(1)
        self.axes_time.spines['top'].set_visible(False)
        self.axes_time.spines['right'].set_visible(False)

        # add to frequency graph
        self.ui.frequency_graph.addWidget(self.canvas_fre)
        self.axes_fre.margins(1)
        self.axes_fre.spines['top'].set_visible(False)
        self.axes_fre.spines['right'].set_visible(False)

        # set spinBox and horizontalScrollBar time graph
        self.ui.horizontalScrollBar_x.setMinimum(0)
        self.ui.horizontalScrollBar_x.setEnabled(False)
        self.ui.spinBox_x.setEnabled(False)

        # set spinBox and horizontalScrollBar frequency graph
        self.ui.horizontalScrollBar_f.setMinimum(0)
        self.ui.horizontalScrollBar_f.setEnabled(False)
        self.ui.spinBox_f.setEnabled(False)

        # init signal window
        self.ui.doubleSpinBox_f0.setEnabled(False)
        self.ui.doubleSpinBox_fs.setEnabled(False)
        self.ui.spinBox_window_length.setEnabled(False)
        self.ui.spinBox_M.setEnabled(False)
        self.ui.comboBox_nor.setEnabled(False)
        self.ui.radioButton_padding.setEnabled(False)
        self.ui.spinBox_padding.setEnabled(False)
        self.ui.pushButton_apply.setEnabled(False)
        self.ui.spinBox_samp_period.setEnabled(False)

        self.ui.stackedWidget_setup.setCurrentIndex(0)
        self.ui.stackedWidget_back.setCurrentIndex(0)

        # clear graph
        self.ui.label_time.setText("Time domain")
        self.ui.label_frequency.setText("Frequency domain")
        self.axes_time.clear()
        self.canvas_time.draw()
        self.axes_fre.clear()
        self.canvas_fre.draw()

    def save_signal(self):
        if self.var.step_number is None:
            print("No signal data available.")
            return

        # Get the filename/path using QFileDialog
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(self, 'Save Time Signal', '',
                                                   'CSV Files (*.csv);;All Files (*)')

        if file_path:  # If user selected a file
            if file_path.endswith('.csv'):
                df = pd.DataFrame({'Signal': self.var.y_plot})  # theo cột
                df.to_csv(file_path, header=True, index=False)

    def save_graph(self):
        if self.var.step_number is None:
            print("No graph available.")
            return

        # Get the filename/path using QFileDialog
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getSaveFileName(self, 'Save Time Figure', '',
                                                   'PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)')

        if file_path:  # If user selected a file
            if file_path.endswith('.png'):
                self.fig_time.savefig(file_path, format='png')
            elif file_path.endswith('.pdf'):
                self.fig_time.savefig(file_path, format='pdf')

        # ==================================================================
        file_path, _ = file_dialog.getSaveFileName(self, 'Save Frequency Figure', '',
                                                   'PNG Files (*.png);;PDF Files (*.pdf);;All Files (*)')

        if file_path:  # If user selected a file
            if file_path.endswith('.png'):
                self.fig_fre.savefig(file_path, format='png')
            elif file_path.endswith('.pdf'):
                self.fig_fre.savefig(file_path, format='pdf')

    def show_information(self):
        # toggle Width
        current_width = self.ui.frame_information.maximumWidth()
        if current_width != 0:
            new_width = 0
            self.ui.pushButton_information.setText("<<")
        else:
            new_width = self.var.max_width_imf
            self.ui.pushButton_information.setText(">>")
        self.ui.frame_information.setFixedWidth(new_width)

    def openfile_window(self):
        # reset led
        self.ui.pushButton_led_1.setStyleSheet('background-color: gray; border: none')
        self.ui.pushButton_led_2.setStyleSheet('background-color: gray; border: none')
        self.ui.pushButton_led_3.setStyleSheet('background-color: gray; border: none')
        self.ui.pushButton_led_4.setStyleSheet('background-color: gray; border: none')
        self.ui.pushButton_led_5.setStyleSheet('background-color: gray; border: none')
        self.ui.pushButton_led_6.setStyleSheet('background-color: gray; border: none')

        # reset Information of signal
        self.ui.label_SC.setText("None")
        self.ui.label_THD.setText("None")
        self.ui.label_ROE.setText("None")
        self.ui.label_HNR.setText("None")
        self.ui.label_time_classify.setText("None")
        self.ui.label_type_signal.setText("Types of signal")
        self.ui.label_HNR_b.setText("None")
        self.ui.label_HNR_a.setText("None")
        self.ui.label_amp.setText("None")
        self.ui.label_area.setText("None")
        self.ui.label_time_centr.setText("None")
        self.ui.label_fre_centr.setText("None")
        self.ui.label_time_comp.setText("None")
        self.ui.label_amp_c.setText("None")
        self.ui.label_area_c.setText("None")
        self.ui.label_time_centr_c.setText("None")
        self.ui.label_fre_centr_c.setText("None")
        self.ui.label_laser_power.setText("None")

        # active and reset signal window
        self.ui.pushButton_next.setEnabled(True)
        self.ui.label_time.setText("Raw signal")
        self.ui.label_frequency.setText("Amplitude spectrum")
        self.ui.spinBox_M.setValue(50)

        self.ui.spinBox_window_length.setEnabled(True)
        self.ui.doubleSpinBox_f0.setEnabled(True)
        self.ui.doubleSpinBox_fs.setEnabled(True)
        self.ui.spinBox_M.setEnabled(True)
        self.ui.comboBox_nor.setEnabled(True)
        self.ui.spinBox_samp_period.setEnabled(True)

        self.ui.radioButton_padding.setEnabled(True)

        self.ui.stackedWidget_setup.setCurrentIndex(0)
        self.ui.stackedWidget_back.setCurrentIndex(0)

    def callback_samp_period(self):
        self.ui.doubleSpinBox_fs.setValue(self.var.f0 * self.ui.spinBox_samp_period.value())

    def callback_order(self):
        if self.ui.spinBox_order.value() > self.max_oder:  # if va5lue > len(data) -> = len(data)
            self.ui.spinBox_order.setValue(self.max_oder)
            QMessageBox.warning(self, 'Warning', f'Maximum order of filter = {self.max_oder}!')

    def callback_update_f0(self):
        # update var
        self.var.f0 = self.ui.doubleSpinBox_f0.value()

        # update Information of signal
        self.ui.label_f0.setText(f"{self.var.f0}")
        num_samp_period = int(self.var.fs // self.var.f0)
        all_num_period = int(len(self.var.y_raw) // num_samp_period)
        self.ui.label_length_signal.setText(f"{all_num_period}")

        # update K subband filter
        self.var.maxK = int(self.var.fs // (2 * self.var.f0)) - 1
        self.ui.spinBox_K_subband.setValue(self.var.maxK)

        # update Analyzed window length(cycle)
        self.ui.spinBox_window_length.setValue(500)

        # update bandwidth of pulse
        max_bwidth_pulse = int(self.var.fs / self.var.f0)
        self.ui.spinBox_bwidth_pulse.setValue(max_bwidth_pulse)

        # # update sample per period
        # self.ui.spinBox_samp_period.setValue(self.var.fs // self.var.f0)

        # update graph
        self.ui.spinBox_x.setValue(10 * (self.var.fs // self.var.f0))

    def callback_update_fs(self):
        # update var
        self.var.fs = self.ui.doubleSpinBox_fs.value()

        # update Information of signal
        self.ui.label_fs.setText(f"{self.var.fs}")
        num_samp_period = int(self.var.fs // self.var.f0)
        all_num_period = int(len(self.var.y_raw) // num_samp_period)
        self.ui.label_length_signal.setText(f"{all_num_period}")

        # update K subband filter
        self.var.maxK = int(self.var.fs // (2 * self.var.f0)) - 1
        self.ui.spinBox_K_subband.setValue(self.var.maxK)

        # update Analyzed window length(cycle)
        self.ui.spinBox_window_length.setValue(500)

        # update bandwidth of pulse
        max_bwidth_pulse = int(self.var.fs / self.var.f0)
        self.ui.spinBox_bwidth_pulse.setValue(max_bwidth_pulse)

        # update sample per period
        self.ui.spinBox_samp_period.setValue(self.var.fs // self.var.f0)

        # update graph
        self.ui.spinBox_x.setValue(10 * (self.var.fs // self.var.f0))

    def callback_update_window_length(self):
        self.var.max_analyzed_length = int(len(self.var.y_raw) * (self.var.f0 / self.var.fs))
        if self.ui.spinBox_window_length.value() > self.var.max_analyzed_length:  # if va5lue > len(data) -> = len(data)
            self.ui.spinBox_window_length.setValue(self.var.max_analyzed_length)
            QMessageBox.warning(self, 'Warning',
                                f'Recommended analyzed window length > 500 period. \n'
                                f'In this case maximum analyzed window length (period) = {self.var.max_analyzed_length}!')

    def callback_update_m(self):
        # update M
        self.var.maxM = 300
        if self.ui.spinBox_M.value() > self.var.maxM:  # if va5lue > len(data) -> = len(data)
            self.ui.spinBox_M.setValue(self.var.maxM)
            if self.var.step_number == 1:
                QMessageBox.warning(self, 'Warning', f'Maximum of M = {self.var.maxM}!')

    def callback_update_k(self):
        # update K subband filter
        self.var.maxK = int(self.var.fs // (2 * self.var.f0))
        if self.ui.spinBox_K_subband.value() > self.var.maxK:  # if va5lue > len(data) -> = len(data)
            self.ui.spinBox_K_subband.setValue(self.var.maxK)
            QMessageBox.warning(self, 'Warning', f'Maximum of K subband = {self.var.maxK}!')

    def callback_bwidth_filter(self):
        if self.ui.doubleSpinBox_bwidth.value() > self.var.f0:  # if va5lue > len(data) -> = len(data)
            self.ui.doubleSpinBox_bwidth.setValue(self.var.f0)
            QMessageBox.warning(self, 'Warning', f'Maximum bandwidth of filter = {self.var.f0}!')

    def callback_bwidth_pulse(self):
        max_bwidth_pulse = int(self.var.fs / self.var.f0)
        if self.ui.spinBox_bwidth_pulse.value() > max_bwidth_pulse:  # if va5lue > len(data) -> = len(data)
            self.ui.spinBox_bwidth_pulse.setValue(max_bwidth_pulse)
            QMessageBox.warning(self, 'Warning', f'Maximum bandwidth of pulse = {max_bwidth_pulse}!')

    def update_plot(self, x=np.array([]), y=np.array([])):
        self.ui.spinBox_x.setEnabled(True)
        self.ui.spinBox_f.setEnabled(True)

        self.update_plot_time(x, y)
        self.update_plot_frequency(y)

    # ================================================================================================
    def update_plot_time(self, x=np.array([]), y=np.array([])):
        length_data = len(y)
        cycle_data = (length_data * self.var.f0) // self.var.fs
        self.maxspinBox_x = length_data

        if cycle_data < 10:  # cycle
            self.ui.spinBox_x.setValue(length_data)
            self.ui.horizontalScrollBar_x.setEnabled(False)
        else:
            self.ui.horizontalScrollBar_x.setEnabled(True)
            self.ui.spinBox_x.setValue(10 * self.ui.spinBox_samp_period.value())

        self.ui.horizontalScrollBar_x.setMaximum(length_data - self.ui.spinBox_x.value())

        scroll_value = self.ui.horizontalScrollBar_x.value()
        spinbox_value = self.ui.spinBox_x.value()

        start_window = scroll_value
        end_window = start_window + spinbox_value

        if end_window <= len(y):
            self.figure_time_plot(start_window, end_window, x, y)
        else:
            print("update_ time plot error")

    def callback_update_time_plot(self, x=np.array([]), y=np.array([])):
        if self.var.step_number is not None:
            if self.ui.spinBox_x.value() > self.maxspinBox_x:  # if value > len(data) -> = len(data)
                self.ui.spinBox_x.setValue(self.maxspinBox_x)

            self.ui.horizontalScrollBar_x.setMaximum(len(y) - self.ui.spinBox_x.value())

            scroll_value = self.ui.horizontalScrollBar_x.value()
            spinbox_value = self.ui.spinBox_x.value()

            if spinbox_value == len(y):
                self.ui.horizontalScrollBar_x.setEnabled(False)
            else:
                self.ui.horizontalScrollBar_x.setEnabled(True)

            start_window = scroll_value
            end_window = start_window + spinbox_value

            if end_window <= len(y):
                self.figure_time_plot(start_window, end_window, x, y)
            else:
                print("callback_update_time_plot error")

    def figure_time_plot(self, start_index=0, end_index=-1, x=np.array([]), y=np.array([])):
        # Clear the previous plot
        self.axes_time.clear()
        # Update data and plot the figure
        self.axes_time.plot(x[start_index:end_index], y[start_index:end_index], linewidth=0.6, color="k")
        self.axes_time.set_xlabel('Data point')

        # if self.var.step_number == 6:
        #     self.axes_time.set_title(
        #         f"Amplitude absorption pulse={np.round(self.var.amplitude_ab, 4)}, "
        #         f"Area under absorption pulse={np.round(self.var.area_under_curve, 4)}", fontsize=10)

        # Adjust the size of the FigureCanvas
        self.fig_time.tight_layout()
        # Redraw the canvas
        self.canvas_time.draw()

    # ================================================================================================
    def callback_update_frequency_plot(self, x=np.array([]), y=np.array([])):
        if self.var.step_number is not None:
            if self.ui.spinBox_f.value() > self.maxspinBox_f:
                self.ui.spinBox_f.setValue(self.maxspinBox_f)

            self.ui.horizontalScrollBar_f.setMaximum(len(y) - self.ui.spinBox_f.value())

            scroll_value = self.ui.horizontalScrollBar_f.value()
            spinbox_value = self.ui.spinBox_f.value()

            if spinbox_value == len(y):
                self.ui.horizontalScrollBar_f.setEnabled(False)
            else:
                self.ui.horizontalScrollBar_f.setEnabled(True)

            start_window = scroll_value
            end_window = start_window + spinbox_value

            if end_window <= len(y):
                self.figure_frequency_plot(start_window, end_window, x, y)
            else:
                print("callback_update_frequency_plot error")

    def figure_frequency_plot(self, start_index=0, end_index=-1, x=np.array([]), y=np.array([])):
        # Clear the previous plot
        self.axes_fre.clear()

        # Update frequency figure

        self.axes_fre.plot(x[start_index:end_index], y[start_index:end_index], linewidth=0.6, color="r")
        self.axes_fre.set_xlabel('Frequency (Hz)')

        # Adjust the size of the FigureCanvas
        self.fig_fre.tight_layout()

        # Redraw the canvas
        self.canvas_fre.draw()

    def update_plot_frequency(self, y=np.array([])):
        fs = self.ui.doubleSpinBox_fs.value()
        if self.var.step_number == 6:
            frequencies, abs_spectrum = pu_f.computer_spectrum_sim(y)
        else:
            frequencies, abs_spectrum = pu_f.computer_spectrum(y, fs)

        self.var.frequencies_plot = frequencies
        self.var.abs_spectrum_plot = abs_spectrum

        length_data = len(abs_spectrum)
        self.maxspinBox_f = length_data

        if length_data < 1000:
            self.ui.spinBox_f.setValue(length_data)
            self.ui.horizontalScrollBar_f.setEnabled(False)
        else:
            index_fre_show = pu_f.index_nearest(self.var.frequencies_plot, 10 * self.var.f0)  # show 10f harmonic
            self.ui.spinBox_f.setValue(index_fre_show)
            self.ui.horizontalScrollBar_x.setEnabled(True)

        self.ui.horizontalScrollBar_f.setMaximum(length_data - self.ui.spinBox_f.value())

        scroll_value = self.ui.horizontalScrollBar_f.value()
        spinbox_value = self.ui.spinBox_f.value()

        start_window = scroll_value
        end_window = start_window + spinbox_value

        if end_window <= len(abs_spectrum):
            self.figure_frequency_plot(start_window, end_window, frequencies, abs_spectrum)
        else:
            print("update_plot error")
