from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox, QFileDialog
from PyQt5.QtCore import QDir, QModelIndex, Qt, QStandardPaths
import pandas as pd
import os
import sys
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
import time
import librosa
from PyQt5.QtGui import QColor

from New_window import Ui_MainWindow
from figure_plot import FigurePlot
from file_var import *

import funtion_pulse as pu_f


class MenuStep(QMainWindow):
    def __init__(self, self_instance):
        super(MenuStep, self).__init__()
        self.ui: Ui_MainWindow = self_instance.ui
        self.var: VarData = self_instance.var
        self.figure_plot_instance: FigurePlot = self_instance.figure_plot_instance

        self.ui.pushButton_next.clicked.connect(self.next)
        self.ui.pushButton_back.clicked.connect(self.back)

        self.ui.pushButton_next.setEnabled(False)

        self.step_function = [self.step_0, self.step_1, self.step_2, self.step_3, self.step_4, self.step_5,
                              self.step_6]

        self.next_flag = 0  # flag next button
        self.width_frame_information = 260

    def next(self):
        if self.var.step_number is not None:  # have data in y
            self.next_flag = 1  # flag next
            # update step
            self.var.step_number += 1
            # print(self.var.step_number)

            # apply function
            self.step_function[self.var.step_number]()

    def back(self):
        # update step
        self.var.step_number -= 1
        # print(self.var.step_number)

        # apply function
        self.step_function[self.var.step_number]()

    def step_led(self, step):
        if step == 0:
            self.ui.pushButton_led_1.setStyleSheet('background-color: gray; border: none')
            self.ui.pushButton_led_2.setStyleSheet('background-color: gray; border: none')
            self.ui.pushButton_led_3.setStyleSheet('background-color: gray; border: none')
            self.ui.pushButton_led_4.setStyleSheet('background-color: gray; border: none')
            self.ui.pushButton_led_5.setStyleSheet('background-color: gray; border: none')
            self.ui.pushButton_led_6.setStyleSheet('background-color: gray; border: none')
        elif step == 1:
            self.ui.pushButton_led_1.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_2.setStyleSheet('background-color: gray; border: none')
            self.ui.pushButton_led_3.setStyleSheet('background-color: gray; border: none')
            self.ui.pushButton_led_4.setStyleSheet('background-color: gray; border: none')
            self.ui.pushButton_led_5.setStyleSheet('background-color: gray; border: none')
            self.ui.pushButton_led_6.setStyleSheet('background-color: gray; border: none')
        elif step == 2:
            self.ui.pushButton_led_1.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_2.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_3.setStyleSheet('background-color: gray; border: none')
            self.ui.pushButton_led_4.setStyleSheet('background-color: gray; border: none')
            self.ui.pushButton_led_5.setStyleSheet('background-color: gray; border: none')
            self.ui.pushButton_led_6.setStyleSheet('background-color: gray; border: none')
        elif step == 3:
            self.ui.pushButton_led_1.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_2.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_3.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_4.setStyleSheet('background-color: gray; border: none')
            self.ui.pushButton_led_5.setStyleSheet('background-color: gray; border: none')
            self.ui.pushButton_led_6.setStyleSheet('background-color: gray; border: none')
        elif step == 4:
            self.ui.pushButton_led_1.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_2.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_3.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_4.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_5.setStyleSheet('background-color: gray; border: none')
            self.ui.pushButton_led_6.setStyleSheet('background-color: gray; border: none')
        elif step == 5:
            self.ui.pushButton_led_1.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_2.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_3.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_4.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_5.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_6.setStyleSheet('background-color: gray; border: none')
        elif step == 6:
            self.ui.pushButton_led_1.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_2.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_3.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_4.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_5.setStyleSheet('background-color: green; border: none')
            self.ui.pushButton_led_6.setStyleSheet('background-color: green; border: none')

    def step_0(self):
        if self.var.step_number == 0:
            # update data
            self.var.y_plot = self.var.y_raw
            self.figure_plot_instance.update_plot(self.var.x, self.var.y_plot)

            # update window
            self.step_led(self.var.step_number)
            self.ui.pushButton_next.setText("Classify >>")
            self.ui.stackedWidget_setup.setCurrentIndex(0)
            self.ui.stackedWidget_back.setCurrentIndex(0)
            self.ui.pushButton_next.setEnabled(True)
            self.ui.label_time.setText("Raw signal")
            self.ui.label_frequency.setText("Amplitude spectrum")

    def step_1(self):  # Classify
        if self.var.step_number == 1:

            analyzed_window = int((self.var.fs // self.var.f0) * self.ui.spinBox_window_length.value())
            self.var.y_analyzed = self.var.y_raw[:analyzed_window]
            # test
            print(f"step_number={self.var.step_number}, analyzed_window = {analyzed_window}")
            # Classify
            if self.next_flag == 1:  # for Classify button
                self.next_flag = 0  # clear flag
                # update var
                self.var.f0 = self.ui.doubleSpinBox_f0.value()
                self.var.fs = self.ui.doubleSpinBox_fs.value()
                # Normalization
                type_nor = self.ui.comboBox_nor.currentText()
                start_time = time.time()
                if type_nor == "Z-score":
                    nor_data = pu_f.nor_zscore(self.var.y_analyzed)
                elif type_nor == "Min-max":
                    nor_data = pu_f.nor_minmax(self.var.y_analyzed)

                # classify signal
                th1 = pu_f.computer_THD(nor_data, up_limit=int(self.var.fs // (2 * self.var.f0)-1), f0=self.var.f0, fs=self.var.fs)
                th2 = pu_f.get_HNR(nor_data, rate=self.var.fs)

                if th1 > 2:
                    if th2 > 2:
                        type_signal = "Methane signal with low noise level"  # methane_low_noise
                        # update M
                        self.ui.spinBox_M.setValue(50)
                    else:
                        type_signal = "Methane signal with high noise level"  # methane_high_noise
                        # update M
                        self.ui.spinBox_M.setValue(200)
                else:
                    type_signal = "Methane leak absence "  # without_methane
                    # update M
                    self.ui.spinBox_M.setValue(200)
                end_time = time.time()
                elapsed_time = end_time - start_time

                cent = np.mean(librosa.feature.spectral_centroid(y=nor_data, sr=self.var.fs))
                SC = np.round(cent / 1000, 2)

                # update window and Information of signal
                self.ui.label_type_signal.setText(f"{type_signal}")
                self.ui.label_time_classify.setText(f"{np.round(elapsed_time, 2)}")
                self.ui.label_th1.setText("2")
                self.ui.label_th2.setText("2")
                self.ui.label_HNR.setText(f"{np.round(th2, 2)}")
                ROE = pu_f.computer_ROE(nor_data, up_limit=int(self.var.fs // (2 * self.var.f0)-1), f0=self.var.f0, fs=self.var.fs)
                self.ui.label_ROE.setText(f"{np.round(ROE, 2)}")
                self.ui.label_THD.setText(f"{np.round(th1, 2)}")
                self.ui.label_SC.setText(f"{SC}")

                QMessageBox.information(None, 'Type of signal', f'{type_signal}')

            # update data
            self.var.y_plot = self.var.y_analyzed
            self.figure_plot_instance.update_plot(self.var.x, self.var.y_plot)

            # update window
            self.step_led(self.var.step_number)
            self.ui.pushButton_next.setText("Next >>")
            self.ui.stackedWidget_setup.setCurrentIndex(self.var.step_number)
            self.ui.stackedWidget_back.setCurrentIndex(1)
            self.ui.pushButton_next.setEnabled(True)
            self.ui.label_time.setText("Raw signal")
            self.ui.label_frequency.setText("Amplitude spectrum")

    def step_2(self): # Setup for TSMA
        if self.var.step_number == 2:
            # update data
            if self.next_flag == 1:  # for next button
                self.next_flag = 0  # clear flag
                # TSMA filter signal
                m = self.ui.spinBox_M.value()  # M neighboring cycles
                start_time = time.time()

                self.var.nor_TSMA = pu_f.TSMA(self.var.y_analyzed, window_size=m, f=self.var.f0, fs=self.var.fs)

                # Normalization
                if self.ui.comboBox_nor.currentText() == "Z-score":
                    self.var.nor_TSMA = pu_f.nor_zscore(self.var.nor_TSMA)
                elif self.ui.comboBox_nor.currentText() == "Min-max":
                    self.var.nor_TSMA = pu_f.nor_minmax(self.var.nor_TSMA)
                else:
                    self.var.nor_TSMA = self.var.nor_TSMA

                start_stop = time.time()
                self.var.time[self.var.step_number] = start_stop - start_time

                # test
                print(f"step_number={self.var.step_number}, M = {self.ui.spinBox_M.value()}, Normalization = {self.ui.comboBox_nor.currentText()}")

            # update window and Information of signal
            self.ui.label_HNR_b.setText(f"{pu_f.get_HNR(self.var.y_analyzed, rate=self.var.fs)}")
            self.ui.label_HNR_a.setText(f"{pu_f.get_HNR(self.var.nor_TSMA, rate=self.var.fs)}")
            # update window
            self.step_led(self.var.step_number)
            self.ui.stackedWidget_setup.setCurrentIndex(self.var.step_number)
            self.ui.stackedWidget_back.setCurrentIndex(1)
            self.ui.label_time.setText("TSMA signal")
            self.ui.label_frequency.setText("Amplitude spectrum")

            # update graph
            self.var.y_plot = self.var.nor_TSMA  # để khi thay đổi horizontalScrollBar_x.valueChanged
            self.figure_plot_instance.update_plot(self.var.x, self.var.y_plot)

    def step_3(self):
        if self.var.step_number == 3:
            if self.next_flag == 1:  # for next button
                self.next_flag = 0  # clear flag
                # init function
                k_subband = self.ui.spinBox_K_subband.value()
                oder = self.ui.spinBox_order.value()
                band_width = self.ui.doubleSpinBox_bwidth.value()
                start_time = time.time()
                bp_sos, self.var.mb_sos = pu_f.init_funtion(k_subband, oder, band_width, self.var.fs, self.var.f0)
                # Laser output signal extraction
                self.var.baseline = pu_f.bandpass_sos(self.var.nor_TSMA, bp_sos)  # bandpass_butter    bandpass_sos
                start_stop = time.time()
                self.var.time[self.var.step_number] = start_stop - start_time

                # test
                # plt.figure(figsize=(16, 4))
                # plt.title("baseline")
                # plt.plot(self.var.baseline[:320])
                # plt.show()
                print(f"step_number={self.var.step_number}, k_subband = {k_subband}, oder = {oder}, band_width={band_width}")

            # update graph
            self.var.y_plot = self.var.baseline  # để khi thay đổi horizontalScrollBar_x.valueChanged
            self.figure_plot_instance.update_plot(self.var.x, self.var.y_plot)
            # update window
            self.step_led(self.var.step_number)
            self.ui.stackedWidget_setup.setCurrentIndex(self.var.step_number)
            self.ui.label_time.setText("Laser output signal")
            self.ui.label_frequency.setText("Amplitude spectrum")

    def step_4(self):
        if self.var.step_number == 4:
            if self.next_flag == 1:  # for next button
                self.next_flag = 0  # clear flag
                # Absorption profile extraction
                start_time = time.time()
                self.var.absorption_profile = pu_f.mul_bandpass_sos(self.var.nor_TSMA, self.var.mb_sos)
                self.var.absorption_profile = np.abs(self.var.absorption_profile)
                start_stop = time.time()
                self.var.time[self.var.step_number] = start_stop - start_time

                # test
                # plt.figure(figsize=(16, 4))
                # plt.title("absorption_profile")
                # plt.plot(self.var.absorption_profile[:320])
                # plt.show()

            # update graph
            self.var.y_plot = self.var.absorption_profile  # để khi thay đổi horizontalScrollBar_x.valueChanged
            self.figure_plot_instance.update_plot(self.var.x, self.var.y_plot)
            # update window
            self.step_led(self.var.step_number)
            self.ui.stackedWidget_setup.setCurrentIndex(self.var.step_number)
            self.ui.label_time.setText("Absorption pulse signal")
            self.ui.label_frequency.setText("Amplitude spectrum")

    def step_5(self):
        if self.var.step_number == 5:
            self.ui.pushButton_next.setEnabled(True)

            if self.next_flag == 1:  # for next button
                self.next_flag = 0  # clear flag
                start_time = time.time()
                # peak detection of baseline, detection p_p_amplitude_baseline
                peak_idx_baseline, p_p_amplitude_baseline = pu_f.peaks_detection(self.var.baseline, type_peak="both")
                # Calibration of absorption profile signal
                if self.ui.radioButton.isChecked():
                    absorption_profile_cali = (
                        np.divide(self.var.absorption_profile, p_p_amplitude_baseline))  # np.mean(p_p_amplitude))
                else:
                    absorption_profile_cali = self.var.absorption_profile

                # Peak detection of absorption profile signal
                peak_idx_ab = pu_f.peaks_profile(absorption_profile_cali, self.var.baseline, peak_idx_baseline)
                # Pulse absorbance forming
                band_pulse = self.ui.spinBox_bwidth_pulse.value()
                self.var.average_peak_data = pu_f.peakfit(absorption_profile_cali, peak_idx_ab, band_pulse)
                start_stop = time.time()
                self.var.time[self.var.step_number] = start_stop - start_time

            # update graph
            self.var.y_plot = self.var.average_peak_data  # để khi thay đổi horizontalScrollBar_x.valueChanged
            x = np.arange(0, len(self.var.y_plot))
            self.figure_plot_instance.update_plot(x, self.var.y_plot)
            # update window
            self.step_led(self.var.step_number)
            self.ui.stackedWidget_setup.setCurrentIndex(self.var.step_number)
            self.ui.label_time.setText("Absorption pulse signal")
            self.ui.label_frequency.setText("Amplitude spectrum")

    # Estimation of absorption pulse parameter and concentration
    def step_6(self):
        if self.var.step_number == 6:

            self.ui.pushButton_next.setEnabled(False)

            if self.next_flag == 1:  # for next button
                self.next_flag = 0  # clear flag
                start_time = time.time()
                # Estimation of absorption pulse parameter
                self.var.amplitude_ab, self.var.area_under_curve, self.var.time_centroi, self.var.fre_centroi= (
                    pu_f.estimation_parameter(self.var.average_peak_data))
                start_stop = time.time()
                self.var.time[self.var.step_number] = start_stop - start_time
            if self.ui.comboBox_power.currentText() == "166":
                y_amp = (self.var.amplitude_ab - 0.02067438707505119) / 7.436019090610386e-05
                y_area = (self.var.area_under_curve - 0.016010362132876592) / 3.7233932958028236e-05
                y_time = (self.var.time_centroi - 0.03312336255642014) / 0.00010156285006864742
                y_fre = (self.var.fre_centroi - 0.0018718228617012219) / 3.693599419098027e-06
            elif self.ui.comboBox_power.currentText() == "9":
                y_amp = (self.var.amplitude_ab - 0.01275115453647872) / 7.041778542869059e-05
                y_area = (self.var.area_under_curve - 0.02405918462566) / 0.00011659086697488249
                y_time = (self.var.time_centroi - 0.08249209336964047) / 0.0005595161705991008
                y_fre = (self.var.fre_centroi - 0.002393861327036351) / 1.660490784146049e-05
            elif self.ui.comboBox_power.currentText() == "2":
                y_amp = (self.var.amplitude_ab - 0.020000270944810168) / 5.7941938645759846e-05
                y_area = (self.var.area_under_curve - 0.023841199147455087) / 9.10407882657668e-05
                y_time = (self.var.time_centroi - 0.05243584584422825) / 0.0004766187157616048
                y_fre = (self.var.fre_centroi - 0.0011596439904267087) / 1.4852897787199998e-05
            else:
                y_amp = 0
                y_area = 0
                y_time = 0
                y_fre = 0

            # update graph
            self.var.y_plot = self.var.average_peak_data  # để khi thay đổi horizontalScrollBar_x.valueChanged
            x = np.arange(0, len(self.var.y_plot))
            self.figure_plot_instance.update_plot(x, self.var.y_plot)
            # update window
            self.step_led(self.var.step_number)
            self.ui.stackedWidget_setup.setCurrentIndex(self.var.step_number)
            self.ui.label_time.setText("Absorption pulse signal")
            self.ui.label_frequency.setText("Amplitude spectrum")

            # update window and Information of signal
            self.ui.label_amp.setText(f"{np.round(self.var.amplitude_ab, 4)}")
            self.ui.label_area.setText(f"{self.var.area_under_curve}")
            self.ui.label_time_centr.setText(f"{np.round(self.var.time_centroi, 4)}")
            self.ui.label_fre_centr.setText(f"{np.round(self.var.fre_centroi, 4)}")
            self.ui.label_time_comp.setText(f"{np.round(np.sum(self.var.time), 4)}")

            self.ui.label_amp_c.setText(f"{np.round(y_amp, 4)}")
            self.ui.label_area_c.setText(f"{np.round(y_area, 4)}")
            self.ui.label_time_centr_c.setText(f"{np.round(y_time, 4)}")
            self.ui.label_fre_centr_c.setText(f"{np.round(y_fre, 4)}")


