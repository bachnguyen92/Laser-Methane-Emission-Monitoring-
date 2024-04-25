from PyQt5.QtWidgets import QApplication, QMainWindow, QFileSystemModel, QMessageBox, QPushButton
from PyQt5.QtCore import QDir, QModelIndex, Qt, QStandardPaths
import pandas as pd
import os
import sys
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtGui import QColor

from New_window import Ui_MainWindow
from figure_plot import FigurePlot
from file_var import *
from menu_step import MenuStep

class CustomFileSystemModel(QFileSystemModel):
    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            if section == 0:
                return "Load file"
        return super(CustomFileSystemModel, self).headerData(section, orientation, role)


class OpenFile(QMainWindow):
    def __init__(self, self_instance):
        super(OpenFile, self).__init__()
        self.instance = self_instance
        self.ui: Ui_MainWindow = self_instance.ui
        self.var: VarData = self_instance.var
        self.figure_plot_instance: FigurePlot = self_instance.figure_plot_instance

        self.ui.pushButton_openfile.clicked.connect(self.open_file)
        self.ui.pushButton_add_data.clicked.connect(self.add_data)
        self.ui.pushButton_parent_fd.clicked.connect(self.parent_directory)
        self.ui.pushButton_load_fd.clicked.connect(self.open_current_directory)
        self.ui.pushButton_toggleWidth.clicked.connect(self.toggleWidth)
        self.ui.pushButton_apply.clicked.connect(self.padding_data)

        self.ui.radioButton_padding.toggled.connect(self.on_padding_toggled)

        self.ui.treeView.doubleClicked.connect(self.tree_item_double_clicked)

        # ================================================================ local var
        self.file_path = None
        self.max_width = self.ui.frame_signal_file.maximumWidth()
        self.add_data_flag = None
        # ================================================================

        # Load file rootPath
        self.dirModel = CustomFileSystemModel()  # đổi Name -> Load file
        self.load_file(QDir.rootPath())

    def toggleWidth(self):
        # toggle Width
        current_width = self.ui.frame_signal_file.maximumWidth()
        if current_width != 0:
            new_width = 0
            self.ui.pushButton_toggleWidth.setText(">>")
            if self.var.step_number is not None:  # phải có dữ liệu plot kng bị lõi
                self.figure_plot_instance.callback_update_frequency_plot(self.var.frequencies_plot,
                                                                         self.var.abs_spectrum_plot)
                self.figure_plot_instance.callback_update_time_plot(self.var.x, self.var.y_plot)
        else:
            new_width = self.max_width
            self.ui.pushButton_toggleWidth.setText("<<")
            if self.var.step_number is not None:
                self.figure_plot_instance.callback_update_frequency_plot(self.var.frequencies_plot,
                                                                         self.var.abs_spectrum_plot)
                self.figure_plot_instance.callback_update_time_plot(self.var.x, self.var.y_plot)
        # set Width of QFrame
        self.ui.frame_signal_file.setFixedWidth(new_width)

    def on_padding_toggled(self):
        if self.ui.radioButton_padding.isChecked():
            self.ui.spinBox_padding.setEnabled(True)
            self.ui.pushButton_apply.setEnabled(True)
        else:
            self.ui.spinBox_padding.setEnabled(False)
            self.ui.pushButton_apply.setEnabled(False)

    def padding_data(self):
        num_samp_period = int(self.var.fs // self.var.f0)
        all_num_period = int(len(self.var.y_raw) // num_samp_period)
        desired_length = num_samp_period * self.ui.spinBox_padding.value()

        if desired_length > len(self.var.y_raw):
            # Tính số lần cần lặp
            num_repeats = desired_length // len(self.var.y_raw)
            remaining_samples = desired_length % len(self.var.y_raw)

            padded_signal = np.array([])
            # Lặp lại tín hiệu y_raw
            for _ in range(num_repeats):
                padded_signal = np.concatenate((padded_signal, self.var.y_raw))
            padded_signal = np.concatenate((padded_signal, self.var.y_raw[:remaining_samples]))

            # Cập nhật tín hiệu y_raw với padding
            self.var.y_raw = padded_signal

            # update Information of signal
            num_samp_period = int(self.var.fs // self.var.f0)
            all_num_period = int(len(self.var.y_raw) // num_samp_period)
            self.ui.label_length_signal.setText(f"{all_num_period}")
            self.ui.label_length_sample.setText(f"{len(self.var.y_raw)}")

    def add_data(self):
        self.add_data_flag = 1
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        self.file_path, _ = QFileDialog.getOpenFileName(None, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)",
                                                        options=options)
        file_name = os.path.basename(self.file_path)

        if file_name:
            file_name, _ = os.path.splitext(file_name)  # loại bỏ đuôi csv

            self.ui.label_name_signal.setText(f"{self.ui.label_name_signal.text()}+{file_name}")
            file_directory = os.path.dirname(self.file_path)
            self.load_file(file_directory)

            self.read_plot_data(file_name)

    def open_file(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog

        self.file_path, _ = QFileDialog.getOpenFileName(None, "Open CSV File", "", "CSV Files (*.csv);;All Files (*)",
                                                        options=options)
        file_name = os.path.basename(self.file_path)

        if file_name:
            file_name, _ = os.path.splitext(file_name)  # loại bỏ đuôi csv
            self.ui.label_name_signal.setText(f"{file_name}")
            file_directory = os.path.dirname(self.file_path)
            self.load_file(file_directory)

            self.read_plot_data(file_name)

    def read_plot_data(self, file_name):
        # update new data
        buffer = np.array([])
        try:
            buffer = pd.read_csv(self.file_path, header=None, low_memory=False)[0].values
        except Exception as e:
            print(f"Error reading CSV file with header=None 1: {e}")

        try:
            buffer = pd.read_csv(self.file_path, low_memory=False)["Signal"].values
        except Exception as e:
            print(f"Error reading CSV file with header=None 2: {e}")

        try:
            laser = np.array([])
            laser = (pd.read_csv(self.file_path, sep=';', encoding='latin1', low_memory=False))['Ëàçåð'].values
            reference = (pd.read_csv(self.file_path, sep=';', encoding='latin1', low_memory=False))['Ýòàëîí'].values
            signal = (pd.read_csv(self.file_path, sep=';', encoding='latin1', low_memory=False))['Ñèãíàë'].values
            if len(laser) != 0:
                msg_box = QMessageBox()
                msg_box.setWindowTitle("Select Signal")
                msg_box.setText("Select the signal to process:")

                # Tạo các nút button với tên tùy chỉnh
                button_signal1 = QPushButton("Laser")
                button_signal2 = QPushButton("Reference")
                button_signal3 = QPushButton("Signal")

                # Thêm các nút button vào hộp thoại
                msg_box.addButton(button_signal1, QMessageBox.YesRole)
                msg_box.addButton(button_signal2, QMessageBox.NoRole)
                msg_box.addButton(button_signal3, QMessageBox.RejectRole)

                # Xử lý sự kiện khi người dùng chọn nút button
                def button_clicked(button):
                    nonlocal buffer
                    nonlocal laser
                    nonlocal reference
                    nonlocal signal
                    if button == button_signal1:
                        buffer = laser
                    elif button == button_signal2:
                        buffer = reference
                    elif button == button_signal3:
                        buffer = signal
                    msg_box.close()

                # Kết nối sự kiện click cho mỗi nút button với hàm xử lý tương ứng
                button_signal1.clicked.connect(lambda: button_clicked(button_signal1))
                button_signal2.clicked.connect(lambda: button_clicked(button_signal2))
                button_signal3.clicked.connect(lambda: button_clicked(button_signal3))

                # Hiển thị hộp thoại
                msg_box.exec_()

        except Exception as e:
            print(f"Error reading CSV file with header=None 3: {e}")

        if len(buffer) == 0:
            QMessageBox.warning(self, 'Warning', 'Error reading CSV file!')
            # update window
            self.figure_plot_instance.init_window(self.instance)
            self.var.y_raw = np.array([])
            return
        else:
            if self.add_data_flag == 1:
                self.add_data_flag = None   # clear flag
                self.var.y_raw = np.concatenate((self.var.y_raw, buffer))
            else:
                self.var.y_raw = buffer

        self.var.step_number = 0
        self.var.y_plot = self.var.y_raw

        print(self.var.step_number)

        self.var.x = np.arange(0, len(self.var.y_plot))
        # update Information of signal
        num_samp_period = int(self.var.fs // self.var.f0)
        all_num_period = int(len(self.var.y_raw) // num_samp_period)
        self.ui.label_length_signal.setText(f"{all_num_period}")
        self.ui.label_length_sample.setText(f"{len(self.var.y_raw)}")
        self.ui.label_name_signal_2.setText(f"Name of signal: {file_name}")
        # update plot
        self.figure_plot_instance.update_plot(self.var.x, self.var.y_plot)
        # update window
        self.figure_plot_instance.openfile_window()

    def load_file(self, directory):
        self.dirModel.setRootPath(directory)
        self.dirModel.setFilter(QDir.NoDotAndDotDot | QDir.AllEntries)

        self.ui.treeView.setModel(self.dirModel)
        self.ui.treeView.setRootIndex(self.dirModel.index(directory))

        for column in range(1, self.dirModel.columnCount()):
            self.ui.treeView.setColumnHidden(column, True)

    def tree_item_double_clicked(self, index: QModelIndex):
        file_info = self.dirModel.fileInfo(index)
        if file_info.isFile() and file_info.suffix().lower() == "csv":
            file_name, _ = os.path.splitext(file_info.fileName())  # loại bỏ đuôi csv
            self.ui.label_name_signal.setText(file_name)

            self.file_path = file_info.filePath()
            self.read_plot_data(file_name)

    def parent_directory(self):
        current_index = self.ui.treeView.rootIndex()
        parent_index = self.dirModel.parent(current_index)
        if not parent_index.isValid():
            return
        parent_path = self.dirModel.filePath(parent_index)

        self.load_file(parent_path)

    def open_current_directory(self):
        current_index = self.ui.treeView.currentIndex()

        if not current_index.isValid():
            return

        current_directory = self.dirModel.filePath(current_index)

        self.load_file(current_directory)
