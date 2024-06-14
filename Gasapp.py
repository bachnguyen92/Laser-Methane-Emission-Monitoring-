from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import Qt
import pyautogui
import time

from New_window import Ui_MainWindow
from figure_plot import FigurePlot
from file_var import VarData
from open_file import OpenFile
from menu_step import MenuStep


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.var = None
        self.open_file_instance = None
        self.figure_plot_instance = None
        self.menu_step_instance = None
        # ========================================================== Init step
        self.var = VarData()

        self.figure_plot_instance = FigurePlot(self)

        self.open_file_instance = OpenFile(self)

        self.menu_step_instance = MenuStep(self)

        # ========================================================== First step

    def keyPressEvent(self, event):
        key = event.key()
        if self.var.step_number is None:
            return
        if 0 <= self.var.step_number < 6:
            if key == Qt.Key_Enter or key == Qt.Key_F6 or key == Qt.Key_Return:   # Key_Return = Qt.Key_Enter
                self.ui.pushButton_next.animateClick()
        if 0 < self.var.step_number <= 6:
            if key == Qt.Key_F5:
                self.ui.pushButton_back.animateClick()


if __name__ == "__main__":
    app = QApplication([])
    main_win = MainWindow()
    main_win.show()
    app.exec_()
