import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtWidgets
from ui_to_py.main_window import Ui_MainWindow
from widgets.camera_stream import CameraStream
from widgets.control_panel import ControlPanel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        desktop = QtWidgets.QDesktopWidget().availableGeometry()
        desktop.setX(0)
        desktop.setY(0)
        self.setGeometry(desktop)
        self.showMaximized()

        self.camera_stream = CameraStream()
        self.ui.horizontalLayout.replaceWidget(self.ui.horizontalLayout.itemAt(0).widget(), self.camera_stream)

        self.control_panel = ControlPanel()
        self.ui.horizontalLayout.replaceWidget(self.ui.horizontalLayout.itemAt(1).widget(), self.control_panel)




def main():
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())

