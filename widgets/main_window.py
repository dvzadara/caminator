import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
from ui_to_py.main_window import Ui_MainWindow
from widgets.camera_stream import CameraStream
from widgets.control_panel import ControlPanel
import traceback


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # icon = QIcon('images/caminator_logo_without_text.png')
        # self.setWindowIcon(icon)

        desktop = QtWidgets.QDesktopWidget().availableGeometry()
        desktop.setX(0)
        desktop.setY(0)
        self.setGeometry(desktop)
        self.showMaximized()

        self.camera_stream = CameraStream()
        self.ui.horizontalLayout.replaceWidget(self.ui.horizontalLayout.itemAt(0).widget(), self.camera_stream)

        self.control_panel = ControlPanel()
        self.ui.horizontalLayout.replaceWidget(self.ui.horizontalLayout.itemAt(1).widget(), self.control_panel)

        self.camera_stream.register_observer(self.control_panel.object_list)


def excepthook(exc_type, exc_value, exc_tb):
    tb = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    print("error message:\n", tb)
    QtWidgets.QApplication.quit()


def main():
    sys.excepthook = excepthook
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
