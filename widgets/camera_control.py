from PyQt5.QtWidgets import QWidget
from ui_to_py.camera_control import Ui_Form


class CameraControl(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
