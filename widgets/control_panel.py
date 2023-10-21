from PyQt5.QtWidgets import QWidget
from ui_to_py.control_panel import Ui_Form
from widgets.object_list import ObjectList
from widgets.camera_control import CameraControl


class ControlPanel(QWidget):
    """
    The control panel widget is responsible for controlling the camera, contains a camera control widget for manual
    control and object list widget for automatically tracking an object from the object list.
    """
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.object_list = ObjectList()
        self.ui.verticalLayout.replaceWidget(self.ui.verticalLayout.itemAt(0).widget(), self.object_list)

        self.camera_control = CameraControl()
        self.ui.verticalLayout.replaceWidget(self.ui.verticalLayout.itemAt(1).widget(), self.camera_control)
