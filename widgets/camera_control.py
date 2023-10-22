from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QWidget
from ui_to_py.camera_control import Ui_Form


class CameraControl(QWidget):
    """
    Camera control widget contains buttons for camera control.
    """

    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)

        self.camera_coordinates = {"azimuth": 0, "inclination": 0}

        # Movement flags contains flags with current directions for camera moving,
        # if direction flag is true camera moves in this direction
        self.movement_flags = {"top": False, "bottom": False, "left": False, "right": False}

        self.ui.pushButton_3.pressed.connect(lambda: self.change_movement_vector("top", True))
        self.ui.pushButton_4.pressed.connect(lambda: self.change_movement_vector("left", True))
        self.ui.pushButton_5.pressed.connect(lambda: self.change_movement_vector("right", True))
        self.ui.pushButton_6.pressed.connect(lambda: self.change_movement_vector("bottom", True))

        self.ui.pushButton_3.released.connect(lambda: self.change_movement_vector("top", False))
        self.ui.pushButton_4.released.connect(lambda: self.change_movement_vector("left", False))
        self.ui.pushButton_5.released.connect(lambda: self.change_movement_vector("right", False))
        self.ui.pushButton_6.released.connect(lambda: self.change_movement_vector("bottom", False))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.change_camera_position)
        self.timer.start(30)

    def change_movement_vector(self, direction, value):
        self.movement_flags[direction] = value
        print("button is clicked", direction, value)

    def change_camera_position(self):
        """
        Function moves camera using movements flags.
        """
        self.camera_coordinates["inclination"] += 10 * (int(self.movement_flags["top"]) -
                                                        int(self.movement_flags["bottom"]))
        self.camera_coordinates["azimuth"] += 10 * (int(self.movement_flags["right"]) -
                                                    int(self.movement_flags["left"]))
        self.camera_coordinates["azimuth"] %= 360
        if -90 > self.camera_coordinates["inclination"]:
            self.camera_coordinates["inclination"] = -90
        if 90 < self.camera_coordinates["inclination"]:
            self.camera_coordinates["inclination"] = 90
        print(self.camera_coordinates["azimuth"], self.camera_coordinates["inclination"])
