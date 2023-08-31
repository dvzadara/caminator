from PyQt5.QtWidgets import QWidget
from ui_to_py.object_list import Ui_Form


class ObjectList(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
