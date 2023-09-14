from PyQt5.QtWidgets import QWidget
from ui_to_py.object_list import Ui_Form
from PyQt5.QtGui import QStandardItem, QStandardItemModel
from usefull_classes.observer import Observer


class ObjectList(QWidget, Observer):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Form()
        self.ui.setupUi(self)
        self.model = QStandardItemModel()
        self.ui.listView.setModel(self.model)

    def update_object_list(self, object_list):
        self.model.clear()
        for i in range(len(object_list)):
            item = QStandardItem(object_list[i])
            self.model.appendRow(QStandardItem(item))

    def notify(self, observable, *args, **kwargs):
        self.update_object_list(kwargs["object_list"])