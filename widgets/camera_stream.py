import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class CameraStream(QWidget):
    def __init__(self):
        super().__init__()
        self.video_label = QLabel(self)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.video_label)

        self.video_capture = cv2.VideoCapture(0)  # 0 for default camera

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 milliseconds

    def update_frame(self):
        ret, frame = self.video_capture.read()

        if ret:
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            q_image = QImage(rgb_image.data, w, h, ch * w, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            self.video_label.setPixmap(pixmap)
            self.video_label.setAlignment(Qt.AlignCenter)

    def closeEvent(self, event):
        self.video_capture.release()
        super().closeEvent(event)
