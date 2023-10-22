import cv2
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget
from usefull_classes.observer import Observable
from object_tracking_pipeline.detection_models.onnx_model.onnx_prediction import *
# from object_tracking_pipeline.detection_models.yolo_model.yolo_prediction import *
from object_tracking_pipeline.tracking.my_tracker import *
from object_tracking_pipeline.drawing_results import *
import datetime


class CameraStream(QWidget, Observable):
    """
    Camera stream widget must process the camera stream and display the results in the main window.
    Also notifies the object list widget`s observer every frame.
    """
    def __init__(self):
        super().__init__()
        self.video_label = QLabel(self)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.video_label)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)

        self.video_capture = cv2.VideoCapture(0)  # 0 for default camera
        self.video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Устанавливаем ширину кадра
        self.video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Устанавливаем высоту кадра

        isError, frame = self.video_capture.read()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 milliseconds

        self.setStyleSheet("background-color: white;")  # Устанавливаем белый фон
        self.tracker = MyTracker(byte_tracker_type)

    def update_frame(self):
        """
        Function read frame from the camera, use model(object_tracking_pipeline package) for prediction,
        notify observers and displaying the image.
        """
        ret, frame = self.video_capture.read()
        if ret:
            now_time = datetime.datetime.now()
            results = run_model(frame)
            detections = results2boxes_and_probs(results)
            self.tracker.track_objects(detections)
            time_delta = datetime.datetime.now() - now_time
            print(f"Frame process time: {time_delta.total_seconds()}")
            frame = draw_boxes(frame, self.tracker)
            frame = draw_tracks(frame, self.tracker)
            object_list = list(map(lambda x: "human " + str(x), self.tracker.current_ids))
            self.notify_observers(object_list=object_list)
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            q_image = QImage(rgb_image.data, w, h, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)

            # Изменение способа масштабирования изображения
            scaled_pixmap = pixmap.scaledToWidth(self.video_label.width(), 0)

            self.video_label.setPixmap(scaled_pixmap)
            self.video_label.setAlignment(Qt.AlignCenter)

    def closeEvent(self, event):
        self.video_capture.release()
        super().closeEvent(event)
