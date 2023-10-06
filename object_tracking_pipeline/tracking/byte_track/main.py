import cv2

from byte_tracker import BYTETracker
from object_tracking_pipeline.drawing_results import draw_box
from ultralytics import YOLO

class ByteTrackArgument:
    track_thresh = 0.5 # High_threshold
    track_buffer = 50 # Number of frame lost tracklets are kept
    match_thresh = 0.8 # Matching threshold for first stage linear assignment
    aspect_ratio_thresh = 10.0 # Minimum bounding box aspect ratio
    min_box_area = 1.0 # Minimum bounding box area
    mot20 = False # If used, bounding boxes are not clipped.

def tlwh_to_xyxy(tlwh):
    x1 = tlwh[0]
    y1 = tlwh[1]
    x2 = tlwh[0] + tlwh[2]
    y2 = tlwh[1] + tlwh[3]
    return (x1, y1, x2, y2)

MIN_THRESHOLD = 0.001


tracker = BYTETracker(ByteTrackArgument)
# model = YOLO('weights/best.pt')
model = YOLO('../../../weights/best.pt')

video_capture = cv2.VideoCapture(0)  # 0 for default camera
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Устанавливаем ширину кадра
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Устанавливаем высоту кадра

img_size = (1280, 720)

while True:
    ret, frame = video_capture.read()
    if ret:
        outputs = model.predict(source=frame, conf=MIN_THRESHOLD)
        img_height, img_width = outputs[0].boxes.orig_shape
        outputs = outputs[0].boxes.data
        class_outputs = outputs[outputs[:, 5] == 0][:,:5]
        if class_outputs is not None:
            online_targets = tracker.update(class_outputs.cpu(), img_size, img_size)
            online_tlwhs = []
            online_ids = []
            online_scores = []
            online_classes = [0] * len(online_targets)
            for t in online_targets:
                tlwh = t.tlwh
                tid = t.track_id
                vertical = tlwh[2] / tlwh[3] > ByteTrackArgument.aspect_ratio_thresh
                if tlwh[2] * tlwh[3] > ByteTrackArgument.min_box_area and not vertical:
                    online_tlwhs.append(tlwh)
                    online_ids.append(tid)
                    online_scores.append(t.score)
                    box = tlwh_to_xyxy(tlwh)
                    frame = draw_box(frame, box, tid)

        cv2.imshow("...", frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # frame = draw_tracks(frame, tracker)
        # rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # cv2.imshow("...", frame)