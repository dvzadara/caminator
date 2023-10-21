import cv2

from PIL import Image
import numpy as np
from torch import tensor
from byte_tracker import BYTETracker
from object_tracking_pipeline.drawing_results import draw_box
from ultralytics import YOLO
import onnxruntime as rt


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


def image_to_input_tensor(image):
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (input_width, input_height))

    # Scale input pixel value to 0 to 1
    input_image = resized / 255.0
    input_image = input_image.transpose(2, 0, 1)
    input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
    return input_tensor


input_width, input_height = (640, 640)
MIN_THRESHOLD = 0.001


tracker = BYTETracker(ByteTrackArgument)
# model = YOLO('weights/best.pt')
model = YOLO('../../../weights/best.pt')
ort_session = rt.InferenceSession(r'C:\Users\mafara\Desktop\moi_shedevri\university\7semestr\caminator\weights\yolov8n.onnx', providers=['CPUExecutionProvider'])

video_capture = cv2.VideoCapture(0)  # 0 for default camera
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Устанавливаем ширину кадра
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Устанавливаем высоту кадра

img_size = (1280, 720)

while True:
    ret, frame = video_capture.read()
    if ret:
        # outputs = model.predict(source=frame, conf=MIN_THRESHOLD)
        model_inputs = ort_session.get_inputs()
        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        input_shape = model_inputs[0].shape
        model_output = ort_session.get_outputs()
        output_names = [model_output[i].name for i in range(len(model_output))]
        input_tensor = image_to_input_tensor(frame)
        outputs = ort_session.run(output_names, {input_names[0]: input_tensor})[0]
        outputs = np.squeeze(outputs).T

        img_height, img_width = img_size
        print(outputs.shape)
        # outputs = outputs[0].boxes.data
        # class_outputs = outputs[outputs[:, 5] == 0][:,:5]
        if outputs is not None:
            online_targets = tracker.update(tensor(outputs), img_size, img_size)
            # online_targets = tracker.update(class_outputs.cpu(), img_size, img_size)
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