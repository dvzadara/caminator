import numpy as np
from ultralytics import YOLO
import cv2
model = YOLO('weights/best.pt')


def run_model(image):
    """
    Process image and return model results with boxes, scores, class_ids
    """
    model_results = model([image])[0]
    return model_results


def results2boxes_and_probs(model_results):
    """
    Convert result of function run_model to format [[x_min, y_min, x_max, y_max, score], ...]
    """
    boxes = model_results.boxes
    detections = []
    for i in range(len(boxes)):
        x_min, y_min, x_max, y_max = list(boxes[i].xyxy[0])
        confidence = boxes[i].conf
        detections.append([int(x_min), int(y_min), int(x_max), int(y_max), float(confidence)])
    detections = np.array(detections)
    return detections


# def detect_objects_and_draw_boxes(image):
#     result = model([image])[0]
#     boxes = result.boxes
#     probs = result.probs
#     im_array = result.plot()
#     objects = []
#     for r in result:
#         objects.append(r.names[0])
#     return im_array, objects


# def draw_boxes(image, model_results):
#     image_draw = image.copy()
#     boxes = model_results.boxes
#     for bbox in boxes:
#         bbox = bbox.xyxy[0]
#         score = bbox.conf
#         cls_id = bbox.id
#         color = (0, 255, 0)
#         cv2.rectangle(image_draw, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
#         cv2.putText(image_draw,
#                     f'Human:{int(score * 100)}', (bbox[0], bbox[1] - 2),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.60, [225, 255, 255],
#                     thickness=1)