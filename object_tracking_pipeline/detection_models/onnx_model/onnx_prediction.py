import cv2
import numpy as np
import torchvision
from ultralytics.utils.ops import non_max_suppression
from PIL import Image
import onnxruntime as rt
from datetime import datetime
import time
from torch import tensor

from object_tracking_pipeline.boxes_converters import xywh2xyxy


def load_model(model_path=r"weights/yolov8n.onnx_model"):
    # EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    EP_list = ['CPUExecutionProvider']

    ort_session = rt.InferenceSession(model_path, providers=EP_list)
    return ort_session


def get_model_params(ort_session):
    """
    Returns input_shape, input_names, output_names from ort_session
    """
    model_inputs = ort_session.get_inputs()
    input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    input_shape = model_inputs[0].shape

    model_output = ort_session.get_outputs()
    output_names = [model_output[i].name for i in range(len(model_output))]
    return input_shape, input_names, output_names


input_width, input_height = (640, 640)

ort_session = load_model("weights/yolov8n.onnx")
input_shape, input_names, output_names = get_model_params(ort_session)


def image_to_input_tensor(image):
    """
    Transforms opencv image to format for oonx model
    """
    Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (input_width, input_height))

    # Scale input pixel value to 0 to 1
    input_image = resized / 255.0
    input_image = input_image.transpose(2, 0, 1)
    input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
    return input_tensor


def predict(input_tensor, image_width, image_height, conf_thresold=0.3):
    """
    Uses onnx model for prediction.
    Returns:
    boxes - np.array, shape=(n, 4) every box is [x_center, y_center, width, height],
    scores - np.array, shape=(n) floats between 0 and 1,
    class_ids - np.array, shape=(n).
    """
    outputs = ort_session.run(output_names, {input_names[0]: input_tensor})[0]
    predictions = np.squeeze(outputs).T
    # Filter out object confidence scores below threshold
    scores = np.max(predictions[:, 4:], axis=1)
    predictions = predictions[scores > conf_thresold, :]
    scores = scores[scores > conf_thresold]

    class_ids = np.argmax(predictions[:, 4:], axis=1)

    # Get bounding boxes for each object
    boxes = predictions[:, :4]

    input_shape = np.array([input_width, input_height, input_width, input_height])
    boxes = np.divide(boxes, input_shape, dtype=np.float32)
    boxes *= np.array([image_width, image_height, image_width, image_height])
    boxes = boxes.astype(np.int32)
    return boxes, scores, class_ids


def nms(boxes, scores, iou_threshold):
    """
    Non maximum supression. Takes boxes(x_min, y_min, x_max, y_max) np.array with shape (n, 4),
    scores(floats between 0 and 1) np.array with shape (n),
    iou_threshold float between 0 and 1
    """
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou


CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella',
    'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror',
    'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def run_model(image, conf_thresold=0.3):
    """
    Process image and return model results with boxes, scores, class_ids
    """
    image_height, image_width = image.shape[:2]
    input_tensor = image_to_input_tensor(image)
    boxes, scores, class_ids = predict(input_tensor, image_width, image_height, conf_thresold)
    model_results = [boxes, scores, class_ids]
    return model_results


def results2boxes_and_probs(model_results):
    """
    Convert result of function run_model to format [[x_min, y_min, x_max, y_max, score], ...]
    """
    boxes, scores, class_ids = model_results
    # boxes_and_scores = non_max_suppression(tensor(model_results), 0.3)
    # boxes = boxes_and_scores[:, :4]
    # scores = boxes_and_scores[:, 4]
    # class_ids = boxes_and_scores[:, 5]
    boxes = xywh2xyxy(boxes)
    i = torchvision.ops.nms(tensor(boxes.astype(float)), tensor(scores.astype(float)), 0.1)  # NMS
    detections = []
    i = [i] if len(i) == 1 else i
    for (bbox, score, label) in zip(boxes[i], scores[i], class_ids[i]):
        x_min, y_min, x_max, y_max = list(bbox)
        confidence = score
        detections.append([int(x_min), int(y_min), int(x_max), int(y_max), float(confidence)])
    detections = np.array(detections)
    return detections
