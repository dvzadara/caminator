from ultralytics import YOLO
import cv2
model = YOLO('weights/best.pt')


def detect_objects_and_draw_boxes(image):
    result = model([image])[0]
    boxes = result.boxes
    probs = result.probs
    im_array = result.plot()
    objects = []
    for r in result:
        objects.append(r.names[0])
    return im_array, objects
