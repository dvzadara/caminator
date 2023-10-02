import random

import cv2
from object_tracking_pipeline.tracking.sort.sort import Sort
from ultralytics import YOLO
import numpy as np

model = YOLO(r'weights/best.pt')
tracker = Sort()

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for x in range(10)]


# Function to detect objects using YOLOv8
def detect_objects(image):
    results = model([image])[0]

    boxes = results.boxes
    probs = results.probs
    detections = []
    for i in range(len(boxes)):
        x_min, y_min, x_max, y_max = list(boxes[i].xyxy[0])
        confidence = boxes[i].conf
        class_id = boxes[i].id
        detections.append([int(x_min), int(y_min), int(x_max), int(y_max), float(confidence)])
    detections = np.array(detections)
    print(detections.shape)
    return detections


# Function to draw bounding boxes on the image
def draw_boxes(image, boxes):
    for i in range(len(boxes)):
        box = boxes[i]
        color = colors[i % len(colors)]
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)


def draw_tracks(image, trackers):
    for tracker in trackers:
        object_id = int(tracker[4])
        color = colors[object_id % 10]
        center_x, center_y = int((tracker[0] + tracker[2]) / 2), int((tracker[1] + tracker[3]) / 2)
        cv2.circle(image, (center_x, center_y), 2, color, -1)


def main():
    # Initialize camera capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects
        detections = detect_objects(frame)

        # Update SORT tracker
        if len(detections) != 0:
            trackers = tracker.update(detections)
        else:
            trackers = tracker.update()

        # Draw bounding boxes
        draw_boxes(frame, trackers[:, :4])
        draw_tracks(frame, trackers)

        # Display the resulting frame
        cv2.imshow('Frame', frame)

        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
