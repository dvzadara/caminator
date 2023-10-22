import cv2
import random

import numpy as np

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for x in range(10)]


def draw_boxes(image, my_tracker):
    image_draw = image.copy()
    for id in my_tracker.current_ids:
        bbox = my_tracker.box_history[id][-1].astype(int)
        color = colors[id % 10]
        cv2.rectangle(image_draw, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
        cv2.putText(image_draw,
                    f'HUMAN, id={int(id)}', (bbox[0], bbox[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, [225, 255, 255],
                    thickness=1)
    return image_draw


def draw_tracks(image, my_tracker, last_frames=50):
    image_draw = image.copy()
    for id in my_tracker.current_ids:
        box_history = np.array(my_tracker.box_history[id])
        box_history = box_history[-last_frames:]
        if box_history.shape[0] > 2:
            track_history = np.array([(box_history[:, 2] - box_history[:, 0]) / 2 + box_history[:, 0],
                                      (box_history[:, 3] - box_history[:, 1]) / 2 + box_history[:, 1]]).T
            track_history = track_history.reshape((-1, 1, 2))
            color = colors[id % 10]
            cv2.polylines(image_draw, np.int32([track_history]), isClosed=False, color=color, thickness=2)
    return image_draw


def draw_box(image, box, id):
    image_draw = image.copy()
    bbox = np.array(box).astype(int)
    color = colors[id % 10]
    cv2.rectangle(image_draw, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
    cv2.putText(image_draw,
                f'HUMAN, id={int(id)}', (bbox[0], bbox[1] - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60, [225, 255, 255],
                thickness=1)
    return image_draw