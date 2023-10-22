import numpy as np
from torch import tensor

from object_tracking_pipeline.boxes_converters import tlwh2xyxy
from object_tracking_pipeline.tracking.sort_algorithm.sort import Sort
from .byte_track.byte_tracker import BYTETracker

sort_tracker_type = "SORT"
byte_tracker_type = "BYTETRACKER"


class ByteTrackArgument:
    track_thresh = 0.7  # High_threshold
    track_buffer = 50  # Number of frame lost tracklets are kept
    match_thresh = 0.8  # Matching threshold for first stage linear assignment
    aspect_ratio_thresh = 10.0  # Minimum bounding box aspect ratio
    min_box_area = 1.0  # Minimum bounding box area
    mot20 = False  # If used, bounding boxes are not clipped.


class MyTracker:
    """
    Class for containing and updating tracks data. Uses sort or bytetrack algorithm(tracker_type argument).
    """

    def __init__(self, tracker_type=sort_tracker_type):
        self.tracker_type = tracker_type
        if self.tracker_type == sort_tracker_type:
            self.tracker = Sort()
        elif self.tracker_type == byte_tracker_type:
            self.tracker = BYTETracker(ByteTrackArgument)
        self.ids = []
        self.current_ids = []
        self.box_history = {}
        self.frame_number = 0

    def track_objects(self, detections):
        """
        Updates tracks data with new frame.
        """
        if self.tracker_type == sort_tracker_type:
            if len(detections) != 0:
                trackers = self.tracker.update(detections)
            else:
                trackers = self.tracker.update()
            self.current_ids = []
            for tracker in trackers:
                track_id = int(tracker[4])
                self.current_ids.append(track_id)
                if track_id in self.box_history:
                    self.box_history[track_id].append(tracker[:4])
                else:
                    self.box_history[track_id] = [tracker[:4]]

        elif self.tracker_type == byte_tracker_type:
            if len(detections) != 0:
                online_targets = self.tracker.update(tensor(detections))
                self.current_ids = []
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    self.current_ids.append(tid)
                    vertical = tlwh[2] / tlwh[3] > ByteTrackArgument.aspect_ratio_thresh
                    if tlwh[2] * tlwh[3] > ByteTrackArgument.min_box_area and not vertical:
                        box = np.array(tlwh2xyxy(tlwh))
                        if tid in self.box_history:
                            self.box_history[tid].append(box)
                        else:
                            self.box_history[tid] = [box]
