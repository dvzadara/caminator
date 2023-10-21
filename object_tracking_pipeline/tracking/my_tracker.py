from object_tracking_pipeline.tracking.sort_algorithm.sort import Sort


class MyTracker:
    """
    Class for containing tracks data.
    """
    def __init__(self):
        self.tracker = Sort()
        self.ids = []
        self.current_ids = []
        self.box_history = {}
        self.frame_number = 0

    def track_objects(self, detections):
        """
        Updates tracks data with new frame.
        """
        if len(detections) != 0:
            trackers = self.tracker.update(detections)
        else:
            trackers = self.tracker.update()
        self.current_ids = []
        for tracker in trackers:
            id = int(tracker[4])
            self.current_ids.append(id)
            if id in self.box_history:
                self.box_history[id].append(tracker[:4])
            else:
                self.box_history[id] = [tracker[:4]]
        return trackers