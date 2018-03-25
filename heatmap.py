import numpy as np

class HeatMap:
    def __init__(self, threshold):
        self.heatmap = np.zeros((720, 1280), dtype=float)
        self.threshold = threshold

    def add_heat(self, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    def apply_threshold(self, threshold=None):
        if threshold == None:
            threshold = self.threshold
        # Zero out pixels below the threshold
        self.heatmap[self.heatmap <= threshold] = 0

    def get_headmap(self):
        return np.clip(self.heatmap, 0, 255)