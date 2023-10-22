import numpy as np


def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def tlwh2xyxy(tlwh):
    x1 = tlwh[0]
    y1 = tlwh[1]
    x2 = tlwh[0] + tlwh[2]
    y2 = tlwh[1] + tlwh[3]
    return (x1, y1, x2, y2)
