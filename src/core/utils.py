import time

class Timer:
    def __init__(self): self.t0 = time.time()
    def lap(self): return time.time() - self.t0

def box_area(b): x1, y1, x2, y2 = b; return max(0, x2-x1) * max(0, y2-y1)

def iou(a, b):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1,bx1), max(ay1,by1), min(ax2,bx2), min(ay2,by2)
    iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
    inter = iw * ih
    if inter == 0: return 0.0
    return inter / max(1, (box_area(a)+box_area(b)-inter))
