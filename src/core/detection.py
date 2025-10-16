from core.models import ModelSuite
import cv2

def detect_text_regions(img_path):
    ocr = ModelSuite.get_detector()
    img = cv2.imread(img_path)
    result = ocr.ocr(img, det=True, rec=False, cls=False)

    bboxes, scores = [], []
    if result and len(result) > 0:
        for line in result[0]:
            if not line: continue
            poly, sc = line[0], line[1] if len(line) > 1 else None
            xs, ys = [p[0] for p in poly], [p[1] for p in poly]
            bboxes.append((int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))))
            scores.append(float(sc) if sc is not None else None)
    return bboxes, scores, img.shape[:2]
