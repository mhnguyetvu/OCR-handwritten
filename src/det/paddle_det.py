from paddleocr import PaddleOCR
import cv2
import numpy as np
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import re
import matplotlib.pyplot as plt
# -------------------
# 1. PaddleOCR DETECTION
# -------------------
# Initialize PaddleOCR with detection only
ocr = PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)
img_path = "C:\\Users\\nguyetnvm\\Documents\\26. Data OCR hand\\data\\Do-Viet-Thi-signed-page-001.jpg"

# Use the correct method call
result = ocr.predict(img_path)

# Debug: Print result structure
print("==== DEBUG: Result Structure ====")
print(f"Result type: {type(result)}")
print(f"Result length: {len(result) if result else 0}")
if result and len(result) > 0:
    print(f"First page type: {type(result[0])}")
    # Access OCRResult attributes properly
    ocr_result = result[0]
    if hasattr(ocr_result, 'json'):
        print(f"OCRResult has json attribute")
        print(f"JSON keys: {ocr_result.json.keys() if hasattr(ocr_result.json, 'keys') else 'No keys'}")

# Handle the OCRResult structure properly
image = cv2.imread(img_path)
bboxes = []
boxes = []

if result and result[0] is not None:
    ocr_result = result[0]
    
    # Try to access the detection results
    if hasattr(ocr_result, 'json') and 'res' in ocr_result.json:
        # New format: use json.res.dt_polys
        res = ocr_result.json['res']
        if 'dt_polys' in res:
            dt_polys = np.array(res['dt_polys'], dtype=float)
            for poly in dt_polys:
                boxes.append(poly.tolist())  # Convert to list of [x,y] points
                
                # Convert polygon to bounding box
                xs, ys = poly[:, 0], poly[:, 1]
                x_min, y_min = int(xs.min()), int(ys.min())
                x_max, y_max = int(max(xs)), int(max(ys))
                bboxes.append((x_min, y_min, x_max, y_max))

print(f"Detected {len(boxes)} text regions")

########
# Hiển thị kết quả box
for box in boxes:
    # box is a list of 4 points: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    pts = np.array([[int(p[0]), int(p[1])] for p in box])
    cv2.polylines(image, [pts], True, (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
