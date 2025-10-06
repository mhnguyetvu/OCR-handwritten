# ocr/paddle_detector.py
from paddleocr import PaddleOCR
import re
from utils.text_utils import clean_line

def init_paddle_ocr(lang="vi", use_textline_orientation=True):
    return PaddleOCR(use_textline_orientation=use_textline_orientation, lang=lang)

def find_rec_texts_from_result(result):
    """Extract rec_texts list from PaddleOCR result (robust to formats)."""
    texts = []
    if isinstance(result, (list, tuple)) and len(result) > 0:
        first = result[0]
        if isinstance(first, dict) and 'rec_texts' in first:
            texts = [clean_line(t) for t in first['rec_texts']]
        else:
            # flatten nested structures, collect strings
            def _walk(x):
                if isinstance(x, str):
                    return [clean_line(x)]
                if isinstance(x, (list, tuple)):
                    out = []
                    for e in x:
                        out.extend(_walk(e))
                    return out
                return []
            texts = _walk(result)
            texts = [t for t in texts if t and re.search(r"[A-Za-z\u00C0-\u1EF90-9]", t)]
    return texts

def extract_boxes(result):
    """
    Extract polygon boxes that look like [[x,y],[x,y],...]
    Returns list of polygon lists.
    """
    boxes = []
    candidates = result[0] if isinstance(result, list) and len(result) == 1 else result
    if not isinstance(candidates, (list, tuple)):
        return boxes
    for line in candidates:
        # direct polygon: list of points
        if isinstance(line, (list, tuple)) and len(line) > 0 and isinstance(line[0], (list, tuple)):
            # quick sanity check
            pt0 = line[0]
            try:
                float(pt0[0]); float(pt0[1])
                boxes.append(line)
            except Exception:
                continue
        else:
            # maybe polygon is nested inside
            if isinstance(line, (list, tuple)):
                for item in line:
                    if isinstance(item, (list, tuple)) and len(item) > 0 and isinstance(item[0], (list, tuple)):
                        try:
                            float(item[0][0]); float(item[0][1])
                            boxes.append(item)
                            break
                        except Exception:
                            continue
    return boxes
