from core.utils import box_area, iou
from config.settings import CFG

def smart_filter_regions(bboxes, scores, img_w, img_h):
    # Drop tiny boxes
    pairs = [(b, (scores[i] or box_area(b))) for i,b in enumerate(bboxes)
             if (b[2]-b[0]) > CFG.min_side_pixels and (b[3]-b[1]) > CFG.min_side_pixels]
    if not pairs: return []
    pairs.sort(key=lambda x: x[1], reverse=True)
    boxes = [b for b,_ in pairs[:CFG.max_keep_boxes]]
    areas = [box_area(b) for b in boxes]
    max_a = max(areas) if areas else 0
    kept = [b for b in boxes if box_area(b) >= CFG.area_keep_ratio * max_a]
    return kept
