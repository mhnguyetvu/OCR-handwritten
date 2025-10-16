from core.detection import detect_text_regions
from core.filtering import smart_filter_regions
from core.recognition import recognize_fields
from core.utils import Timer
import os, json
from datetime import datetime

def run_pipeline(img_path, output_json=None):
    total = Timer()

    bboxes, scores, (H, W) = detect_text_regions(img_path)
    filtered = smart_filter_regions(bboxes, scores, W, H)

    detection_data = {"image_path": img_path, "bboxes": filtered, "total_regions": len(filtered)}
    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(detection_data, f, ensure_ascii=False, indent=2)

    texts, fields = recognize_fields(detection_data)
    structured = {
        "file": os.path.basename(img_path),
        "datetime": datetime.now().strftime("%d/%m/%Y"),
        "fields": fields,
    }

    print(f"✅ Done {os.path.basename(img_path)} in {total.lap():.2f}s ({len(filtered)} boxes kept)")
    return {"detection_data": detection_data, "raw_text": "\n".join(texts), "structured_output": structured}
