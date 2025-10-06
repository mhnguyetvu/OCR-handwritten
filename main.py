# main.py
import argparse
import json
import os
import sys
import cv2
import time
from utils.logger import get_logger
from ocr.paddle_detector import init_paddle_ocr, find_rec_texts_from_result, extract_boxes
from ocr.vietocr_recognizer import load_vietocr
from extractors.field_extractor import extract_fields_from_lines

logger = get_logger("ocr_pipeline")

def parse_args():
    p = argparse.ArgumentParser(description="OCR pipeline: PaddleOCR -> VietOCR -> field extraction")
    p.add_argument("--image", "-i", required=True, help="Path to input image")
    p.add_argument("--weights", "-w", default="weights/vgg_transformer.pth", help="Path to VietOCR weights")
    p.add_argument("--device", "-d", default="cpu", choices=["cpu", "cuda"], help="Device for VietOCR")
    p.add_argument("--out", "-o", default="result.json", help="Output JSON path")
    p.add_argument("--lang", default="vi", help="Language for PaddleOCR (default: vi)")
    p.add_argument("--no-vietocr", action="store_true", help="Skip VietOCR cropping; rely on PaddleOCR rec_texts only")
    return p.parse_args()

def main():
    args = parse_args()
    start = time.time()

    if not os.path.exists(args.image):
        logger.error("Image not found: %s", args.image)
        sys.exit(1)

    image = cv2.imread(args.image)
    if image is None:
        logger.error("cv2.imread returned None for %s", args.image)
        sys.exit(1)

    ocr = init_paddle_ocr(lang=args.lang)
    logger.info("Running PaddleOCR on %s", args.image)
    result = ocr.predict(args.image)

    # 1) collect PaddleOCR recognized text lines
    lines = find_rec_texts_from_result(result)
    logger.info("PaddleOCR returned %d text lines", len(lines))

    # 2) optionally run VietOCR over detected boxes for better recognition
    texts_from_viet = []
    if not args.no_vietocr:
        # ensure weights exist
        try:
            vietocr = load_vietocr(args.weights, device=args.device)
        except FileNotFoundError as e:
            logger.error(str(e))
            logger.info("Falling back to PaddleOCR-recognized lines only.")
            vietocr = None

        boxes = extract_boxes(result)
        logger.info("Extracted %d polygon boxes", len(boxes))

        if vietocr and boxes:
            h, w = image.shape[:2]
            for i, box in enumerate(boxes):
                # compute bbox
                x_min = int(min(p[0] for p in box))
                y_min = int(min(p[1] for p in box))
                x_max = int(max(p[0] for p in box))
                y_max = int(max(p[1] for p in box))
                # clamp
                x_min = max(0, min(w-1, x_min)); x_max = max(0, min(w, x_max))
                y_min = max(0, min(h-1, y_min)); y_max = max(0, min(h, y_max))
                if x_max <= x_min or y_max <= y_min:
                    continue
                crop = image[y_min:y_max, x_min:x_max]
                try:
                    t = vietocr.predict(crop)
                    texts_from_viet.append(t)
                except Exception as ex:
                    logger.warning("VietOCR failed on crop #%d: %s", i, ex)

    # Prefer VietOCR texts if present, else PaddleOCR lines
    final_lines = texts_from_viet if texts_from_viet else lines

    # Extract fields
    extracted = extract_fields_from_lines(final_lines)

    # Ensure keys exist with None defaults
    keys = ["decision_number","decision_date","appointee_name","position","term","company"]
    json_output = {k: extracted.get(k) for k in keys}

    # Save JSON
    out_path = args.out
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(json_output, f, ensure_ascii=False, indent=2)

    logger.info("Saved JSON to %s", out_path)
    logger.info("Runtime: %.2fs", time.time() - start)

if __name__ == "__main__":
    main()
