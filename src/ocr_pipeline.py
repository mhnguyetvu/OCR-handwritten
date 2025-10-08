"""
Integration script to connect PaddleOCR detection with VietOCR recognition
"""

import json
import sys
import os
import time
from datetime import datetime

# Add parent directory to path to import paddle_det and paddle_reg
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reg.paddle_reg import PaddleRecognition

def run_detection_and_recognition(img_path, output_json_path=None):
    """
    Complete pipeline: Detection -> Recognition -> Field Extraction
    Args:
        img_path: Path to input image
        output_json_path: Optional path to save detection results
    """
    
    # Start total timing
    total_start_time = time.time()
    
    # Step 1: Run detection (import and run paddle_det logic)
    from paddleocr import PaddleOCR
    import cv2
    
    print("==== STEP 1: DETECTION ====")
    detection_start_time = time.time()
    
    # Initialize PaddleOCR
    ocr_init_start = time.time()
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    ocr_init_time = time.time() - ocr_init_start
    print(f"PaddleOCR initialization time: {ocr_init_time:.2f} seconds")
    
    # Run detection
    predict_start = time.time()
    result = ocr.predict(img_path)
    predict_time = time.time() - predict_start
    print(f"PaddleOCR prediction time: {predict_time:.2f} seconds")
    
    # Process results - handle OCRResult structure
    process_start = time.time()
    bboxes = []
    if result and len(result) > 0:
        # result[0] is OCRResult object, need to access .json["res"]
        try:
            if hasattr(result[0], 'json'):
                res = result[0].json["res"]
                dt_polys = res.get("dt_polys", [])
                
                for poly in dt_polys:
                    if len(poly) == 4 and len(poly[0]) == 2:  # 4 points, each with x,y
                        xs = [p[0] for p in poly]
                        ys = [p[1] for p in poly]
                        x_min, y_min = int(min(xs)), int(min(ys))
                        x_max, y_max = int(max(xs)), int(max(ys))
                        bboxes.append((x_min, y_min, x_max, y_max))
            else:
                # Fallback: try old format
                for line in result[0]:
                    if len(line) >= 1:
                        box_coords = line[0]
                        if isinstance(box_coords, list) and len(box_coords) >= 4:
                            xs = [p[0] for p in box_coords if isinstance(p, (list, tuple)) and len(p) >= 2]
                            ys = [p[1] for p in box_coords if isinstance(p, (list, tuple)) and len(p) >= 2]
                            if xs and ys:
                                x_min, y_min = int(min(xs)), int(min(ys))
                                x_max, y_max = int(max(xs)), int(max(ys))
                                bboxes.append((x_min, y_min, x_max, y_max))
        except Exception as e:
            print(f"Error processing detection results: {e}")
            print(f"Result structure: {type(result[0])}")
            if hasattr(result[0], 'json'):
                print(f"Available keys: {result[0].json.keys()}")
    
    process_time = time.time() - process_start
    detection_total_time = time.time() - detection_start_time
    
    print(f"Result processing time: {process_time:.2f} seconds")
    print(f"Detection total time: {detection_total_time:.2f} seconds")
    print(f"Detected {len(bboxes)} text regions")
    
    # Prepare detection data
    detection_data = {
        "image_path": img_path,
        "bboxes": bboxes,
        "total_regions": len(bboxes)
    }
    
    # Save detection results if requested
    if output_json_path:
        save_start = time.time()
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(detection_data, f, ensure_ascii=False, indent=2)
        save_time = time.time() - save_start
        print(f"Detection results saved to: {output_json_path} ({save_time:.2f}s)")
    
    # Step 2: Run recognition
    print("\n==== STEP 2: RECOGNITION ====")
    recognition_start_time = time.time()
    
    recognizer_init_start = time.time()
    recognizer = PaddleRecognition(device='cpu')
    recognizer_init_time = time.time() - recognizer_init_start
    print(f"VietOCR initialization time: {recognizer_init_time:.2f} seconds")
    
    recognize_start = time.time()
    texts = recognizer.recognize_from_detection(detection_data)
    recognize_time = time.time() - recognize_start
    recognition_total_time = time.time() - recognition_start_time
    
    print(f"Text recognition time: {recognize_time:.2f} seconds")
    print(f"Recognition total time: {recognition_total_time:.2f} seconds")
    
    # Step 3: Extract fields
    print("\n==== STEP 3: FIELD EXTRACTION ====")
    extraction_start_time = time.time()
    
    raw_text = "\n".join(texts)
    json_output = recognizer.extract_fields(raw_text)
    
    # Step 4: Detect seal
    seal_start = time.time()
    seal_detected = recognizer.detect_seal_from_bboxes(detection_data)
    json_output["seal_present"] = seal_detected
    seal_time = time.time() - seal_start
    
    extraction_total_time = time.time() - extraction_start_time
    print(f"Field extraction time: {extraction_total_time:.2f} seconds")
    print(f"Seal detection time: {seal_time:.2f} seconds")
    
    # Display results
    print("\n==== OCR RAW TEXT ====")
    print(raw_text)
    
    print("\n==== JSON OUTPUT ====")
    print(json.dumps(json_output, ensure_ascii=False, indent=2))
    
    # Create final structured output according to required format
    import os
    from datetime import datetime
    
    filename = os.path.basename(img_path)
    current_datetime = datetime.now().strftime("%d/%m/%Y")
    
    final_structured_output = {
        "file": filename,
        "datetime": current_datetime,
        "fields": json_output
    }
    
    # Calculate and display total time
    total_time = time.time() - total_start_time
    
    print(f"\n==== PERFORMANCE SUMMARY ====")
    print(f"PaddleOCR initialization: {ocr_init_time:.2f}s")
    print(f"PaddleOCR prediction: {predict_time:.2f}s") 
    print(f"Result processing: {process_time:.2f}s")
    print(f"Detection total: {detection_total_time:.2f}s")
    print(f"VietOCR initialization: {recognizer_init_time:.2f}s")
    print(f"Text recognition: {recognize_time:.2f}s")
    print(f"Recognition total: {recognition_total_time:.2f}s")
    print(f"Field extraction: {extraction_total_time:.2f}s")
    print(f"Seal detection: {seal_time:.2f}s")
    print(f"TOTAL PIPELINE TIME: {total_time:.2f}s")
    
    return {
        "detection_data": detection_data,
        "raw_text": raw_text,
        "structured_output": final_structured_output,
        "performance": {
            "ocr_init_time": ocr_init_time,
            "prediction_time": predict_time,
            "processing_time": process_time,
            "detection_total": detection_total_time,
            "vietocr_init_time": recognizer_init_time,
            "recognition_time": recognize_time,
            "recognition_total": recognition_total_time,
            "extraction_time": extraction_total_time,
            "seal_detection_time": seal_time,
            "total_time": total_time
        }
    }

def main():
    print("Starting OCR Pipeline...")
    main_start_time = time.time()
    
    img_path = "C:\\Users\\nguyetnvm\\Documents\\26. Data OCR hand\\data\\Do-Viet-Thi-signed-page-001.jpg"
    output_json = "detection_output.json"
    
    # Run complete pipeline
    results = run_detection_and_recognition(img_path, output_json)
    
    # Save final results
    save_start = time.time()
    final_output_path = "final_results.json"
    with open(final_output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    save_time = time.time() - save_start
    
    total_main_time = time.time() - main_start_time
    
    print(f"\nFinal results saved to: {final_output_path} ({save_time:.2f}s)")
    print(f"Total execution time: {total_main_time:.2f} seconds")

if __name__ == "__main__":
    main()