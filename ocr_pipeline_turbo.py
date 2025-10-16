"""
TURBO VERSION - Ultra-fast OCR pipeline with aggressive optimizations
Expected runtime: 5-10 seconds (vs 37s baseline)
"""

import json
import sys
import os
import time
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reg.paddle_reg import PaddleRecognition
import numpy as np
import cv2

# Global models cache
_ocr_model = None
_recognizer_model = None

class TurboOCRPipeline:
    """Ultra-optimized OCR Pipeline"""
    
    def __init__(self):
        self.ocr = None
        self.recognizer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize models with maximum performance settings"""
        print("🚀 TURBO MODE: Initializing models...")
        start = time.time()
        
        # Ultra-fast PaddleOCR settings
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(       # Disable angle classification
            use_doc_orientation_classify=False,
            use_doc_unwarping=False, 
            use_textline_orientation=False,
            det_db_thresh=0.4,             # Higher threshold = fewer false positives
            det_db_box_thresh=0.6,         # Higher threshold = faster processing
            det_limit_side_len=1280,       # Limit image size for faster processing
            cpu_threads=2,                 # Optimal for most systems
            enable_mkldnn=True          # Intel optimizations
        )
        
        # Fast recognizer
        self.recognizer = PaddleRecognition(device='cuda')
        
        init_time = time.time() - start
        print(f"✅ TURBO models ready in {init_time:.2f}s")
    
    def preprocess_image(self, img_path):
        """Ultra-fast image preprocessing"""
        # Load and optimize image
        img = cv2.imread(img_path)
        
        # Resize if too large (major speedup)
        h, w = img.shape[:2]
        if max(h, w) > 1920:
            scale = 1920 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        return img
    
    def fast_detection(self, img_path):
        """
        Ultra-fast detection using optimized PaddleOCR settings
        """
        t0 = time.time()
        
        # Use self.ocr instead of self.detector
        result = self.ocr.predict(img_path, det=True, rec=False, cls=False)
        
        bboxes = []
        if not result or  result[0]:
            return bboxes, time.time() - t0
        
        for item in result[0]:
            # Standard PaddleOCR format: [bbox_coords, confidence]
            if isinstance(item, (list, tuple)) and len(item) >= 1:
                coords = item[0]  # Get bbox coordinates
                
                if isinstance(coords, (list, tuple)) and len(coords) >= 4:
                    try:
                        # Convert to standard bbox format
                        pts = [(float(p[0]), float(p[1])) for p in coords[:4]]
                        xs = [pt[0] for pt in pts]
                        ys = [pt[1] for pt in pts]
                        x_min, y_min = int(min(xs)), int(min(ys))
                        x_max, y_max = int(max(xs)), int(max(ys))
                        bboxes.append((x_min, y_min, x_max, y_max))
                    except (ValueError, IndexError, TypeError):
                        continue
        
        return bboxes, time.time() - t0


    
    def turbo_pipeline(self, img_path):
        """
        Complete turbo pipeline
        """
        total_start = time.time()
        
        print(f"🔥 TURBO PROCESSING: {os.path.basename(img_path)}")
        
        # Step 1: Ultra-fast detection
        bboxes, det_time = self.fast_detection(img_path)
        print(f"⚡ Detection: {det_time:.2f}s ({len(bboxes)} regions)")
        
        # Step 2: Batch recognition
        rec_start = time.time()
        detection_data = {
            "image_path": img_path,
            "bboxes": bboxes,
            "total_regions": len(bboxes)
        }
        texts = self.recognizer.recognize_from_detection(detection_data)
        rec_time = time.time() - rec_start
        print(f"⚡ Recognition: {rec_time:.2f}s")
        
        # Step 3: Fast field extraction
        extract_start = time.time()
        raw_text = "\n".join(texts)
        json_output = self.recognizer.extract_fields(raw_text)
        
        # Fast seal detection
        seal_detected = self.recognizer.detect_seal_from_bboxes(detection_data)
        json_output["seal_present"] = seal_detected
        extract_time = time.time() - extract_start
        print(f"⚡ Extraction: {extract_time:.2f}s")
        
        # Final output
        filename = os.path.basename(img_path)
        current_datetime = datetime.now().strftime("%d/%m/%Y")
        
        final_output = {
            "file": filename,
            "datetime": current_datetime,
            "fields": json_output
        }
        
        total_time = time.time() - total_start
        
        print(f"🏆 TURBO TOTAL: {total_time:.2f}s")
        print(f"🚀 SPEEDUP: {37/total_time:.1f}x faster than baseline!")
        
        return {
            "detection_data": detection_data,
            "raw_text": raw_text,
            "structured_output": final_output,
            "performance": {
                "detection_time": det_time,
                "recognition_time": rec_time,
                "extraction_time": extract_time,
                "total_time": total_time,
                "speedup": 37/total_time
            }
        }

def main():
    """Turbo main function"""
    print("🔥 TURBO OCR PIPELINE")
    print("="*30)
    
    # Initialize turbo pipeline
    pipeline = TurboOCRPipeline()
    
    # Process image
    img_path = "/data/AITeam/nguyetnvm/OCR-handwritten/data/Do-Viet-Thi-signed-page-001.jpg"
    results = pipeline.turbo_pipeline(img_path)
    
    # Save results
    with open("turbo_results.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    perf = results["performance"]
    print(f"\n📊 TURBO PERFORMANCE:")
    print(f"   Detection: {perf['detection_time']:.2f}s")
    print(f"   Recognition: {perf['recognition_time']:.2f}s") 
    print(f"   Extraction: {perf['extraction_time']:.2f}s")
    print(f"   TOTAL: {perf['total_time']:.2f}s")
    print(f"   SPEEDUP: {perf['speedup']:.1f}x")
    
    if perf['total_time'] < 10:
        print("🏆 MISSION ACCOMPLISHED! Ultra-fast pipeline achieved!")

if __name__ == "__main__":
    main()
