import json
import cv2
import numpy as np
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import re
import os

class PaddleRecognition:
    def __init__(self, weights_path='vgg_transformer.pth', device='cpu'):
        """
        Initialize VietOCR recognition model
        Args:
            weights_path: Path to VietOCR weights file
            device: 'cpu' or 'cuda' for GPU
        """
        self.config = Cfg.load_config_from_name('vgg_transformer')
        self.config['weights'] = weights_path
        self.config['device'] = device
        self.detector = Predictor(self.config)
        print(f"VietOCR initialized with device: {device}")

    def load_detection_results(self, json_file_path):
        """
        Load detection results from JSON file
        Args:
            json_file_path: Path to JSON file containing detection results
        Returns:
            dict: Detection results with image_path and bboxes
        """
        with open(json_file_path, 'r', encoding='utf-8') as f:
            detection_data = json.load(f)
        return detection_data

    def recognize_from_detection(self, detection_data):
        """
        Perform text recognition on detected bounding boxes
        Args:
            detection_data: Dict containing image_path and bboxes
        Returns:
            list: List of recognized texts
        """
        image_path = detection_data['image_path']
        bboxes = detection_data['bboxes']
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        texts = []
        print(f"Processing {len(bboxes)} detected regions...")
        
        for i, bbox in enumerate(bboxes):
            try:
                # bbox format: (x_min, y_min, x_max, y_max)
                x_min, y_min, x_max, y_max = bbox
                
                # Crop the image region
                crop = image[y_min:y_max, x_min:x_max]
                if crop.size == 0:
                    print(f"Warning: Empty crop for bbox {i}: {bbox}")
                    continue
                
                # Convert numpy array to PIL Image for VietOCR
                from PIL import Image
                if len(crop.shape) == 3:
                    # BGR to RGB conversion for OpenCV to PIL
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(crop_rgb)
                else:
                    pil_image = Image.fromarray(crop)
                
                # Recognize text using VietOCR
                text = self.detector.predict(pil_image)
                texts.append(text)
                
            except Exception as e:
                print(f"Error processing bbox {i}: {e}")
                continue
        
        return texts

    def recognize_from_json_file(self, json_file_path):
        """
        Complete recognition pipeline from JSON file
        Args:
            json_file_path: Path to JSON file containing detection results
        Returns:
            tuple: (raw_text, json_output)
        """
        # Load detection results
        detection_data = self.load_detection_results(json_file_path)
        
        # Perform recognition
        texts = self.recognize_from_detection(detection_data)
        
        # Combine texts
        raw_text = "\n".join(texts)
        
        # Extract structured fields
        json_output = self.extract_fields(raw_text)
        
        return raw_text, json_output

    def extract_fields(self, text):
        """
        Extract structured fields from raw OCR text
        Args:
            text: Raw OCR text
        Returns:
            dict: Extracted fields
        """
        result = {}

        # Decision number - improved patterns
        patterns = [
            r"Số\s+(\d+\.\d+\.?\/QĐ-[A-ZĐÂĂƠƯ\-]+)",  # Số 14.6./QĐ-HĐQT
            r"(\d+\.\d+\.?\/QĐ-[A-ZĐÂĂƠƯ\-]+)",      # 14.6./QĐ-HĐQT
            r"(\d+\/QĐ-[A-ZĐÂĂƠƯ\-]+)",              # 123/QĐ-HĐQT
            r"(\d+\/[A-ZĐÂĂƠƯ\-]+)",                 # General pattern
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["no_decision"] = match.group(1)
                break

        # Decision date - multiple formats
        date_patterns = [
            r"ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})",  # ngày X tháng Y năm Z
            r"(\d{1,2})\/(\d{1,2})\/(\d{4})",                          # DD/MM/YYYY
            r"(\d{1,2})-(\d{1,2})-(\d{4})",                            # DD-MM-YYYY
            r"(\d{1,2})\s+(\d{1,2})\s+(\d{4})",                        # DD MM YYYY
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                if len(match.groups()) == 3:
                    day, month, year = match.groups()
                    result["decision_date"] = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                else:
                    result["decision_date"] = match.group(0)
                break

        # Company name - common patterns (extract just the company name, not the whole paragraph)
        company_patterns = [
            r"CÔNG TY CỔ PHẦN XÂY DỰNG BẢO TÀNG HỒ CHÍ MINH",
            r"Công ty Cổ phần Xây dựng Bảo tàng Hồ Chí Minh",
            r"CÔNG TY CỔ PHẦN[A-ZĐÂĂÊẾÎÔƠƯA-ZĐ\s]+",
            r"Công ty Cổ phần[A-Za-zĐÂĂÊẾÎÔƠƯàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ\s]+",
            r"CÔNG TY TNHH[A-ZĐÂĂÊẾÎÔƠƯA-ZĐ\s]+",
            r"Công ty TNHH[A-Za-zĐÂĂÊẾÎÔƠƯàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ\s]+",
        ]
        for pattern in company_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["company_name"] = match.group(0).strip()
                break

        # Appointee name - improved patterns
        name_patterns = [
            r"(?:bổ nhiệm|Bổ nhiệm)\s+(?:Ông|Bà|ông|bà)\s+([A-ZĐÂĂÊẾÎÔƠƯA-ZĐ][a-zA-ZĐÂĂÊẾÎÔƠƯàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ\s]+)",
            r"(?:Ông|Bà|ông|bà)\s+([A-ZĐÂĂÊẾÎÔƠƯA-ZĐ][a-zA-ZĐÂĂÊẾÎÔƠƯàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ\s]+)",
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["appointee_name"] = match.group(1).strip()
                break

        # Position - expanded patterns
        position_patterns = [
            r"Tổng Giám đốc",
            r"Giám đốc",
            r"Phó Giám đốc",
            r"Kế toán trưởng",
            r"Chủ tịch",
            r"Phó Chủ tịch",
            r"Thành viên HĐQT",
        ]
        for pattern in position_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["position"] = match.group(0)
                break

        # Signer name - look for signature patterns
        signer_patterns = [
            r"CHỦ TỊCH HỘI ĐỒNG QUẢN TRỊ\s*\n\s*([A-ZĐÂĂÊẾÎÔƠƯA-ZĐ][a-zA-ZĐÂĂÊẾÎÔƠƯàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ\s]+)",
            r"CHỦ TỊCH.*?\n\s*([A-ZĐÂĂÊẾÎÔƠƯA-ZĐ][a-zA-ZĐÂĂÊẾÎÔƠƯàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ\s]{1,30})",
            r"Phạm Minh [A-ZĐÂĂÊẾÎÔƠƯA-ZĐ][a-zA-ZĐÂĂÊẾÎÔƠƯàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ]*",
        ]
        for pattern in signer_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                result["signer_name"] = match.group(1).strip() if len(match.groups()) > 0 else match.group(0).replace("Phạm Minh ", "Phạm Minh ")
                break

        # Detect seal presence - look for circular patterns or seal-related text
        seal_indicators = [
            r"con dấu",
            r"dấu",
            r"seal",
            r"stamp",
        ]
        result["seal_present"] = any(re.search(pattern, text, re.IGNORECASE) for pattern in seal_indicators)

        return result

    def detect_seal_from_bboxes(self, detection_data):
        """
        Detect seal presence based on detection patterns
        Args:
            detection_data: Detection data containing bboxes
        Returns:
            bool: True if seal is likely present
        """
        bboxes = detection_data.get('bboxes', [])
        
        # Simple heuristics for seal detection:
        # 1. Check for many small regions (seal text tends to be small and curved)
        # 2. Check for circular/square patterns
        
        if len(bboxes) < 5:
            return False
            
        # Calculate area of each bbox
        areas = []
        for bbox in bboxes:
            if len(bbox) == 4:
                x_min, y_min, x_max, y_max = bbox
                area = (x_max - x_min) * (y_max - y_min)
                areas.append(area)
        
        if not areas:
            return False
            
        # If there are many small regions compared to average, likely has seal
        avg_area = sum(areas) / len(areas)
        small_regions = [a for a in areas if a < avg_area * 0.3]
        
        # If more than 20% of regions are very small, likely has seal
        seal_threshold = len(bboxes) * 0.2
        
        return len(small_regions) > seal_threshold

def main():
    # Example usage
    img_path = "C:\\Users\\nguyetnvm\\Documents\\26. Data OCR hand\\data\\Do-Viet-Thi-signed-page-001.jpg"
    
    # For demonstration, create sample detection data
    # In practice, this would come from your detection pipeline
    sample_detection_data = {
        "image_path": img_path,
        "bboxes": [
            # Add your bboxes here - format: (x_min, y_min, x_max, y_max)
            # These would normally come from paddle_det.py output
        ]
    }
    
    # Save sample detection results to JSON (for testing)
    json_output_path = "detection_results.json"
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(sample_detection_data, f, ensure_ascii=False, indent=2)
    
    # Initialize recognition
    recognizer = PaddleRecognition(device='cpu')
    
    # Method 1: From detection data directly
    if sample_detection_data['bboxes']:  # Only if we have bboxes
        texts = recognizer.recognize_from_detection(sample_detection_data)
        raw_text = "\n".join(texts)
        json_output = recognizer.extract_fields(raw_text)
        
        print("==== OCR RAW TEXT ====")
        print(raw_text)
        print("\n==== JSON OUTPUT ====")
        print(json_output)
    
    # Method 2: From JSON file
    # raw_text, json_output = recognizer.recognize_from_json_file(json_output_path)

if __name__ == "__main__":
    main()