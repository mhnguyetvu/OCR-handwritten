"""
Example usage of paddle_reg.py with detection results
This script shows how to use the recognition module with JSON input from detection
"""

import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reg.paddle_reg import PaddleRecognition

def create_sample_detection_data():
    """
    Create sample detection data for testing
    In practice, this data would come from paddle_det.py
    """
    img_path = "C:\\Users\\nguyetnvm\\Documents\\26. Data OCR hand\\data\\Do-Viet-Thi-signed-page-001.jpg"
    
    # Sample bounding boxes (these would come from your detection step)
    sample_bboxes = [
        (100, 50, 300, 80),    # Header area
        (50, 100, 400, 130),   # Decision number line
        (50, 150, 350, 180),   # Date line
        (50, 200, 450, 230),   # Name line
        (50, 250, 400, 280),   # Position line
        # Add more bboxes as needed
    ]
    
    detection_data = {
        "image_path": img_path,
        "bboxes": sample_bboxes,
        "total_regions": len(sample_bboxes)
    }
    
    return detection_data

def example_usage():
    print("==== EXAMPLE: Using paddle_reg.py ====")
    
    # Method 1: Using detection data directly
    print("\n--- Method 1: Direct usage with detection data ---")
    
    # Create or load detection data
    detection_data = create_sample_detection_data()
    
    # Initialize recognizer
    recognizer = PaddleRecognition(device='cpu')
    
    # Run recognition
    try:
        texts = recognizer.recognize_from_detection(detection_data)
        raw_text = "\n".join(texts)
        structured_output = recognizer.extract_fields(raw_text)
        
        print("\n==== RAW OCR TEXT ====")
        print(raw_text)
        
        print("\n==== STRUCTURED OUTPUT ====")
        print(json.dumps(structured_output, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"Error in recognition: {e}")
    
    # Method 2: Using JSON file
    print("\n--- Method 2: Using JSON file ---")
    
    # Save detection data to JSON file
    json_file_path = "sample_detection.json"
    with open(json_file_path, 'w', encoding='utf-8') as f:
        json.dump(detection_data, f, ensure_ascii=False, indent=2)
    
    print(f"Sample detection data saved to: {json_file_path}")
    
    try:
        # Load from JSON and process
        raw_text, structured_output = recognizer.recognize_from_json_file(json_file_path)
        
        print("\n==== RAW OCR TEXT (from JSON) ====")
        print(raw_text)
        
        print("\n==== STRUCTURED OUTPUT (from JSON) ====")
        print(json.dumps(structured_output, ensure_ascii=False, indent=2))
        
    except Exception as e:
        print(f"Error processing JSON file: {e}")

def integration_example():
    """
    Example of how to integrate with actual detection results
    """
    print("\n==== INTEGRATION EXAMPLE ====")
    
    # This is how you would integrate with real detection results
    # from your paddle_det.py or similar detection script
    
    detection_json_content = """
    {
        "image_path": "C:\\\\Users\\\\nguyetnvm\\\\Documents\\\\26. Data OCR hand\\\\data\\\\Do-Viet-Thi-signed-page-001.jpg",
        "bboxes": [
            [50, 100, 400, 130],
            [50, 150, 350, 180],
            [50, 200, 450, 230]
        ],
        "total_regions": 3
    }
    """
    
    # Save to file
    integration_json_path = "integration_detection.json"
    with open(integration_json_path, 'w', encoding='utf-8') as f:
        f.write(detection_json_content)
    
    # Process with paddle_reg
    recognizer = PaddleRecognition(device='cpu')
    
    try:
        raw_text, structured_output = recognizer.recognize_from_json_file(integration_json_path)
        
        print("Integration successful!")
        print(f"Extracted {len(structured_output)} fields")
        
        # Save results
        results = {
            "raw_text": raw_text,
            "structured_fields": structured_output,
            "processing_status": "success"
        }
        
        results_path = "integration_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Results saved to: {results_path}")
        
    except Exception as e:
        print(f"Integration error: {e}")

if __name__ == "__main__":
    # Run examples
    example_usage()
    integration_example()
    
    print("\n==== FILES CREATED ====")
    print("- sample_detection.json: Sample detection data")
    print("- integration_detection.json: Integration example")
    print("- integration_results.json: Final results")