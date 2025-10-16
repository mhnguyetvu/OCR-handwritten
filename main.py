from pipeline.ocr_pipeline import run_pipeline
from core.batch import run_batch

def main():
    img = "data/Do-Viet-Thi-signed-page-001.jpg"
    run_pipeline(img, output_json="outputs/detection.json")

if __name__ == "__main__":
    main()