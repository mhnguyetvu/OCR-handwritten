from pipeline.ocr_pipeline import run_pipeline
from core.models import ModelSuite
import os, json
from core.utils import Timer

def run_batch(img_paths, output_dir="outputs"):
    ModelSuite.get_detector()
    ModelSuite.get_recognizer()
    total = Timer()
    results = []
    for i, img_path in enumerate(img_paths):
        print(f"\n[{i+1}/{len(img_paths)}] {os.path.basename(img_path)}")
        res = run_pipeline(img_path)
        results.append(res)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "batch_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"🎉 Batch completed in {total.lap():.2f}s ({len(img_paths)} images)")
