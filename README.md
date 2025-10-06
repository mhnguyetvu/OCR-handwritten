# OCR Handwritten Project

This project provides an end-to-end pipeline for extracting and processing text from scanned handwritten documents, specifically appointment forms in Vietnamese. It combines PaddleOCR for detection and VietOCR for recognition, with post-processing and rule-based field extraction.

## Features
- Text detection using PaddleOCR
- Text recognition using VietOCR (with custom weights)
- Rule-based extraction of key fields (decision number, date, appointee name, position, term, company)
- Utility functions for cleaning and normalizing OCR output
- Modular code structure for easy extension

## Project Structure
```
OCR-handwrite/
├── pipeline-ocr.py           # Main pipeline script
├── requirements.txt          # Python dependencies
├── data/                     # Input images (e.g., Appointment Form.jpg)
├── utils/
│   ├── __init__.py
│   ├── ocr_utils.py
│   └── text_utils.py         # Text cleaning utilities
├── weights/
│   └── vgg_transformer.pth   # VietOCR model weights (not tracked in git)
```

## Getting Started
### 1. Install dependencies
```sh
pip install -r requirements.txt
```

### 2. Download model weights
- Place `vgg_transformer.pth` in the `weights/` directory. (File is not included in repo; download from official VietOCR sources or request from the author.)

### 3. Run the pipeline
```sh
python main.py --image "/Users/mhnguyetvu/workspace/OCR-handwrite/data/Appointment Form.jpg" \
               --weights weights/vgg_transformer.pth \
               --out results.json
```

## Notes
- Large files (e.g., model weights) are excluded from git tracking via `.gitignore`.
- For files >100MB, use external storage or Git LFS if needed.
- The pipeline expects input images in the `data/` folder.

## References
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [VietOCR](https://github.com/quantra/VietOCR)

## License
MIT License
