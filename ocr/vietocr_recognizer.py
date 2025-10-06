# ocr/vietocr_recognizer.py
from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import os

def load_vietocr(weights_path: str, device: str = "cpu"):
    """
    Load VietOCR predictor. Raises FileNotFoundError if weights missing.
    """
    config = Cfg.load_config_from_name("vgg_transformer")
    config["weights"] = weights_path
    config["device"] = device
    if not os.path.exists(config["weights"]):
        raise FileNotFoundError(f"VietOCR weights not found: {config['weights']}")
    return Predictor(config)
