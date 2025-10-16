from paddleocr import PaddleOCR
from reg.paddle_reg import PaddleRecognition
from config.settings import CFG
import time

class ModelSuite:
    _det = None
    _rec = None

    @classmethod
    def get_detector(cls):
        if cls._det is None:
            print("🚀 Initializing PaddleOCR detector...")
            t0 = time.time()
            cls._det = PaddleOCR(
                # use_angle_cls=False,
                use_textline_orientation=False,
                det_limit_side_len=CFG.det_limit_side_len,
                cpu_threads=CFG.cpu_threads,
                enable_mkldnn=CFG.enable_mkldnn,
                # use_gpu=CFG.use_gpu_for_paddle,
                det_db_thresh=CFG.det_db_thresh,
                det_db_box_thresh=CFG.det_db_box_thresh,
            )
            print(f"✅ Detector ready in {time.time() - t0:.2f}s")
        return cls._det

    @classmethod
    def get_recognizer(cls):
        if cls._rec is None:
            print("🚀 Initializing VietOCR recognizer...")
            t0 = time.time()
            device = "cuda" if CFG.use_gpu_for_recog else "cpu"
            cls._rec = PaddleRecognition(device=device)
            print(f"✅ Recognizer ready in {time.time() - t0:.2f}s")
        return cls._rec
