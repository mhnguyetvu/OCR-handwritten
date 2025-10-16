from dataclasses import dataclass

@dataclass(frozen=True)
class PipelineConfig:
    use_gpu_for_paddle: bool = True
    use_gpu_for_recog: bool = True
    det_limit_side_len: int = 960
    cpu_threads: int = 4
    enable_mkldnn: bool = True
    det_db_thresh: float = 0.3
    det_db_box_thresh: float = 0.5
    max_keep_boxes: int = 24
    area_keep_ratio: float = 0.2
    min_side_pixels: int = 12
    enable_nms: bool = True
    nms_iou_thresh: float = 0.3

CFG = PipelineConfig()
