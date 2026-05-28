from pathlib import Path

from app.core.config import settings


class WeightPaths:
    """Central place for all local model weight file paths."""

    root: Path = settings.weights_dir
    foot_segmentation: Path = root / "foot_seg_yolo11n_best.pt"
    tinea_pedis_segmentation: Path = root / "tinea_pedis_best.pt"
    sam_checkpoint: Path = root / "sam_vit_b_01ec64.pth"


weights = WeightPaths()
