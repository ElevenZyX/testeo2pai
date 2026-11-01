from dataclasses import dataclass, field
from typing import Optional, Tuple, List


@dataclass
class BlackboxConfig:
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    prompt: str = "Describe the clinical findings and abnormalities in this medical scan."
    max_new_tokens: int = 256
    batch_size: int = 1
    timeout: int = 30


@dataclass
class JudgeConfig:
    model_name: str = "Qwen/Qwen3-32B"
    mode: str = "mtr"  # "mtr" (Paper Appendix A.9 Clinical Perturbation Evaluation) or "similarity"
    alpha: float = 0.5
    beta: float = 0.5


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "medfocusleak_local"
    entity: Optional[str] = None


@dataclass
class DataConfig:
    batch_size: int = 1
    num_samples: int = 5
    cle_data_path: str = "resources/images/bigscale"       # path for clean images
    tgt_data_path: str = "resources/images/target_images"  # path for target images
    mask_data_path: str = "resources/images/masks"         # path for masks
    output: str = "./img_output"
    modalities: Optional[List[str]] = field(default_factory=lambda: ["Mammography", "MRI"])


@dataclass
class OptimConfig:
    alpha: float = 1.0
    epsilon: int = 16
    steps: int = 300


@dataclass
class ModelConfig:
    k: int = 10
    non_overlap: bool = True
    input_res: int = 336
    use_source_crop: bool = True
    use_target_crop: bool = True
    crop_scale: Tuple[float, float] = (0.5, 0.9)
    ensemble: bool = True
    device: str = "cuda:0"
    backbone: List[str] = field(default_factory=lambda: ["L336", "B16", "B32", "Laion"])


@dataclass
class MainConfig:
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    blackbox: BlackboxConfig = field(default_factory=BlackboxConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    attack: str = "fgsm"


@dataclass
class Ensemble3ModelsConfig(MainConfig):
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        use_source_crop=True,
        use_target_crop=True,
        backbone=["B16", "B32", "Laion"]
    ))
