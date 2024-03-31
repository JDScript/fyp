from dataclasses import dataclass, field
from typing import Optional, Any


@dataclass
class _ClassObjectConfig:
    target: str = ""
    params: Optional[dict[str, Any]] = field(default_factory=dict)


@dataclass
class _ModelConfig:
    num_views: int
    pretrained: str
    pretrained_unet: Optional[str]
    unet: _ClassObjectConfig


@dataclass
class _DatasetsConfig:
    train_dataset: _ClassObjectConfig
    val_dataset: _ClassObjectConfig
    train_val_dataset: _ClassObjectConfig


@dataclass
class _TrainingConfig:
    resume_from_checkpoint: Optional[str]
    trainable_modules: Optional[list]
    output_dir: str
    mixed_precision: str = "fp16"
    use_ema: bool = True
    log_with: str = "tensorboard"
    batch_size: int = 4
    max_train_steps: int = 20000
    gradient_accumulation_steps: int = 16
    gradient_checkpointing: bool = True
    checkpointing_steps: int = 5000
    learning_rate: float = 1e-4
    optimizer: _ClassObjectConfig = field(default_factory=_ClassObjectConfig)
    lr_scheduler: _ClassObjectConfig = field(default_factory=_ClassObjectConfig)
    condition_drop_rate: float = 0.05
    condition_drop_type: str = "drop_as_a_whole"
    snr_gamma: float = 5.0
    max_grad_norm: float = 1.0
    validation_steps: int = 1250
    validation_sanity_check: bool = True


@dataclass
class Config:
    name: str
    model: _ModelConfig
    datasets: _DatasetsConfig
    training: _TrainingConfig
