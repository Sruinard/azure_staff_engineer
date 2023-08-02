import dataclasses

@dataclasses.dataclass
class PipelineConfig:
    artifact_dir: str
    train_split: float
    val_split: float
    test_split: float

@dataclasses.dataclass
class ModelConfig:
    pass

@dataclasses.dataclass
class Config:
    pipeline_config: PipelineConfig
    model_config: ModelConfig
