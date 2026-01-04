from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class DataConfig:
    path: str
    target: str
    test_size: float


@dataclass
class TrainConfig:
    model_type: str


@dataclass
class ArtifactsConfig:
    dir: str
    model_path: str
    metrics_path: str


@dataclass
class AppConfig:
    project_name: str
    random_state: int
    data: DataConfig
    train: TrainConfig
    artifacts: ArtifactsConfig
    business: dict



def load_config(path: str = "configs/config.yaml") -> AppConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at {cfg_path.resolve()}")

    raw = yaml.safe_load(cfg_path.read_text())

    return AppConfig(
        project_name=raw["project_name"],
        random_state=raw["random_state"],
        data=DataConfig(**raw["data"]),
        train=TrainConfig(**raw["train"]),
        artifacts=ArtifactsConfig(**raw["artifacts"]),
        business=raw["business"],

    )

@dataclass
class BusinessConfig:
    retention_offer_cost: float
    retention_success_prob: float
    churn_value: float
