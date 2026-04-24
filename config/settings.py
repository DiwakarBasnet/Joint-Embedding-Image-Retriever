import torch
from pathlib import Path
from typing import Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # Pinecone Settings (Loaded from .env)
    pinecone_api_key: str
    pinecone_cloud: str
    pinecone_region: str
    pinecone_index_name: str

    project_root: Path = Field(default=Path(__file__).parent.parent)

    @property
    def data_dir(self) -> Path: return self.project_root / "data"

    @property
    def models_dir(self) -> Path: return self.project_root / "models"

    # Model & Inference Settings
    model_type: str = "huge"
    batch_size: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True
    embedding_dim: int = 5120
    checkpoint_path: str = "weights/IN22K-vit.h.14-900e.pth.tar"
    model_path: str = "weights/ijepa-target-encoder"

    # Model Architectures
    model_architectures: Dict[str, Dict[str, Any]] = {
        "huge": {
            "embed_dim": 1280,
            "depth": 32,
            "num_heads": 16,
            "patch_size": 14,
            "mlp_ratio": 4,
        },
        "giant": {
            "embed_dim": 1408,
            "depth": 40,
            "num_heads": 16,
            "patch_size": 16,
            "mlp_ratio": 4.36,
        },
    }

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    @property
    def active_model_params(self) -> Dict[str, Any]:
        return self.model_architectures.get(self.model_type, {})


# Create the singleton instance
settings = Settings()
