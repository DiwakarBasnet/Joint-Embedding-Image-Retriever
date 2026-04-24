import torch
import json
from dataclasses import dataclass, asdict
from src.models.ijepa import IJEPATargetEncoder


@dataclass
class ViTConfig:
    img_size: int = 224
    in_chans: int = 3
    patch_size: int = 14
    embed_dim: int = 1280
    depth: int = 32
    num_heads: int = 16
    mlp_ratio: float = 4.0


def save_model_package(
    model: torch.nn.Module,
    config: ViTConfig,
    save_dir: str = "weights/ijepa_model_package"
):
    """
    Save model weights + config separately (smaller than full model).
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    # Save config as JSON
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)

    # Save weights only
    weights_path = os.path.join(save_dir, "model_weights.pth")
    torch.save(model.state_dict(), weights_path)

    # Check sizes
    weights_size = os.path.getsize(weights_path) / 1e9
    print(f"Model package saved to: {save_dir}")
    print(f"Weights size: {weights_size:.2f} GB")
    print(f"Config: {config_path}")

    return save_dir


def load_model_package(
    package_dir: str = "weights/ijepa-target-encoder",
    device: str = "cuda"
):
    """Load model from saved package (weights + config)."""
    import os

    # Load config
    config_path = os.path.join(package_dir, "config.json")
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    config = ViTConfig(**config_dict)

    # Create model with correct architecture
    model = IJEPATargetEncoder(
        img_size=config.img_size,
        patch_size=config.patch_size,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        mlp_ratio=config.mlp_ratio
    )

    # Load weights
    weights_path = os.path.join(package_dir, "model_weights.pth")
    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)

    # Move to device and set to eval
    # model = model.half().to(device).eval()
    model = model.to(device).eval()

    return model
