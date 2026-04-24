import gc
import torch
from config import settings
from src.models.ijepa import IJEPATargetEncoder
from utils.saving_loading_util import save_model_package, load_model_package, ViTConfig


class IJEPAManager:
    def __init__(self, model_key: str = "huge"):
        self.model = None
        self.device = settings.device
        self.checkpoint_path = settings.checkpoint_path
        self.model_params = settings.model_architectures.get(model_key, settings.model_architectures["huge"])

    def load_model(self):
        """Standardized loading sequence with memory management."""
        if self.model is not None:
            return self.model

        print(f"1. Initializing ViT with params: {self.model_params}...")
        self.model = IJEPATargetEncoder(
            **self.model_params
        )

        print(f"2. Loading pre-trained weights from {self.checkpoint_path}...")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        print("3. Getting target encoder weights...")
        state_dict = checkpoint.get('target_encoder', checkpoint)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

        print("4. Loading weights into model...")
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        # Immediate cleanup
        del checkpoint
        del state_dict
        gc.collect()
        torch.cuda.empty_cache()

        print("Model loaded successfully.")
        return self.model

    def save_target_encoder(self):
        """Save the target encoder weights to disk."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        config = ViTConfig(
            img_size=224,
            in_chans=3,
            patch_size=self.model_params['patch_size'],
            embed_dim=self.model_params['embed_dim'],
            depth=self.model_params['depth'],
            num_heads=self.model_params['num_heads'],
            mlp_ratio=self.model_params['mlp_ratio']
        )
        save_model_package(self.model, config, settings.model_path)

    def load_target_encoder(self):
        """Load the target encoder weights from disk."""
        self.model = load_model_package(
            package_dir=settings.model_path,
            device=self.device
        )
        print("Model loaded successfully.")
        return self.model

    def get_features(self, processed_image):
        """Standard inference call."""
        if self.model is None:
            self.load_target_encoder()

        with torch.no_grad():
            if processed_image.device != self.device:
                processed_image = processed_image.to(self.device)
            return self.model(processed_image)
