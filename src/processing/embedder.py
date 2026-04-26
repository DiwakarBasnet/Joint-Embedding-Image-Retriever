import tqdm
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import List, Union, Tuple, Optional
from torch.utils.data import DataLoader, Dataset


class ImageEmbeddingDataset(Dataset):
    """Dataset for batch image embedding generation"""
    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        transform=None
    ):
        self.image_paths = [Path(p) for p in image_paths]
        self.transform = transform or self.default_transform()

    @staticmethod
    def default_transform():
        # I-JEPA uses mean=05 and std=0.5 normalization
        return transforms.Compose([
            transforms.Resize(
                224, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, str(img_path)


class EmbeddingGenerator:
    """Generate embeddings for image database using batch inference."""
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        batch_size: int = 4,
        num_workers: int = 1,
        layer_strategy: str = "second_last",
        specific_indices: Optional[List[int]] = None,
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.layer_strategy = layer_strategy
        self.specific_indices = specific_indices

        # Freeze model
        for param in self.model.parameters():
            param.requires_grad = False

    def _get_features(self, images: torch.Tensor) -> torch.Tensor:
        """Central routing method - all forward calss go through here."""
        if self.layer_strategy == "last":
            return self.model(images)
        return self.model.get_layer_representations(
            images,
            strategy=self.layer_strategy,
            specific_indices=self.specific_indices,
        )

    @torch.no_grad()
    def generate_embeddings(
        self,
        image_paths: List[Union[str, Path]],
        return_paths: bool = True,
        show_progress: bool = True,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[str]]]:
        """
        Generate embeddings for all images.

        Returns:
            embeddings: (N, D) array of embeddings
            paths: (optional) list of image paths
        """
        print("   3.1 Image Embedding Dataset...")
        dataset = ImageEmbeddingDataset(image_paths)
        print("   3.2 DataLoader...")
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

        all_embeddings = []
        all_paths = []

        print("   3.3 tqdm iterator...\n")
        iterator = tqdm.tqdm(dataloader, desc="Generating embeddings") if show_progress else dataloader

        print("   3.4 for loop...")
        for batch_images, batch_paths in iterator:
            batch_images = batch_images.to(self.device, non_blocking=True)

            # Get embeddings: average pool patch tokens for global representation
            features = self._get_features(batch_images)   # (B, N, D)
            embeddings = features.mean(dim=1)     # (B, D)

            # L2 normalization for cosine similarity
            embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().numpy())
            all_paths.extend(batch_paths)

        embeddings = np.vstack(all_embeddings)

        if return_paths:
            return embeddings, all_paths
        return embeddings

    def generate_single_embedding(
        self, image: Union[str, Path, Image.Image, torch.Tensor]
    ) -> np.ndarray:
        """Generate embedding for a single image"""
        transform = ImageEmbeddingDataset.default_transform()

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')

        if isinstance(image, Image.Image):
            image = transform(image)

        if isinstance(image, torch.Tensor):
            image = image.unsqueeze(0) if image.dim() == 3 else image

        image = image.to(self.device)

        with torch.no_grad():
            features = self._get_features(image)
            embedding = features.mean(dim=1)
            embedding = nn.functional.normalize(embedding, p=2, dim=1)

        return embedding.cpu().numpy()
