import faiss
import pickle
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union, Optional


class ImageSimilaritySearch:
    """FAISS-based similarity search for image embeddings"""

    def __init__(
        self,
        dimension: int = 768,  # Dependes on model size (768 for ViT-B)
        index_type: str = "cosine",  # "cosine", "l2", "ip" (inner product)
        use_gpu: bool = False,
    ):
        self.dimension = dimension
        self.index_type = index_type
        self.use_gpu = use_gpu
        self.index = None
        self.image_paths = []
        self.metadata = {}

        self._create_index()

    def _create_index(self):
        """Create FAISS index based on similarity metric"""
        if self.index_type == "cosine":
            # For cosine similarity, we use IndexFlatIP with normalized vectors
            # vectors are already L2 normalized in embedding generator
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "l2":
            self.index = faiss.IndexFlatL2(self.dimension)
        elif self.index_type == "ip":
            self.index = faiss.IndexFlatIP(self.dimension)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        # Move to GPU if requested
        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        print(f"Created {self.index_type} index (dim={self.dimension})")

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        image_paths: List[str],
        metadata: Optional[dict] = None,
    ):
        """
        Add embeddings to the index.

        Args:
            embeddings: (N, D) array of L2-normalized embeddings
            image_paths: List of N image paths
            metadata: Optional dict with additional info for each image
        """
        assert len(embeddings) == len(image_paths), "Embeddings and paths must match"
        assert embeddings.shape[1] == self.dimension, f"Expected dim {self.dimension}, got {embeddings.shape[1]}"

        # Ensure float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings.astype('float32'))

        # Add to index
        start_idx = len(self.image_paths)
        self.index.add(embeddings)
        self.image_paths.extend(image_paths)

        # Store metadata
        if metadata:
            for i, path in enumerate(image_paths):
                self.metadata[path] = metadata.get(start_idx + i, {})

        print(f"Added {len(embeddings)} embeddings. Total: {len(self.image_paths)}")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        return_scores: bool = True,
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """
        Search for k most similar images.

        Args:
            query_embeddings: (1, D) or (D,) array - single query embedding
            k: Number of results to return
            return_scores: If True, return (path, score) tuples

        Returns:
            List of image paths or (path, similarity_score) tuple
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        # Ensure float32 and contiguous
        query_embedding = np.ascontiguousarray(query_embedding.astype('float32'))

        # Search
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:   # FAISS returns -1 for not enough results
                continue
            path = self.image_paths[idx]
            if return_scores:
                results.append((path, float(score)))
            else:
                results.append(path)

        return results

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        k: int = 5,
    ) -> List[List[Tuple[str, float]]]:
        """Search for multiple queries at once."""
        query_embeddings = np.ascontiguousarray(
            query_embeddings.astype('float32')
        )
        scores, indices = self.index.search(query_embeddings, k)

        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                if idx == -1:
                    continue
                results.append((self.image_paths[idx], float(score)))
            all_results.append(results)

        return all_results

    def save(self, save_dir: Union[str, Path]):
        """Save index and metadata."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        index_path = save_dir / "index.faiss"
        # Move to CPU before saving if on GPU
        if self.use_gpu:
            cpu_index = faiss.index_gpu_to_cpu(self.index)
            faiss.write_index(cpu_index, str(index_path))
        else:
            faiss.write_index(self.index, str(index_path))

        # Save metadata
        metadata = {
            'image_paths': self.image_paths,
            'metadata': self.metadata,
            'dimension': self.dimension,
            'index_type': self.index_type,
        }
        with open(save_dir / "metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Saved index to {save_dir}")

    @classmethod
    def load(cls, save_dir: Union[str, Path], use_gpu: bool = False):
        """Load index and metadata."""
        save_dir = Path(save_dir)

        # Load metadata
        with open(save_dir / "metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)

        # Create instance
        instance = cls(
            dimension=metadata['dimension'],
            index_type=metadata['index_type'],
            use_gpu=use_gpu,
        )

        # Load FAISS index
        index_path = save_dir / "index.faiss"
        instance.index = faiss.read_index(str(index_path))

        if use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            instance.index = faiss.index_cpu_to_gpus(res, 0, instance.index)

        instance.image_paths = metadata['image_paths']
        instance.metadata = metadata['metadata']

        print(f"Loaded index with {len(instance.image_paths)} embeddings")
        return instance
