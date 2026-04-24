import uuid
import numpy as np
from pathlib import Path
from typing import List, Union, Tuple, Optional


class PineconeImageSimilaritySearch:
    """
    Pinecone-backend similarity search for image embeddings. Image paths are stored as
    Pinecone vector metadata so no local state is needed between sessions.
    """

    def __init__(
        self,
        index_name: str,
        api_key: str,
        cloud: str,
        region: str,
        dimension: int = 768,
        namespace: str = "",
        metric: str = "cosine",
        create_if_missing: bool = True,
    ):
        try:
            from pinecone import Pinecone, ServerlessSpec
        except ImportError:
            raise ImportError("Run `pip install pinecone` first.")

        self.dimension = dimension
        self.index_type = metric
        self.namespace = namespace
        self._index_name = index_name
        self.image_paths: List[str] = []
        self.metadata: dict = {}

        pc = Pinecone(api_key=api_key)

        existing = [idx.name for idx in pc.list_indexes()]
        if index_name not in existing:
            if not create_if_missing:
                raise ValueError(
                    f"Index '{index_name}' not found and create_if_missing=False."
                )
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            print(f"Created Pinecone index '{index_name}' ({metric}, dim={dimension})")
        else:
            print(f"Connected to existing Pinecone index '{index_name}'")

        self.index = pc.Index(index_name)

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        image_paths: List[str],
        metadata: Optional[dict] = None,
    ):
        """
        Upsert embeddings into Pinecone.

        Args:
            embeddings: (N, D) float32 array for L2-normalised embeddings
            image_paths: List of N image paths (stored as Pinecone metadata)
            metadata: Optional {int_index: dict} of extra per-image metadata
        """
        assert len(embeddings) == len(image_paths), "Embeddings and paths must match"
        assert embeddings.shape[1] == self.dimension, (
            f"Expected dim {self.dimension}, got {embeddings.shape[1]}"
        )

        embeddings = embeddings.astype("float32")

        vectors = []
        for i, (emb, path) in enumerate(zip(embeddings, image_paths)):
            vec_id = str(uuid.uuid4())
            meta = {"image_path": path}
            if metadata and i in metadata:
                meta.update(metadata[i])
            vectors.append({"id": vec_id, "values": emb.tolist(), "metadata": meta})

        # Pinecone recommends batches of <= 100
        batch_size = 100
        for start in range(0, len(vectors), batch_size):
            self.index.upsert(
                vectors=vectors[start: start + batch_size],
                namespace=self.namespace,
            )

        self.image_paths.extend(image_paths)
        if metadata:
            for i, path in enumerate(image_paths):
                if i in metadata:
                    self.metadata[path] = metadata[i]

        print(f"Upserted {len(embeddings)} vectors. "
              f"Total (local cache): {len(self.image_paths)}")

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        return_scores: bool = True,
        filter: Optional[dict] = None,
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """
        Search for the k most similar images.

        Args:
            query_embedding: (1, D) or (D,) float32 array
            k: Number of results
            return_scores: If True, return (path, score) tuples
            filter: Optional Pinecone metadata filter dict
        """
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_list = query_embedding[0].astype("float32").tolist()

        kwargs = dict(
            vector=query_list,
            top_k=k,
            include_metadata=True,
            namespace=self.namespace,
        )
        if filter:
            kwargs["filter"] = filter

        response = self.index.query(**kwargs)

        results = []
        for match in response.matches:
            path = match.metadata.get("image_path", match.id)
            if return_scores:
                results.append((path, float(match.score)))
            else:
                results.append(path)

        return results

    def batch_search(
        self,
        query_embeddings: np.ndarray,
        k: int = 5,
        filter: Optional[dict] = None,
    ) -> List[List[Tuple[str, float]]]:
        """Search for multiple query embeddings sequentially."""
        return [
            self.search(q, k=k, return_scores=True, filter=filter)
            for q in query_embeddings
        ]

    def save(self, save_dir: Union[str, Path]):
        """
        Pinecone vectors are already persisted server-side.
        This optionally saves the local image_path cache to disk so we don't
        have to re-scane the index on startup.
        """
        import pickle

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        local_state = {
            "image_paths": self.image_paths,
            "metadata": self.metadata,
            "dimension": self.dimension,
            "namespace": self.namespace,
            "index_name": self._index_name,
            "metric": self.index_type,
        }
        with open(save_dir / "pinecone_local_cache.pkl", "wb") as f:
            pickle.dump(local_state, f)
        print(f"Saved local state to: {save_dir}")

    @classmethod
    def load(
        cls, save_dir: Union[str, Path], api_key: str, use_gpu: bool = False
    ) -> "PineconeImageSimilaritySearch":
        """
        Load local state and return a new PineconeImageSimilaritySearch instance.
        """
        import pickle

        save_dir = Path(save_dir)
        with open(save_dir / "pinecone_local_cache.pkl", "rb") as f:
            local_state = pickle.load(f)

        instance = cls(
            index_name=local_state["index_name"],
            api_key=api_key,
            dimension=local_state["dimension"],
            namespace=local_state["namespace"],
            metric=local_state["index_type"],
            create_if_mission=False,
        )
        instance.image_paths = local_state["image_paths"]
        instance.metadata = local_state["metadata"]
        return instance
