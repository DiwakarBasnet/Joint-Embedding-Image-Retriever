from pathlib import Path
from config import settings
from src.processing.model import IJEPAManager
from src.processing.embedder import EmbeddingGenerator
from src.search.pinecone_indexer import PineconeImageSimilaritySearch


def main():
    print("--- 1. Initializing I-JEPA Manager ---")
    manager = IJEPAManager(model_key=settings.model_type)

    if Path(settings.model_path).exists():
        print(f"Loading packaged model from {settings.model_path}...")
        model = manager.load_target_encoder()
    else:
        print(f"Loading raw checkpoint from {settings.checkpoint_path}...")
        model = manager.load_model()
        manager.save_target_encoder()

    print("\n--- 2. Creating Embedding Generator ---")
    generator = EmbeddingGenerator(
        model=model,
        device=settings.device,
        batch_size=settings.batch_size,
        num_workers=2,
        layer_strategy="last_four_concat"
    )

    print("\n--- 3. Generating Embeddings ---")
    IMAGE_DIR = settings.data_dir / "images"
    image_paths = list(IMAGE_DIR.glob("**/*.jpg")) \
        + list(IMAGE_DIR.glob("**/*.png")) \
        + list(IMAGE_DIR.glob("**/*.jpeg"))

    if not image_paths:
        print(f"No images found in {IMAGE_DIR}. Check your settings.")
        return

    print(f"Found {len(image_paths)} images. Processing...")
    embeddings, paths = generator.generate_embeddings(image_paths)

    print("\n--- 4. Syncing with Pinecone ---")
    searcher = PineconeImageSimilaritySearch(
        index_name=settings.pinecone_index_name,
        api_key=settings.pinecone_api_key,
        dimension=settings.embedding_dim,
        metric="cosine",
    )
    searcher.add_embeddings(embeddings, [str(p) for p in paths])

    print("\nIndexing Complete.")
    print(f"Index: {settings.pinecone_index_name}")
    print(f"Dimension: {settings.embedding_dim}")


if __name__ == "__main__":
    main()
