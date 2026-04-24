from pathlib import Path
from src.processing.model import IJEPAManager
from src.processing.embedder import EmbeddingGenerator
from src.search.pinecone_indexer import PineconeImageSimilaritySearch
from config import settings


def main(test_img: str):
    manager = IJEPAManager(model_key=settings.model_type)
    model = manager.load_target_encoder()

    generator = EmbeddingGenerator(
        model=model,
        device=settings.device,
        batch_size=settings.batch_size,
        num_workers=2,
        layer_strategy="last_four_concat"
    )

    searcher = PineconeImageSimilaritySearch(
        index_name=settings.pinecone_index_name,
        api_key=settings.pinecone_api_key,
        dimension=settings.embedding_dim,
        metric="cosine",
    )

    query_image_path = Path(test_img)
    query_embedding = generator.generate_single_embedding(query_image_path)

    results = searcher.search(query_embedding, k=5, return_scores=True)

    print(f"\nTop 5 similar images to {query_image_path}:")
    for rank, (path, score) in enumerate(results, 1):
        print(f" {rank}. {path} (similarty: {score:.4f})")


if __name__ == "__main__":
    main(test_img="data/test_images/query_image_6.png")
