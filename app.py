import gradio as gr
from pathlib import Path
from PIL import Image
import numpy as np

from src.processing.model import IJEPAManager
from src.processing.embedder import EmbeddingGenerator
from src.search.pinecone_indexer import PineconeImageSimilaritySearch
from config import settings


print("Initializing model and search index...")
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


def search_similar_images(input_img):
    if input_img is None:
        return None

    # Convert Gradio input (can be numpy array) to PIL Image
    if isinstance(input_img, np.ndarray):
        input_img = Image.fromarray(input_img)

    query_embedding = generator.generate_single_embedding(input_img)
    results = searcher.search(query_embedding, k=5, return_scores=True)

    gallery_items = []
    for path, score in results:
        img_path = Path(path)
        if not img_path.is_absolute():
            img_path = settings.project_root / img_path

        if img_path.exists():
            gallery_items.append((str(img_path), f"Similarity: {score:.4f}"))
        else:
            print(f"Warning: Image path not found: {img_path}")

    return gallery_items


custom_css = """
.container {
    max-width: 1000px;
    margin: auto;
    padding: 20px;
}
.header {
    text-align: center;
    margin-bottom: 30px;
}
.header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #4F46E5, #EC4899);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}
.header p {
    color: #6B7280;
    font-size: 1.1rem;
}
.gradio-container {
    background-color: #F9FAFB !important;
}
.gallery-container {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}
"""

with gr.Blocks() as demo:
    with gr.Column(elem_classes="container"):
        with gr.Column(elem_classes="header"):
            gr.Markdown("# Meme Similarity Search")
            gr.Markdown(
                "Upload an image to find the top 5 most similar memes in our database.")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Image",
                    type="pil",
                    elem_id="input-img"
                )
                search_btn = gr.Button("Find Similar Memes", variant="primary")

            with gr.Column(scale=2):
                output_gallery = gr.Gallery(
                    label="Top 5 Similar Memes",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=3,
                    object_fit="contain",
                    height="600px"
                )

        search_btn.click(
            fn=search_similar_images,
            inputs=input_image,
            outputs=output_gallery
        )

        input_image.upload(
            fn=search_similar_images,
            inputs=input_image,
            outputs=output_gallery
        )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=custom_css
    )
