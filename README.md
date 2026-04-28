# Meme Similarity Search

[![Demo Space](https://img.shields.io/badge/🤗-Open%20in%20Spaces-blue)](https://huggingface.co/spaces/Unspoiled-Egg/Meme-Recommender)

A high-performance image retrieval system for memes using **I-JEPA (Image Joint-Embedding Predictive Architecture)** and **Pinecone**. This project enables semantic similarity search, allowing you to find visually and contextually similar memes by uploading an image.

## How it Works

The project leverages self-supervised learning and vector database technology:

1.  **I-JEPA Feature Extraction**: I used I-JEPA huge vision transformer to generate rich, high-dimensional embeddings (5120-dim) for each image. Unlike standard contrastive models, I-JEPA learns representations by predicting missing parts of an image in latent space, capturing deep semantic features.
2.  **Embedding Strategy**: I concatenated the outputs of the last four layers of the transformer to capture both fine-grained details and high-level concepts.
3.  **Vector Storage (Pinecone)**: Extracted embeddings are indexed in a Pinecone serverless index for ultra-fast cosine similarity search.
4.  **Retrieval**: When a query image is uploaded, it is processed through the same pipeline, and its embedding is used to query Pinecone for the top-5 nearest neighbors.

## Setup & Installation

This project uses [uv](https://github.com/astral-sh/uv) for Python package management.

### 1. Install `uv`
If you haven't installed `uv` yet:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Initialize Environment
Sync the dependencies:
```bash
# Create virtual environment and install dependencies
uv sync
```

### 3. Configuration
Create a `.env` file in the root directory with your Pinecone credentials:
```env
PINECONE_API_KEY=your_api_key
PINECONE_CLOUD=pinecone_cloud(aws/gcp)
PINECONE_REGION=region(us-west-2/us-east-1)
PINECONE_INDEX_NAME=your_pinecone_index_name
```

## Usage

### 1. Indexing Images
Before searching, you need to process your image collection and store the embeddings in Pinecone. Place your images in `data/images/`.

```bash
uv run scripts/index_images.py
```

### 2. Running the Gradio UI
Launch the interactive web interface to search for similar memes:

```bash
uv run app.py
```
By default, the app will be available at `http://localhost:7860`.

## Project Structure

- `src/models/`: Implementation of I-JEPA Vision Transformer.
- `src/processing/`: Logic for embedding generation and model management.
- `src/search/`: Pinecone and FAISS indexer implementations.
- `scripts/`: Utility scripts for indexing and metadata updates.
- `app.py`: Gradio interface for the end-user.
- `config/`: Pydantic-based settings management.

## Tech Stack
- **Model**: I-JEPA (Vision Transformer)
- **Vector DB**: Pinecone
- **Frameworks**: PyTorch, Gradio
- **Package Manager**: uv
