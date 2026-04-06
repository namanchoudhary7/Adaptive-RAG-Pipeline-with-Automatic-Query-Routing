"""
Automated Setup Script for the Adaptive RAG Project.

Run this immediately after cloning the repository:
    uv run python -m scripts.setup
"""

import os
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# --- Force online mode before any AI libraries are imported ---
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings

def download_router():
    """Download the Zero-Shot Router Model (DeBERTa)."""
    print("📦 Downloading Router Model (DeBERTa)...")
    try:
        pipeline(
            task='zero-shot-classification',
            model='cross-encoder/nli-deberta-v3-small'
        )
        print("✅ Router model cached successfully")
    except Exception as e:
        print(f"❌ Failed to download router model: {e}")

def download_embeddings():
    """Download the Embedding Model (MiniLM)."""
    print("📦 Downloading Embedding Model (MiniLM)...")
    try:
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✅ Embedding model cached successfully")
    except Exception as e:
        print(f"❌ Failed to download embedding model: {e}")

def main():
    print("\n" + "="*50)
    print("🚀 Initializing Adaptive RAG Environment...")
    print("="*50 + "\n")

    # 1. Setup .env file
    env_path = Path(".env")
    example_env_path = Path(".env.example")
    
    if not env_path.exists():
        if example_env_path.exists():
            shutil.copy(example_env_path, env_path)
            print("✅ Created .env file from .env.example")
            print("⚠️  ACTION REQUIRED: Open .env and add your GROQ_API_KEY!")
        else:
            with open(env_path, "w") as f:
                f.write("HF_HUB_OFFLINE=0\nGROQ_API_KEY=your_api_key_here\n")
            print("✅ Created a fresh .env file")
            print("⚠️  ACTION REQUIRED: Open .env and add your GROQ_API_KEY!")
    else:
        print("✅ .env file already exists")

    # 2. & 3. Download Models in Parallel
    # We use a ThreadPool to run both independent network requests at once
    print("\n⏳ Pre-fetching models in parallel...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        executor.submit(download_router)
        executor.submit(download_embeddings)

    # 4. Run Ingestion (Build ChromaDB)
    # This remains sequential as it depends on the models being ready
    print("\n📚 Building Vector Database...")
    try:
        from scripts import ingest
        ingest.main()
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        print("Make sure you have added your GROQ_API_KEY to the .env file if ingestion requires it.")

    print("\n" + "="*50)
    print("🎉 Setup Complete!")
    print("="*50)
    print("Next Steps:")
    print("1. Open .env and change HF_HUB_OFFLINE=0 to HF_HUB_OFFLINE=1 for maximum speed.")
    print("2. Start the API: uv run uvicorn backend.main:app --reload")
    print("3. Start the UI:  uv run streamlit run frontend/app.py\n")

if __name__ == "__main__":
    main()