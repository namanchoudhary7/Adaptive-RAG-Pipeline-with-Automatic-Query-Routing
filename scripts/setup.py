"""
Automated Setup Script for the Adaptive RAG Project.

Run this immediately after cloning the repository:
    uv run python -m scripts.setup
"""

import os
import shutil
from pathlib import Path
from transformers import pipeline
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["HF_HUB_OFFLINE"] = "0"

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
            # Fallback if they didn't make an .env.example
            with open(env_path, "w") as f:
                f.write("HF_HUB_OFFLINE=0\nGROQ_API_KEY=your_api_key_here\n")
            print("✅ Created a fresh .env file")
            print("⚠️  ACTION REQUIRED: Open .env and add your GROQ_API_KEY!")
    else:
        print("✅ .env file already exists")

    # 2. Download Zero-Shot Router Model
    print("\n📦 Downloading Router Model (DeBERTa)...")
    try:
        pipeline(
            task='zero-shot-classification',
            model='cross-encoder/nli-deberta-v3-small'
        )
        print("✅ Router model cached successfully")
    except Exception as e:
        print(f"❌ Failed to download router model: {e}")

    # 3. Download Embedding Model
    print("\n📦 Downloading Embedding Model (MiniLM)...")
    try:
        HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        print("✅ Embedding model cached successfully")
    except Exception as e:
        print(f"❌ Failed to download embedding model: {e}")

    # 4. Run Ingestion (Build ChromaDB)
    print("\n📚 Building Vector Database...")
    try:
        # Import dynamically so it uses the models we just downloaded
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