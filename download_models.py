"""
Run this script ONCE on the server before starting the service in offline mode.
It downloads both models into the local Hugging Face cache so that
HF_HUB_OFFLINE=1 / TRANSFORMERS_OFFLINE=1 work without errors.

Usage:
    python download_models.py

Optional — override cache location to match your HF_HOME in .env:
    HF_HOME=/path/to/model-cache python download_models.py
"""

import os
import shutil

# Force online mode for this script regardless of what .env says
os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("TRANSFORMERS_OFFLINE", None)

from sentence_transformers import SentenceTransformer, CrossEncoder

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
RERANK_MODEL    = os.getenv("RERANK_MODEL",    "BAAI/bge-reranker-v2-m3")

# ------------------------------------------------------------------
# Purge any broken/partial cache entries for the models we're about
# to download so that a previous failed attempt doesn't block this one.
# ------------------------------------------------------------------
hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
hub_cache = os.path.join(hf_home, "hub")

for model_id in (EMBEDDING_MODEL, RERANK_MODEL):
    # HF stores models as  models--<org>--<name>
    folder_name = "models--" + model_id.replace("/", "--")
    model_cache_path = os.path.join(hub_cache, folder_name)
    if os.path.exists(model_cache_path):
        print(f"Removing existing cache for {model_id} …")
        shutil.rmtree(model_cache_path)

print(f"Downloading embedding model: {EMBEDDING_MODEL}")
embedder = SentenceTransformer(EMBEDDING_MODEL)
print("  Done.\n")

print(f"Downloading reranker model:  {RERANK_MODEL}")
reranker = CrossEncoder(RERANK_MODEL)
print("  Done.\n")

# Quick smoke-test so you know the models actually work before going offline
test_vecs = embedder.encode(["smoke test"], normalize_embeddings=True)
test_score = reranker.predict([["smoke test", "smoke test"]])
print(f"Smoke test passed — embedding dim: {len(test_vecs[0])}, reranker score: {test_score[0]:.4f}")

hf_home = os.environ.get("HF_HOME") or os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
print(f"\nModels cached in: {hf_home}")
print("You can now start the service with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1.")
