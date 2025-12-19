import os
import time
import requests
import json
import pytest
from typing import Dict, Any

# Configuration
BASE_URL = os.getenv("BGE_SERVICE_URL", "http://localhost:8011")
TEST_TEXT_1 = "Как зарегистрировать выпуск облигаций?"
TEST_TEXT_2 = "Для регистрации выпуска облигаций необходимо подготовить проспект эмиссии и решение о выпуске."
TEST_TEXT_3 = "Инструкция по работе с эмитентами содержит общие рекомендации."


def log(msg: str):
    print(f"\n[TEST] {msg}")


def wait_for_service(url: str, timeout: int = 300):
    start = time.time()
    log(f"Waiting for service at {url}/health ...")
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                data = r.json()
                if data.get("status") == "ok":
                    log(f"Service is UP! Info: {data}")
                    return True
                else:
                    log(f"Service running but degraded: {data}")
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    raise RuntimeError("Service failed to start within timeout")


def test_health():
    """Verify service health endpoint."""
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["embedder_loaded"] is True
    assert data["reranker_loaded"] is True


def test_dense_embeddings_openai_format():
    """UseCase A: Dense embeddings via OpenAI-compatible endpoint."""
    payload = {
        "model": "bge-m3",
        "input": [TEST_TEXT_1, TEST_TEXT_2]
    }
    r = requests.post(f"{BASE_URL}/v1/embeddings", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    
    # Check structure
    assert data["object"] == "list"
    assert len(data["data"]) == 2
    
    # Check embedding properties
    emb1 = data["data"][0]["embedding"]
    emb2 = data["data"][1]["embedding"]
    assert isinstance(emb1, list)
    assert len(emb1) > 0
    assert len(emb1) == len(emb2)
    # BGE-M3 dense vectors should be 1024-dim by default
    assert len(emb1) == int(os.getenv("EMBEDDING_SIZE", "1024"))
    log(f"Dense vector dim: {len(emb1)}")


def test_sparse_embeddings():
    """UseCase B: Sparse embeddings (indices + values)."""
    payload = {
        "model": "bge-m3",
        "input": [TEST_TEXT_1]
    }
    r = requests.post(f"{BASE_URL}/v1/sparse-embeddings", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    
    item = data["data"][0]
    sparse = item["sparse"]
    
    assert "indices" in sparse
    assert "values" in sparse
    assert len(sparse["indices"]) == len(sparse["values"])
    assert len(sparse["indices"]) > 0
    
    # Check mapping metadata matches config default (hash)
    assert sparse["mapping"] in ["hash", "tokenizer"]
    log(f"Sparse vector NNZ: {len(sparse['indices'])}, mapping: {sparse['mapping']}")


def test_hybrid_embeddings():
    """UseCase C: Hybrid embeddings (Dense + Sparse in one call)."""
    payload = {
        "model": "bge-m3",
        "input": [TEST_TEXT_1]
    }
    r = requests.post(f"{BASE_URL}/v1/hybrid-embeddings", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    
    item = data["data"][0]
    assert "dense" in item
    assert "sparse" in item
    
    assert len(item["dense"]) > 0
    assert len(item["sparse"]["indices"]) > 0
    log("Hybrid embeddings received successfully")


def test_rerank():
    """UseCase D: Rerank documents for a query."""
    payload = {
        "model": "bge-reranker-v2-m3",
        "query": TEST_TEXT_1,
        "documents": [
            TEST_TEXT_2, # Relevant
            TEST_TEXT_3, # Less relevant
            "Рецепт приготовления пиццы." # Irrelevant
        ],
        "top_n": 3,
        "return_documents": True
    }
    r = requests.post(f"{BASE_URL}/v1/rerank", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    
    results = data["results"]
    assert len(results) == 3
    
    # Check sorting (first should be most relevant)
    assert results[0]["index"] == 0 # TEST_TEXT_2 should be first
    assert results[0]["relevance_score"] > results[2]["relevance_score"]
    
    assert "document" in results[0]
    assert results[0]["document"]["text"] == TEST_TEXT_2
    log(f"Rerank top score: {results[0]['relevance_score']}")


if __name__ == "__main__":
    # Wait for service availability before running tests
    try:
        wait_for_service(BASE_URL)
        
        # Run tests manually if not using pytest runner
        test_health()
        log("Health check passed")
        
        test_dense_embeddings_openai_format()
        log("Dense embeddings passed")
        
        test_sparse_embeddings()
        log("Sparse embeddings passed")
        
        test_hybrid_embeddings()
        log("Hybrid embeddings passed")
        
        test_rerank()
        log("Rerank passed")
        
        print("\nAll integration tests passed successfully!")
    except Exception as e:
        print(f"\nTests failed: {e}")
        exit(1)
