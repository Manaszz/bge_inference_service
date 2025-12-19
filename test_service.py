import os
import requests

BASE_URL = os.getenv("BGE_SERVICE_URL", "http://localhost:8011")


def main() -> None:
    print("Health:")
    print(requests.get(f"{BASE_URL}/health", timeout=10).json())

    text = "Пример запроса для теста"

    print("\nDense (OpenAI compatible):")
    r = requests.post(
        f"{BASE_URL}/v1/embeddings",
        json={"input": [text], "model": "bge-m3"},
        timeout=60,
    )
    r.raise_for_status()
    print({"dims": len(r.json()["data"][0]["embedding"])})

    print("\nSparse:")
    r = requests.post(
        f"{BASE_URL}/v1/sparse-embeddings",
        json={"input": [text]},
        timeout=60,
    )
    r.raise_for_status()
    s = r.json()["data"][0]["sparse"]
    print({"nnz": len(s["indices"]), "mapping": s["mapping"], "index_space": s["index_space"]})

    print("\nHybrid:")
    r = requests.post(
        f"{BASE_URL}/v1/hybrid-embeddings",
        json={"input": [text]},
        timeout=60,
    )
    r.raise_for_status()
    h = r.json()["data"][0]
    print({"dense_dims": len(h["dense"]), "sparse_nnz": len(h["sparse"]["indices"])})

    print("\nRerank:")
    r = requests.post(
        f"{BASE_URL}/v1/rerank",
        json={
            "query": "как зарегистрировать выпуск облигаций",
            "documents": [
                "Процедура регистрации выпуска облигаций включает подготовку проспекта...",
                "Инструкция по работе с эмитентами содержит общие рекомендации...",
            ],
            "top_n": 2,
        },
        timeout=60,
    )
    r.raise_for_status()
    print(r.json())


if __name__ == "__main__":
    main()
