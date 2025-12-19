# BGE Inference Service

Микросервис для векторного поиска и реранкинга, предоставляющий API для моделей семейства BGE (FlagEmbedding).

## Архитектура и Best Practices

Данный сервис реализует паттерн **Model-as-a-Service**. Вынесение тяжелого инференса (GPU/PyTorch) в отдельный сервис обеспечивает:
1. **Изоляцию ресурсов:** GPU-вычисления не блокируют I/O операции основного API.
2. **Оптимизацию Docker-образов:** Основной бэкенд не тянет за собой `torch` и CUDA-зависимости (размер образа снижается с 5GB+ до <500MB).
3. **Независимое масштабирование:** Можно масштабировать GPU-воркеры отдельно от бизнес-логики.

Сервис спроектирован как drop-in замена (или дополнение) для решений типа Text Embeddings Inference (TEI), добавляя поддержку специфичных для проекта алгоритмов (кастомный sparse-mapping).

## API Reference

- **Swagger UI:** `http://localhost:8011/docs`
- **OpenAPI Schema:** `http://localhost:8011/openapi.json`

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/v1/embeddings` | Dense векторы (OpenAI compatible format) |
| POST | `/v1/sparse-embeddings` | Sparse векторы (Indices + Values) |
| POST | `/v1/hybrid-embeddings` | Одновременно Dense + Sparse |
| POST | `/v1/rerank` | Реранкинг пар запрос-документ (Cohere compatible) |
| GET | `/health` | Статус загрузки моделей |

---

## Integration Guide (Примеры использования)

Ниже приведены примеры реализации RAG-пайплайна с использованием этого сервиса.

### Конфигурация клиента

```python
import os
import requests
from openai import OpenAI
from qdrant_client import QdrantClient

# Настройки
BGE_SERVICE_URL = os.getenv("BGE_SERVICE_URL", "http://localhost:8011")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Клиенты
bge_session = requests.Session()
openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(url="http://localhost:6333")

def get_bge_sparse(text: str) -> dict:
    """Helper для получения sparse вектора."""
    resp = bge_session.post(
        f"{BGE_SERVICE_URL}/v1/sparse-embeddings",
        json={"input": [text]}
    )
    resp.raise_for_status()
    # Возвращает структуру: {"indices": [...], "values": [...]}
    return resp.json()["data"][0]["sparse"]

def get_bge_hybrid(text: str) -> dict:
    """Helper для получения hybrid (dense+sparse)."""
    resp = bge_session.post(
        f"{BGE_SERVICE_URL}/v1/hybrid-embeddings",
        json={"input": [text]}
    )
    resp.raise_for_status()
    return resp.json()["data"][0]
```

### Сценарий A: Ingest (OpenAI Dense + BGE Sparse)

Используется, если вы хотите оставить качество Dense-векторов от OpenAI, но добавить keyword-search через BGE-M3.

```python
def ingest_document_openai_bge(text: str, doc_id: str):
    # 1. Получаем Dense от OpenAI
    dense_resp = openai_client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    dense_vector = dense_resp.data[0].embedding

    # 2. Получаем Sparse от BGE Service
    sparse_vector = get_bge_sparse(text)

    # 3. Сохраняем в Qdrant
    qdrant_client.upsert(
        collection_name="my_collection",
        points=[{
            "id": doc_id,
            "vector": {
                "dense": dense_vector,
                "sparse": sparse_vector
            },
            "payload": {"text": text}
        }]
    )
```

### Сценарий B: Ingest (Full Hybrid via BGE)

Полностью локальный вариант (бесплатно, без OpenAI).

```python
def ingest_document_full_bge(text: str, doc_id: str):
    # 1. Получаем оба вектора за один запрос
    hybrid_result = get_bge_hybrid(text)
    
    dense_vector = hybrid_result["dense"]
    sparse_vector = hybrid_result["sparse"]

    # 2. Сохраняем в Qdrant
    qdrant_client.upsert(
        collection_name="my_collection",
        points=[{
            "id": doc_id,
            "vector": {
                "dense": dense_vector,
                "sparse": sparse_vector
            },
            "payload": {"text": text}
        }]
    )
```

### Сценарий C: Retrieval (Hybrid Search)

Получение вектора запроса и поиск в БД.

```python
def search_documents(query: str, top_k: int = 10):
    # 1. Генерируем векторы для запроса
    # Вариант с OpenAI + BGE:
    # dense_query = openai_client.embeddings.create(input=query, ...).data[0].embedding
    # sparse_query = get_bge_sparse(query)
    
    # Вариант Pure BGE:
    hybrid_query = get_bge_hybrid(query)
    dense_query = hybrid_query["dense"]
    sparse_query = hybrid_query["sparse"]

    # 2. Выполняем поиск в Qdrant (Hybrid Query)
    # Используем prefetch для гибридного поиска (RRF или Score Fusion)
    search_result = qdrant_client.query_points(
        collection_name="my_collection",
        prefetch=[
            {
                "query": dense_query,
                "using": "dense",
                "limit": top_k
            },
            {
                "query": sparse_query,
                "using": "sparse",
                "limit": top_k
            }
        ],
        query=None, # RRF fusion strategy would go here in newer Qdrant APIs
        limit=top_k
    )
    
    # Упрощенно возвращаем payload
    return [hit.payload for hit in search_result.points]
```

### Сценарий D: Rerank & LLM Context

Переранжирование результатов поиска перед отправкой в LLM.

```python
def generate_answer(query: str, initial_docs: list[dict]):
    # initial_docs - список словарей с полем 'text' из шага Retrieval
    
    # 1. Подготовка документов для реранкера
    # API ожидает список строк или объектов
    docs_text = [doc["text"] for doc in initial_docs]

    # 2. Запрос к сервису реранкинга
    rerank_resp = bge_session.post(
        f"{BGE_SERVICE_URL}/v1/rerank",
        json={
            "query": query,
            "documents": docs_text,
            "top_n": 5,           # Берем только топ-5 самых релевантных
            "return_documents": False 
        }
    )
    rerank_resp.raise_for_status()
    results = rerank_resp.json()["results"]
    
    # 3. Формируем контекст из топ-5
    top_docs = []
    for res in results:
        # res['index'] указывает на индекс в исходном списке docs_text
        original_doc = initial_docs[res["index"]]
        top_docs.append(original_doc["text"])

    context_str = "\n\n".join(top_docs)

    # 4. Отправка в LLM
    completion = openai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Answer based on the context."},
            {"role": "user", "content": f"Context:\n{context_str}\n\nQuestion: {query}"}
        ]
    )
    
    return completion.choices[0].message.content
```

---

## Запуск и Установка

### Локально
```bash
pip install -r requirements.txt
python -m uvicorn bge_inference_service.main:app --host 0.0.0.0 --port 8011
```

### Docker
```bash
docker build -t bge-service .
docker run --gpus all -p 8011:8011 bge-service
```
