# Changelog
Все заметные изменения этого репозитория будут документироваться в этом файле.

Формат основан на принципах [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
и проект придерживается семантического версионирования (SemVer) по мере появления версий.

## [Unreleased]

## [0.3.0] - 2025-12-22
### Changed
- Updated dependencies to latest available versions for closed-contour compatibility:
  - FastAPI: 0.115.0 → 0.122.0
  - Uvicorn: 0.32.0 → 0.38.0
  - Pydantic: 2.9.0 → 2.12.5
  - Pydantic-settings: 2.5.2 → 2.12.0
  - Starlette: 0.38.6 → 0.50.0
  - Pydantic-core: 2.23.2 → 2.41.5
- Added `.cursorrules` with "Always Latest" library policy and mandatory `context7` usage.
- Added Memory Bank documentation structure for project knowledge retention.

### Fixed
- Fixed `BGEEngine.rerank` bug: replaced missing `_prepare_texts` call with `prepare_texts` method.

### Added
- Added `pip-freeze.txt` as the source of truth for installed dependencies.
- Added `.dockerignore` to optimize Docker build context.

## [0.2.0] - 2025-12-19
### Added
- Очередь + микробатчинг для эмбеддингов (dense/sparse/hybrid) для эффективной обработки параллельных запросов и лучшей утилизации GPU.
- Новые переменные окружения для настройки микробатчинга:
  - `EMBEDDING_MICROBATCH_ENABLED`
  - `EMBEDDING_MICROBATCH_MAX_WAIT_MS`
  - `EMBEDDING_MICROBATCH_MAX_BATCH_TEXTS`
  - `EMBEDDING_MICROBATCH_QUEUE_MAXSIZE`
- Документация по multi-GPU рекомендации (1 реплика сервиса = 1 GPU).
- Документация с примером сопоставления входных текстов и эмбеддингов по `index`/порядку (полезно для загрузки в Qdrant с метаданными).

### Changed
- Endpoints эмбеддингов переведены на `async def` для корректного ожидания очереди микробатчера; тяжёлые вычисления по-прежнему выполняются в threadpool через `anyio.to_thread`.
- `BGEEngine` расширен helper-методами для переиспользования подготовки текстов и низкоуровневого вызова `encode()` в микробатчере.


