# System Patterns

## Architecture
- **Model-as-a-Service (MaaS)**: ML logic is encapsulated in a dedicated service.
- **FastAPI/Uvicorn**: Async web framework for high-throughput I/O.
- **Lazy Loading**: Models are loaded during the lifespan startup to ensure the API starts even if models fail initially (though health will show degraded).

## Dependency Pattern
- **Always Latest**: Use `context7` to find the newest versions.
- **Freeze Policy**: Post-install, always output `pip-freeze.txt` for traceability.

## Implementation Details
- **Microbatching**: Optional batching of incoming requests to improve GPU throughput.
- **Sparse Mapping**: Deterministic SHA256 or Tokenizer-based mapping for lexical weights.

