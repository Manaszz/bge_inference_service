# Active Context

## Current Focus
- Environment stabilization and dependency management in restricted network conditions.

## Recent Changes
- Updated `requirements.txt` to newer versions (FastAPI 0.122.0, etc.) to match mirror availability.
- Fixed `BGEEngine.rerank` bug (missing `_prepare_texts` method).
- Rebuilt Docker image with GPU passthrough confirmed.
- Initialized Memory Bank and `.cursorrules` with the "Always Latest" library policy.

## Next Steps
- Maintain `pip-freeze.txt` as the source of truth for the installed environment.

