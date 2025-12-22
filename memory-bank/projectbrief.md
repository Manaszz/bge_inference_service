# Project Brief: BGE Inference Service

## Overview
High-performance inference microservice for BGE (FlagEmbedding) models, supporting dense, sparse, and hybrid embeddings + reranking.

## Core Goals
- Provide OpenAI-compatible embeddings API.
- Support hybrid search (dense + sparse) optimized for GPU.
- Isolate heavy ML dependencies from main applications.

## Rules & Constraints
- **Library Version Policy**: ALWAYS use the latest available versions of libraries.
- **Tools**: Use `context7` mandatory for version checks and documentation.
- **Deployment**: Docker-first, GPU-optimized (NVIDIA CUDA).
- **Environment**: Support closed-contour deployments with specific version mirrors.

