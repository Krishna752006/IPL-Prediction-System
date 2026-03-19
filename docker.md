# Docker Image Size Optimization

## Overview
During development of the IPL Prediction Platform, the Docker setup initially consumed **30+ GB of disk space**, causing performance issues and instability.

This document outlines the steps taken to optimize the Docker environment and reduce the total footprint to **~10 GB**.

## Optimization Steps

### 1. Lightweight Base Images
Replaced default images with slim alternatives:

| Before | After |
|------|------|
| python:3.11 | python:3.11-slim |
| node:18 | node:18-alpine |

**Impact:** Reduced base image size significantly.

### 2. Added `.dockerignore`
Prevented unnecessary files from being included in build context:

- `data/`, `models/`
- `.git`, `.env`
- `node_modules/`
- notebooks and logs

**Impact:** Avoided duplicating large files inside Docker layers.

### 3. Separated Training and Inference Environments

Removed heavy ML libraries from production:

- tensorflow
- keras
- scipy
- tensorboard
- ml_dtypes

Created a lightweight `requirements-prod.txt` for inference.

**Impact:** Major reduction in ML container size.

### 4. Disabled Pip Cache

Used:

```bash
pip install --no-cache-dir -r requirements-prod.txt
````

**Impact:** Prevented unnecessary storage of installation cache.

### 5. Externalized Model Storage

Instead of copying models into the image:

```yaml
volumes:
  - ./models:/app/models
```

**Impact:** Avoided embedding large model files into Docker images.


### 6. Optimized Node.js Dependencies

Used:

```bash
npm install --only=production
```

**Impact:** Excluded development dependencies.

### 7. Fixed Docker Compose Configuration

* Corrected volume definitions
* Structured services properly
* Avoided misconfigured storage layers

### 8. Used Named Volumes for MongoDB

```yaml
volumes:
  mongo_data:
```

**Impact:** Controlled persistent storage growth.

### 9. Cleaned Docker System

Executed:

```bash
docker system prune -a --volumes
docker builder prune -a
```

**Impact:** Removed unused images, containers, and cache layers.

### 10. Clean Rebuild

Executed:

```bash
docker compose build --no-cache
```

**Impact:** Eliminated legacy layers and ensured optimized builds.

## Final Results

| Metric     | Before | After  |
| ---------- | ------ | ------ |
| Disk Usage | 30+ GB | ~10 GB |


## Key Learnings

* Production containers should only include **inference dependencies**
* Never bundle **datasets or training pipelines** into Docker images
* Use `.dockerignore` aggressively
* Monitor Docker cache regularly
* Optimize base images and dependencies early

## Future Improvements

* Further reduce size by removing pandas (if possible)
* Use ONNX / model compression for inference
* Introduce container image scanning and optimization tools