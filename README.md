
# 🎬 Local Video Intelligence Platform

> **A GPU-constrained video analysis platform designed for laptops with limited VRAM (6GB).**

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Ollama](https://img.shields.io/badge/AI-Ollama-black?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## 📖 Overview

Running modern AI pipelines usually requires massive server-grade GPUs. This project solves that problem. It is a **Video RAG (Retrieval Augmented Generation)** system specifically engineered to run on consumer hardware (like an NVIDIA RTX 3060 Laptop).

It allows you to upload videos, automatically extract insights, and **chat with your video library** using a local LLM, all without crashing your GPU memory.

### Key Capabilities
* **Video Ingestion:** Upload MP4/MOV/MKV (up to 500MB).
* **Automatic Transcription:** Speech-to-text with millisecond timestamps.
* **Object Detection:** Identify people, cars, and objects in every frame.
* **Semantic Search:** "Find the part where they talk about the budget."
* **Q&A:** "What color was the car in the video?" (Powered by Llama 3).

---

## 🏗️ Architecture

The system uses a microservices architecture with a custom **GPU Governor** to manage VRAM resources.

### System Design
```mermaid
graph TD
    Client[User / Client] -->|HTTP Upload| API[FastAPI Container]
    API -->|Metadata| DB[(PostgreSQL)]
    API -->|Video File| MinIO[(MinIO Storage)]
    API -->|Task| Redis[(Redis Queue)]
    
    subgraph "GPU Worker Node"
        Worker[Celery Worker]
        Gov[Mutex Governor]
        
        Worker -->|Check VRAM| Gov
        Gov -->|Load/Unload| Whisper[Faster-Whisper]
        Gov -->|Load/Unload| YOLO[YOLOv8]
        Gov -->|Load/Unload| Embed[Embedding Model]
        Gov -->|External Call| Ollama[Ollama Container]
    end
    
    Worker -->|Vectors| Milvus[(Milvus Vector DB)]
````

### The "Mutex Governor" (How it works on 6GB VRAM)

To prevent Out-Of-Memory (OOM) errors, the system implements an **Interlock Protocol**:

1.  **State Check:** Monitors VRAM usage (Green \< 4GB, Yellow 4-5GB, Red \> 5GB).
2.  **Locking:** Only *one* heavyweight model (Whisper, YOLO, or LLM) can be loaded onto the GPU at a time.
3.  **Cleanup:** Aggressively unloads models and clears CUDA cache between pipeline stages.

-----

## ✨ Features

| Feature | Tech Stack | Details |
| :--- | :--- | :--- |
| **Speech-to-Text** | Faster-Whisper | Large-v3 model (Int8 quantized) for high accuracy. |
| **Object Detection** | YOLOv8s | Detects 80+ classes @ 1 FPS. |
| **Vector Database** | Milvus | Stores embeddings for semantic search. |
| **LLM Inference** | Ollama + Llama 3 | Running locally for privacy and zero cost. |
| **Orchestrator** | Celery + Redis | Handles async processing of long videos. |

-----

## 🚀 Quick Start

### Prerequisites

  * **OS:** Linux or Windows (via WSL2).
  * **GPU:** NVIDIA GPU (6GB+ VRAM recommended).
  * **Drivers:** NVIDIA Driver 525.0+ & Container Toolkit installed.
  * **Docker:** Docker Desktop or Engine.

### 1\. Clone & Setup

```bash
git clone [https://github.com/yourusername/video-intelligence-platform.git](https://github.com/yourusername/video-intelligence-platform.git)
cd video-intelligence-platform

# Setup environment variables
cp .env.example .env
```

### 2\. Launch Infrastructure

This starts the database, vector store, and object storage.

```bash
cd infra
docker compose up -d
```

> *Note: First run may take 10-20 minutes to download Docker images and AI models.*

### 3\. Initialize LLM

We need to pull the Llama 3 model into the Ollama container.

```bash
docker exec -it infra-ollama-1 ollama pull llama3:8b
```

### 4\. Verify Installation

```bash
curl http://localhost:8000/health
```

*Expected output: `{"status": "healthy", "gpu_available": true, ...}`*

-----

## 💻 API Usage Examples

### 1\. Upload a Video

```bash
curl -X POST "http://localhost:8000/videos" \
  -F "file=@/path/to/my_video.mp4"
```

### 2\. Get Analysis Results

\<details\>
\<summary\>\<strong\>👇 Click to see Transcript Response JSON\</strong\>\</summary\>

```json
{
  "segments": [
    {
      "start_ms": 0,
      "end_ms": 3500,
      "text": "Hello and welcome to this video.",
      "confidence": 0.95
    }
  ],
  "language": "en",
  "duration_ms": 120000
}
```

\</details\>

\<details\>
\<summary\>\<strong\>👇 Click to see Object Detection JSON\</strong\>\</summary\>

```json
{
  "frames": [
    {
      "timestamp_ms": 1500,
      "detections": [
        { "class_name": "person", "confidence": 0.92, "bbox": [0.1, 0.2, 0.5, 0.8] }
      ]
    }
  ],
  "unique_objects": { "person": 5, "car": 2 }
}
```

\</details\>

### 3\. Ask Questions (RAG)

```bash
curl -X POST "http://localhost:8000/queries" \
  -H "Content-Type: application/json" \
  -d '{"query": "What happens in the video?", "video_id": "your-video-id"}'
```

**Response:**

```json
{
  "answer": "The video shows a person entering a room at 00:15 and discussing project updates.",
  "sources": [
    { "start_ms": 15000, "text": "Let me give you the project updates", "score": 0.89 }
  ]
}
```

-----

## 🛠️ Configuration (.env)

| Variable | Default | Description |
| :--- | :--- | :--- |
| `MAX_VIDEO_SIZE_MB` | 500 | Max upload size limit |
| `VRAM_GREEN_THRESHOLD` | 4.0 | Safety threshold for loading new models (GB) |
| `OLLAMA_MODEL` | llama3:8b | The LLM used for Q\&A |
| `MINIO_ENDPOINT` | minio:9000 | S3-compatible storage URL |

-----

## 🐛 Troubleshooting

**Processing is stuck?**
Check the worker logs to see if the GPU is hanging.

```bash
docker compose logs -f worker
```

**Out of Memory (OOM)?**
Force a GPU cleanup via the administrative endpoint:

```bash
curl -X POST "http://localhost:8000/health/gpu/cleanup"
```

**Ollama model not found?**
Ensure the model is pulled inside the container:

```bash
docker exec -it infra-ollama-1 ollama list
```

-----

## 📁 Project Structure

```text
video-intelligence-platform/
├── infra/                  # Docker infrastructure (Compose, Dockerfiles)
├── src/
│   ├── api/                # FastAPI Endpoints & Schemas
│   ├── governor/           # GPU VRAM Manager (The "Brain")
│   ├── services/           # AI Pipelines (Whisper, YOLO, RAG)
│   ├── db/                 # Database Models
│   └── workers/            # Celery Task Definitions
├── tests/                  # Pytest suite
└── .env.example            # Config template
```

-----

## 🤝 Contributing

1.  Fork the repository
2.  Create your feature branch (`git checkout -b feature/amazing-feature`)
3.  Commit your changes (`git commit -m 'Add some amazing feature'`)
4.  Push to the branch (`git push origin feature/amazing-feature`)
5.  Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
