# Enterprise Knowledge Assistant

A production-ready RAG-based AI assistant with observability, monitoring, caching, and safety controls.

---

## 📦 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/raushan291/enterprise-ai-assistant.git
cd enterprise-ai-assistant
pip install -r requirements.txt
```

## Septup pre-commit hooks

1. Install pre-commit
```bash
pip install pre-commit
```

2. Install the hooks to your git repo
```bash
pre-commit install
```
This sets up pre-commit so it runs automatically before every commit.

3. Run on all files one time (optional but recommended)
```bash
pre-commit run --all-files
```
---

## ⚙️ Environment Setup

```bash
cp .env.example .env
```

Edit `.env` and set values.

---

## 🚀 Run the Backend API and Chat UI

Ensure your Python environment is set up with dependencies installed.

Start the FastAPI server:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8001
```

Start the Streamlit UI:

```bash
streamlit run ui/chat_app.py
```

---

## 🧠 Vector Database (ChromaDB) Setup

ChromaDB stores embeddings for search and retrieval.

```bash
mkdir -p data/chroma_db

docker run -d \
  --name chroma \
  -p 8000:8000 \
  -e CHROMA_DB_IMPL=duckdb+parquet \
  -e CHROMA_PERSIST_DIRECTORY=/data \
  -v $(pwd)/data/chroma_db:/data \
  chromadb/chroma:latest
```

## 🔧 Redis Setup

Redis is used for caching and memory store.

 **Option 1: Start Redis locally**

```bash
redis-server --daemonize yes
```

**Option 2: Run Redis with Docker**

```bash
docker run -d --name redis -p 6379:6379 redis:latest
```

---

## 📊 Install and Start Grafana

Grafana is used for monitoring dashboard visualization.

```bash
sudo apt-get install -y apt-transport-https software-properties-common wget
sudo mkdir -p /etc/apt/keyrings/
wget -q -O - https://packages.grafana.com/gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/grafana.gpg
echo "deb [signed-by=/etc/apt/keyrings/grafana.gpg] https://packages.grafana.com/oss/deb stable main" | sudo tee /etc/apt/sources.list.d/grafana.list
sudo apt-get update
sudo apt-get install grafana -y
```

Start and enable Grafana:

```bash
sudo systemctl start grafana-server
sudo systemctl enable grafana-server
```

Access Grafana UI:

```
http://localhost:3000
```

(Default login: `admin` / `admin`)

---

## 📈 Prometheus Setup

Prometheus collects metrics for Grafana.

Install Prometheus:

```bash
sudo apt install prometheus
```

Run with custom config:

```bash
prometheus --config.file=src/config/prometheus.yml --storage.tsdb.path=data/prom_data
```

---

## 🛡️ Guardrails Setup

Generate your Guardrails API key:
https://hub.guardrailsai.com/keys

Configure Guardrails:
```bash
guardrails configure
```

Install validators:
```bash
guardrails hub install hub://guardrails/detect_pii
guardrails hub install hub://guardrails/toxic_language
```

---

## 🏗️ System Architecture
* **FastAPI** backend for API
* **ChromaDB** as the vector database for embeddings storage and retrieval
* **Redis** for caching and conversation memory
* **Guardrails** for safety validation
* **Pydantic** for data validation and serialization
* **Prometheus** for metrics ingestion
* **Grafana** for monitoring and dashboard visualization

---

## 📬 Support

For issues, create a GitHub Issue or contact the maintainer: [raushan291](https://github.com/raushan291)

---

## 📄 License

This project is licensed under the **PolyForm Noncommercial License 1.0.0**.

You may use, modify, and share this project for **personal, academic, or research purposes**.  
**Commercial use is not permitted.**

See the `LICENSE` file for the full text.

---

Happy building! ⚡
