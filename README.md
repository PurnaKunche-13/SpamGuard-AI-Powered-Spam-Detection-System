# SpamGuard AI Powered Spam Detection System

SpamGuard is a full-stack spam detection project built with Python, NLP, PyTorch, and FastAPI. It includes a trainable TF-IDF based classifier, a lightweight inference API, and a frontend website for live predictions.

## Features

- TF-IDF text vectorization for short message classification
- PyTorch feed-forward classifier for spam vs ham prediction
- FastAPI backend with `/health`, `/predict`, and `/train` endpoints
- Static frontend for live message testing
- Sample dataset for local demos and a structure that supports larger datasets

## Project Structure

```text
Spamguard-ai/
├── backend/
│   └── app/
│       ├── main.py
│       ├── predictor.py
│       └── schemas.py
├── ml/
│   ├── artifacts/
│   ├── data/
│   │   └── sample_messages.csv
│   └── src/
│       ├── data_utils.py
│       ├── inference.py
│       ├── model.py
│       ├── preprocess.py
│       └── train.py
├── website/
│   ├── index.html
│   ├── public/
│   │   ├── css/
│   │   │   └── styles.css
│   │   └── js/
│   │       └── app.js
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Train the Model

```bash
python -m ml.src.train --data ml/data/sample_messages.csv
```

This writes model artifacts into `ml/artifacts/`.

## Run the API

```bash
uvicorn backend.app.main:app --reload
```

API base URL: `http://127.0.0.1:8000`

## Frontend

Open `website/index.html` in a browser after starting the API, or serve it with any static server.

## Notes

- The bundled dataset is intentionally small for demo purposes.
- For production-grade accuracy, train on a larger dataset such as SMS Spam Collection or a domain-specific email/SMS corpus.
- `transformers` and `huggingface-hub` are included so the project can be extended with Hugging Face models later without restructuring the repo.

