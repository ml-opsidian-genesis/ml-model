FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p models

ENV MODEL_VERSION="v1.0.0"

RUN wget -q -O models/flood_model.onnx "https://huggingface.co/teamfalsepositives/flood-risk-model/resolve/${MODEL_VERSION}/flood_model.onnx"
RUN wget -q -O models/flood_model.pkl "https://huggingface.co/teamfalsepositives/flood-risk-model/resolve/${MODEL_VERSION}/flood_model.pkl"

COPY . .

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
