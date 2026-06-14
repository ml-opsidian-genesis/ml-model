# FloodRisk Prediction Model

This project contains a machine learning pipeline and a FastAPI application to predict flood risk scores based on geographical and environmental features.

## Quick Start (Environment Setup)

Run the following commands in the terminal to set up the environment:

### 1. Run the Setup Script
From the root of `ml-model/`, run the appropriate script for your operating system:

**Linux / macOS:**
```bash
./setup_env.sh
```

**Windows:**
```cmd
setup_env.bat
```

This script will:
- Check for Python 3.
- Create a virtual environment named `.venv`.
- Upgrade `pip` to the latest version.
- Install all required dependencies from `requirements.txt`.

### 2. Activate the Environment
Before running any code, make sure to activate your virtual environment:

**Linux / macOS:**
```bash
source .venv/bin/activate
```

**Windows:**
```cmd
.venv\Scripts\activate
```

---

## Project Structure

- `app/main.py`: The FastAPI server exposing the flood risk prediction model.
- `src/`: Contains the pipeline logic, model evaluation, and training scripts.
  - `train.py`: The main script to train the model and save artifacts.
  - `pipeline.py`: Feature engineering functions.
  - `evaluate.py`: Evaluation metrics and logging functions.
- `models/`: Directory where the trained models (`.pkl`, `.onnx`) are stored.
- `requirements.txt`: Python package dependencies (including `fastapi`, `lightgbm`, `onnxruntime`, `loguru`, etc.).

---

## Training the Model

To run the training pipeline and generate the model artifacts (saved into `models/`), ensure your virtual environment is activated and execute:

```bash
python -m src.train
```

*Note: Make sure your dataset files (`train.csv` and `test.csv`) are located either in the root directory or in a `data/` folder as expected by the training script.*

---

## Running the API

We use **FastAPI** to serve the predictions. To start the local server, run:

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### API Documentation
Once the server is running, you can explore the interactive API documentation:
- **Swagger UI:** `http://127.0.0.1:8000/docs`
- **ReDoc:** `http://127.0.0.1:8000/redoc`

---

## Troubleshooting

- **Import Errors:** If you experience `ModuleNotFoundError` (such as `loguru` or `onnxruntime`), ensure your virtual environment is active and up to date by running `./setup_env.sh` again.
- **Permission Denied:** If the setup script fails due to permissions, ensure it is executable by running `chmod +x setup_env.sh`.
