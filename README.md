# NASA Turbofan Engine RUL Prediction API

This project implements a **Predictive Maintenance** solution using an **LSTM (Long Short-Term Memory)** neural network to predict the **Remaining Useful Life (RUL)** of aircraft engines. The model is served via a **FastAPI** REST interface and containerized using **Docker**.

## 🚀 Features
- **Predictive Modeling:** LSTM architecture trained on the NASA CMAPSS dataset (FD001).
- **Production Ready:** Scalable FastAPI implementation with asynchronous lifespan management.
- **Containerized:** Fully Dockerized for consistent deployment across environments.
- **Data Validation:** Strict input validation using Pydantic schemas.

- ## 🛠️ Tech Stack
- **Framework:** FastAPI
- **Machine Learning:** PyTorch, Scikit-learn
- **Data Handling:** Pandas, NumPy
- **Containerization:** Docker
- **Server:** Uvicorn
- ## 📂 Project Structure
```text
├── app/
│   ├── main.py          # FastAPI application & lifespan logic
│   ├── schemas.py       # Pydantic request/response models
│   ├── utils.py         # LSTM Model class & transformation logic
│   ├── config.py        # Pydantic-settings configuration
│   └── __init__.py
├── models/
│   ├── model.pth        # Trained PyTorch Model
│   └── scaler.pkl       # Scikit-learn Scaler
├── Dockerfile           # Optimized multi-layer build
├── requirements.txt     # Python dependencies
└── .dockerignore        # Build context optimization
```

- ## ⚡ Quick Start

- ### Using Docker (Recommended)
  This is the easiest way to run the API without worrying about local dependencies.

1. **Build the image:**
   ```bash
   docker build -t rul-api .
   ```
2. **Run the container:**
   ```bash
   docker run -p 8000:8000 rul-api
   ```

 3. **Explore the API:**
    
    Once running, visit http://localhost:8000/docs to use the interactive Swagger UI.

- ## ⚡ Local Development
  If you want to run it without Docker:
  
1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/scripts/activate  # On Windows: venv\Scripts\activate
   ```


2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
    ```
3. **Start the server:**
   ```bash
   uvicorn app.main:app --reload
   ```

  

## 📊 Model & Preprocessing Details
### Dataset
I used the **NASA CMAPSS Turbofan Engine Degradation Dataset (FD001)**. The model is trained to predict Remaining Useful Life based on multivariate time-series sensor data.

### Preprocessing Pipeline
- **Sliding Window:** Input data is transformed into sequences of **30 time cycles**.
- **Feature Selection:** 21 sensor measurements are used as input features.
- **Scaling:** A `StandardScaler` was fitted on the "healthy" phase (the first 30% of engine life) to establish a baseline for normal operation.
- **RUL Clipping:** Targets were clipped at **125 cycles** to prevent the model from trying to predict overly large values during early-life stages where degradation is negligible.

### Architecture
- **Type:** LSTM (Long Short-Term Memory)
- **Hidden Layers:** 64 units
- **Regularization:** Dropout (0.2)
- **Output:** Linear layer predicting a single RUL value.
   
## 📡 API Endpoints

### `POST /predict`
Submit a sensor window to get an RUL prediction.

**Request Body:**
```json
{
  "engine_id": 1,
  "window": [
    [518.67, 1.3, 47.47, ...], // Cycle 1 (21 sensors)
    [518.67, 1.3, 47.48, ...], // Cycle 2
    ... // Continued for 30 cycles
  ]
}
```
**Response Body:**
```json
{
  "predicted_RUL": 124.52,
  "engine_id": 1
}
```
### `GET /health`
Returns {"status": "healthy"} if the model and scaler are loaded successfully.
