# IPL Project

A full-stack application with a Python backend, ML service, and frontend for IPL data/analysis.

## Getting Started

### 1. Install Dependencies

Install all required Python packages using `uv`:

```bash
uv pip install -r requirements.txt
```

### 2. Backend Setup

Navigate to the backend folder and verify that everything is working:

```bash
cd backend
py .\verify_simulation.py
```

Then set up your environment:

- Connect to **MongoDB**
- Fill in your `.env` file with the required credentials/config

Start the backend server:

```bash
uvicorn main:app --reload
```

### 3. Frontend Setup

Open a **new terminal** and run:

```bash
cd frontend
npm i
npm run dev
```

Once running, open your browser and go to:

```
localhost:1537
```

Enjoy the app! 🎉

## Extra: ML Service

To view ML experiment runs, navigate to the `ml-service` folder and run:

```bash
mlflow ui --backend-store-uri sqlite:///experiments/mlruns.db --port 5000
```

This launches the MLflow UI on `localhost:5000`.
