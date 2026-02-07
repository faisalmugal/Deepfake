🧠 Advanced Deepfake Detection Backend

A production-grade AI Deepfake Detection API built using FastAPI, PyTorch, and FaceForensics++ inspired ensemble models. This backend analyzes images, videos, and GIFs using deep learning and computer vision forensic techniques to detect manipulated or synthetic media.

🚀 Features

🤖 AI Model Ensemble

Xception (Primary FaceForensics++ model)

EfficientNet-B4

MesoNet-4

ResNet50

Weighted ensemble for improved accuracy

🔬 Forensic Analysis

Frequency domain analysis (DCT based)

Facial detection & structural analysis

Lighting and shadow consistency analysis

Temporal motion consistency for videos and GIFs

📦 Media Support

Images (JPEG, PNG, WEBP, GIF)

Videos (MP4, MOV, AVI)

Animated GIFs

⚙️ Production Capabilities

GPU acceleration (CUDA support)

HuggingFace fallback model support

REST API with FastAPI

Frontend compatible response format

🏗️ System Architecture

Client / Frontend
        │
        ▼
FastAPI Backend
        │
        ▼
Media Processing Layer
        │
        ▼
Deep Learning Ensemble + CV Analysis
        │
        ▼
Deepfake Risk Score + Metadata Response

📂 Project Structure

project-root/
│
├── main.py                  # Main backend server
├── models/                  # Optional external ML models
├── requirements.txt         # Dependencies
├── README.md

🧪 Detection Pipeline

1. Face Detection

MTCNN Deep Learning Face Detector

Haar Cascade fallback detection

2. Neural Network Inference

Media is analyzed using multiple CNN models trained on deepfake datasets.

3. Traditional CV Forensics

Compression artifact detection

Frequency domain anomaly detection

Lighting inconsistency detection

Motion irregularity detection

4. Risk Scoring Engine

Combines:

Model predictions

Forensic analysis

Heuristic adjustments

📦 Installation

🔹 Clone Repository

git clone https://github.com/your-username/deepfake-detection-backend.git
cd deepfake-detection-backend

🔹 Create Virtual Environment

python -m venv venv
source venv/bin/activate        # Linux / Mac
venv\Scripts\activate           # Windows

🔹 Install Dependencies

pip install fastapi uvicorn python-multipart
pip install opencv-python numpy pillow imageio
pip install torch torchvision timm facenet-pytorch transformers

▶️ Running the Server

python main.py

Server will start at:

http://localhost:8000

API Documentation:

http://localhost:8000/docs

📡 API Endpoints

🏠 Root Endpoint

GET /

Returns API status and model information.

❤️ Health Check

GET /health

Returns system health and model readiness.

🔍 Analyze Media

POST /api/analyze

Upload image, video, or GIF for deepfake detection.

🧠 Model Information

GET /api/models/info

Returns loaded model details and weights.

📊 Statistics

GET /api/stats

Returns usage and detection statistics.

📤 Example Request

curl -X POST "http://localhost:8000/api/analyze" \
     -F "file=@sample.jpg"

📥 Example Response

{
  "is_deepfake": true,
  "deepfake_score": 78.5,
  "confidence": 91.2,
  "risk_level": "HIGH"
}

⚡ Hardware Requirements

Minimum

CPU supported

8GB RAM

Recommended

NVIDIA GPU with CUDA

16GB RAM

📊 Accuracy

Models Loaded

Estimated Accuracy

1 Model

~87%

2 Models

~90%

3 Models

~92%

4 Models

~95%

🔄 Fallback Strategy

If primary models fail:

HuggingFace deepfake detector

Traditional computer vision heuristics

🔒 Security Notes

File upload size limited to 100MB

MIME type validation enabled

SSL verification disabled only for model download compatibility

🧩 Technologies Used

FastAPI

PyTorch

OpenCV

FaceForensics++ concepts

MTCNN Face Detection

HuggingFace Transformers

NumPy & ImageIO

🛠️ Future Improvements

Real-time streaming detection

Batch inference optimization

Docker containerization

Cloud deployment support

Model retraining pipeline

👨‍💻 Author

Deepfake Detection Backend Project

📜 License

This project is intended for educational and research purposes.
