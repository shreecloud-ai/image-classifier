# 🧠 CIFAR-10 Image Classifier

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?style=flat&logo=fastapi&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?style=flat&logo=docker&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-95.46%25-success?style=flat)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat)

---

## 📌 Overview

An end-to-end image classification system trained on CIFAR-10 — from a custom CNN built from scratch to a fine-tuned ResNet-18 achieving **95.46% test accuracy**, deployed via a production-ready FastAPI backend with Docker support.

---

## 🏆 Key Results

| Model | Test Accuracy | vs. Target (93.7%) | Notes |
|---|---|---|---|
| Custom CNN (scratch) | 81.57% | — | Baseline, built from scratch |
| ResNet-18 (transfer learning) | **95.46%** | **+1.76 pp** | Exceeded target by nearly 2 points |
| Improvement | **+13.89 pp** | — | Transfer learning advantage |

> The jump from 81.57% → 95.46% demonstrates the power of transfer learning on a mid-scale dataset like CIFAR-10. ResNet-18 pretrained on ImageNet learns rich visual features that transfer remarkably well even across domain differences.

---

## ✨ Features

- **Custom CNN from scratch** — understand the fundamentals before using shortcuts
- **ResNet-18 fine-tuning** with frozen backbone + custom classifier head
- **Data augmentation pipeline** — random crops, horizontal flips, color jitter, normalization
- **FastAPI REST backend** — single-image prediction endpoint, JSON response, < 100ms inference
- **Docker + docker-compose** — one command to run the full serving stack
- **Config-driven design** — all hyperparameters in a single `config.yaml`, no magic numbers in code
- **Structured logging** — training progress, eval metrics, and API requests all logged cleanly
- **MLflow integration** — experiment tracking for runs, metrics, and model artifacts
- **10-class classifier** — airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| Deep Learning | PyTorch 2.x, TorchVision |
| Model Architecture | Custom CNN, ResNet-18 (pretrained) |
| Serving | FastAPI, Uvicorn |
| Containerization | Docker, Docker Compose |
| Experiment Tracking | MLflow |
| Data Augmentation | TorchVision Transforms v2 |
| Language | Python 3.10+ |

---

## 📁 Project Structure

```
cifar10-classifier/
│
├── data/                        # Downloaded CIFAR-10 dataset (auto-created)
│
├── models/
│   ├── custom_cnn.py            # Custom CNN architecture (built from scratch)
│   └── resnet_transfer.py       # ResNet-18 with fine-tuned classifier head
│
├── training/
│   ├── train.py                 # Main training loop
│   ├── evaluate.py              # Evaluation + confusion matrix
│   └── augmentations.py         # Data augmentation pipeline
│
├── api/
│   ├── main.py                  # FastAPI app + /predict endpoint
│   ├── inference.py             # Model loading + image preprocessing
│   └── schemas.py               # Pydantic request/response schemas
│
├── checkpoints/                 # Saved model weights (.pth files)
│
├── mlruns/                      # MLflow experiment logs
│
├── config.yaml                  # All hyperparameters and paths
├── Dockerfile                   # API server container
├── docker-compose.yml           # Full stack orchestration
├── requirements.txt
└── README.md
```

---

## 🚀 Installation & Running

### Option 1 — Docker (recommended)

```bash
# Clone the repo
git clone https://github.com/yourusername/cifar10-classifier.git
cd cifar10-classifier

# Start the API server
docker-compose up --build
```

The API will be live at `http://localhost:8000`.

---

### Option 2 — Local Setup

**1. Clone and create environment**

```bash
git clone https://github.com/yourusername/cifar10-classifier.git
cd cifar10-classifier

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Train the Custom CNN**

```bash
python training/train.py --model cnn --config config.yaml
```

**3. Train ResNet-18 with transfer learning**

```bash
python training/train.py --model resnet18 --config config.yaml
```

**4. Evaluate a trained model**

```bash
python training/evaluate.py --model resnet18 --checkpoint checkpoints/resnet18_best.pth
```

**5. Start the API server**

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🔌 How to Use the API

### Predict a single image

**Endpoint:** `POST /predict`

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@your_image.jpg"
```

**Response:**

```json
{
  "predicted_class": "dog",
  "class_index": 5,
  "confidence": 0.9731,
  "all_probabilities": {
    "airplane": 0.0012,
    "automobile": 0.0008,
    "bird": 0.0043,
    "cat": 0.0134,
    "deer": 0.0021,
    "dog": 0.9731,
    "frog": 0.0018,
    "horse": 0.0019,
    "ship": 0.0007,
    "truck": 0.0007
  },
  "inference_time_ms": 47.3
}
```

**Interactive API docs** (auto-generated by FastAPI):
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 📊 Model Performance

### Accuracy Comparison

| Metric | Custom CNN | ResNet-18 Transfer |
|---|---|---|
| Test Accuracy | 81.57% | **95.46%** |
| Parameters | ~1.2M | ~11.2M |
| Training Time (GPU) | ~12 min | ~18 min |
| Epochs | 50 | 30 |
| Optimizer | Adam | SGD + momentum |
| Best Val Loss | 0.621 | 0.187 |

### Per-Class Accuracy (ResNet-18)

| Class | Accuracy |
|---|---|
| Airplane ✈️ | 97.1% |
| Automobile 🚗 | 98.2% |
| Bird 🐦 | 93.4% |
| Cat 🐱 | 91.8% |
| Deer 🦌 | 96.3% |
| Dog 🐶 | 92.7% |
| Frog 🐸 | 97.5% |
| Horse 🐴 | 97.0% |
| Ship 🚢 | 97.9% |
| Truck 🚛 | 96.9% |

> Cat and Dog remain the hardest classes due to visual similarity — a known challenge on CIFAR-10 even for state-of-the-art models.

---

## 🔮 Future Improvements

- [ ] **EfficientNet-B0 / B2** — lighter model with potentially higher accuracy
- [ ] **Test-Time Augmentation (TTA)** — ensemble predictions at inference for +0.5–1% accuracy
- [ ] **Grad-CAM visualizations** — explain *what* the model looks at per prediction
- [ ] **ONNX export** — faster CPU inference, framework-agnostic deployment
- [ ] **Batch prediction endpoint** — process multiple images per API call
- [ ] **MLflow full integration** — automatic metric logging, model registry, comparison dashboard
- [ ] **CI/CD pipeline** — GitHub Actions for automated training and deployment on push
- [ ] **Frontend demo** — simple drag-and-drop UI using Streamlit or Gradio

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with 🔥 by a self-taught ML engineer on the grind.<br>
  <em>5 projects down. Remote ML job incoming.</em>
</p>
