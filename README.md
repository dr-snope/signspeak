# MudraSetu

> **Real-Time Indian Sign Language (ISL) Detection & Transcription Engine**  
> *Bridging silence and sound through edge computer vision and lightweight machine learning.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-009688?logo=fastapi&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Landmark%20Extraction-orange?logo=google)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Engine-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## 🌟 Overview

**MudraSetu** is an AI-powered assistive web platform designed to transcribe Indian Sign Language (ISL) gestures in real time. By transforming browser webcam video feeds into normalized geometric landmark vectors, MudraSetu delivers low-latency gesture classification without requiring heavy cloud runtimes.

---

## 🏗️ System Architecture

MudraSetu executes an optimized real-time client-server inference loop:

1. **Browser Camera Capture**: Client webcam feeds are drawn to an HTML5 canvas and transmitted over a WebSocket connection.
2. **MediaPipe Hand Landmark Tracking**: High-fidelity detection and extraction of 21 3D joint coordinates (63 raw coordinate features) on the server.
3. **Geometric Spatial Normalization**: Centering all points relative to Landmark 0 (wrist) and scaling to `[-1.0, 1.0]` to guarantee camera distance, hand-scale, and screen-position invariance.
4. **Machine Learning Inference**: A `RandomForestClassifier` running sub-millisecond predictions gated by a confidence threshold (`0.65`) to eliminate gesture-transition noise.
5. **Real-Time Client Feedback**: The server returns JSON payloads containing recognized sign labels, confidence scores, and joint coordinates to render live green skeleton overlays on the browser preview.

---

## 📁 Repository Structure

```text
mudrasetu/
├── backend/
│   ├── camera.py
│   ├── download_model.py
│   ├── server.py
│   └── tracker.py
├── frontend/
│   ├── assets/
│   ├── about.html
│   ├── app.js
│   ├── home.html
│   ├── how-it-works.html
│   ├── style.css
│   ├── style2.css
│   └── style3.css
├── ml/
│   ├── collect_data.py
│   ├── predict.py
│   ├── train.py
│   └── utilities.py
├── .gitignore
├── README.md
└── requirements.txt
```

---

## 🚀 Quickstart Guide (Note that this project was built with Python 3.11, so using the same version is recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/nexushci/mudrasetu.git
cd mudrasetu
```

### 2. Set Up Virtual Environment
- **Windows (PowerShell):**
  ```
  py -3.11 -m venv .venv
  ```
- **Linux / macOS:**
  ```
  python3.11 -m venv .venv
  ```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Data Collection & Model Training
- **Record gesture samples:**
  ```
  ml/collect_data.py
  ```
- **Train and export classifier:**
  ```
  ml/train.py
  ```

### 5. Launch MudraSetu
1. **Start the FastAPI Backend:**
   ```
   uvicorn backend.server:app --reload --port 8000
   ```
2. **Open the Frontend:**
   -  Open `frontend/home.html` in your browser (or via VS Code Live Server).
   -  Click **"Start Camera"** to grant webcam access and begin live sign transcription.

---

## 👥 Contributors

- **Yatharth** — System Architecture & Machine Learning Lead
- **Abel** — Computer Vision & Backend Lead
- **Hasini & Harshika** — UI/UX Design & Frontend Leads
- *The MudraSetu Team | NexusHCI*

---

<p align="center">
  <i>Built with ❤️ for accessible communication.</i>
</p>
