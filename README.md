# SignSpeak

> **Real-Time Indian Sign Language (ISL) Detection & Transcription Engine**  
> *Bridging silence and sound through edge computer vision and lightweight machine learning.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Landmark%20Extraction-orange?logo=google)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Engine-F7931E?logo=scikit-learn&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## 🌟 Overview

**SignSpeak** is an AI-powered assistive platform designed to transcribe Indian Sign Language (ISL) gestures in real time. By transforming optical camera feeds into normalized geometric landmark vectors, SignSpeak delivers low-latency gesture classification without requiring heavy cloud compute or bulky deep-learning sequence runtimes.

Featuring **Signy**, our official project mascot, the platform pairs an intuitive web interface with a high-performance computer vision pipeline.

---

## 🏗️ System Architecture

SignSpeak executes an optimized 5-stage inference pipeline:

1. **Webcam Stream (30 FPS)**: Continuous video frame capture via OpenCV.
2. **MediaPipe Hands Engine**: High-fidelity detection and extraction of 21 3D joint landmarks (63 raw coordinate features).
3. **Spatial Normalization**: Centering all points relative to Landmark 0 (wrist) and scaling to `[-1.0, 1.0]` to guarantee camera distance and hand-scale invariance.
4. **Machine Learning Classifier**: Fast `scikit-learn` ensemble model (`RandomForestClassifier`) executing sub-millisecond inference with confidence thresholding.
5. **SignSpeak Web Interface**: Live transcription canvas rendering predictions and visual feedback dynamically.

---

## 📁 Repository Structure

Our codebase is organized into 3 primary subdirectories alongside essential root configurations:

```text
signspeak/
├── frontend/            
│   ├── assets/
│   ├── home.html
│   ├── about.html
│   ├── how_it_works.html
│   └── (TBA)
├── backend/              
│   └── (TBA)
├── ml/                   
│   ├── utilities.py      
│   ├── collect_data.py  
│   ├── train.py         
│   └── predict.py        
├── .gitignore            
├── requirements.txt      
└── README.md
```

---

## 🚀 Quickstart Guide

1. **Clone the Repository**
   ```bash
   git clone [https://github.com/dr-snope/signspeak.git](https://github.com/dr-snope/signspeak.git)
   cd signspeak
   ```

2. **Set Up Virtual Environment**
   * **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   * **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install Dependencies**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Train / Test Pipeline**
   * Collect gesture samples:
     ```bash
     python ml/collect_data.py
     ```
   * Train the classifier:
     ```bash
     python ml/train.py
     ```
   * Launch backend stream:
     ```bash
     python backend/main.py
     ```

---


## 👥 Contributors

- **Yatharth** — System Architect & Machine Learning Lead
- **Abel** — Computer Vision & Backend Lead
- **Hasini & Harshika** — UI/UX Design & Frontend Leads
- *The SignSpeak Team*

---

<p align="center">
  <i>Built with ❤️ for accessible communication.</i>
</p>