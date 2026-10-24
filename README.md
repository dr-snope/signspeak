# SignSpeak

> **Real-Time Indian Sign Language (ISL) Detection & Transcription Engine**  
> *Bridging silence and sound through edge computer vision and deep sequence learning.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Landmark%20Extraction-orange?logo=google)
![License](https://img.shields.io/badge/License-MIT-purple)

---

## 🌟 Overview

**SignSpeak** is an AI-powered assistive platform designed to transcribe dynamic Indian Sign Language (ISL) gestures in real-time. By transforming continuous optical video into spatio-temporal coordinate trajectories, SignSpeak enables seamless communication for the deaf and hard-of-hearing community.

Featuring **Signy**, our official project mascot, the platform blends an accessible web interface with a high-performance computer vision pipeline.

---

## 🏗️ System Architecture

SignSpeak processes dynamic gestures using a 5-stage spatio-temporal pipeline:

- **1. Webcam Stream (30 FPS)**: Continuous video frame acquisition.
- **2. MediaPipe Hands Engine**: Extracts 21 3D Joint Coordinates (x, y, z).
- **3. Spatial Normalization**: Invariant to camera distance, hand size, and positioning.
- **4. Temporal Sequence Model**: LSTM / GRU classifies dynamic movement windows.
- **5. SignSpeak Web Interface**: Instant real-time transcription canvas.

---

## 📁 Repository Structure

Our codebase is organized into domain-driven subsystems:

- `frontend/` — Interactive UI, Mascot assets & styling

- `backend/` — Real-time computer vision engine

- `ml/` — Machine learning training & datasets

- `requirements.txt` — Universal dependency manifest
- `.gitignore` — Environment & dataset guardrails

---

## 🚀 Quickstart Guide

1. **Clone the Repository**
   `git clone https://github.com/dr-snope/signspeak.git`

2. **Set Up Virtual Environment**
   `python -m venv venv`
   `.\venv\Scripts\Activate.ps1`

3. **Install Dependencies**
   `pip install -r requirements.txt`

4. **Run Camera Test**
   `python camera_test.py`

---

## 🛡️ Engineering Workflow

This repository operates on a **Domain-Driven Branching Strategy**:
- `main`: Sacred production branch (always stable and deployable).
- `frontend-base` & `backend-base`: Subsystem integration branches.
- `*-base`: Isolated developer sandboxes for rapid feature development and testing.

---

## 👥 Contributors

- **Yatharth** — ML model & Systems Architect
- **Hasini** — UI/UX Design & Frontend Development
- **Abel** — Computer Vision & Backend Engineering
- *The SignSpeak Team*

---

<p align="center">
  <i>Built with ❤️ for accessible communication.</i>
</p>