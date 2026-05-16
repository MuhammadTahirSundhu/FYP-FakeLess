# FYP-FakeLess Project: File Overview and Architecture

This document explains the purpose of each major component in the FYP-FakeLess project. The system uses a **Hybrid Architecture**, consisting of a Flutter mobile app, a cloud-deployed Protection API (Render), and a local GPU-accelerated Attacker API.

---

## 1. Cloud Backend (Protection API on Render)

### CloneShield/render_deployment/app.py
- **Purpose:** Cloud-based FastAPI server deployed on Render. Acts as the primary gateway for the mobile app.
- **Key Endpoints:**
  - `/protect`: Receives audio, applies the CloneShield perturbation model (CPU inference), and returns the protected audio. Also calculates PESQ and SNR metrics.
  - `/clone`: Acts as a secure proxy. Receives audio and forwards it via `ngrok` to the Local Attacker API for voice cloning.
- **Interconnection:** Communicates directly with the Flutter app. Proxies cloning requests to the local GPU machine to bypass Render's lack of free GPUs.

---

## 2. Local Backend (Attacker API for Verification)

### CloneShield/attacker_api.py
- **Purpose:** Local FastAPI server running on a machine with a dedicated NVIDIA GPU.
- **Key Endpoints:**
  - `/clone`: Receives audio and text, runs the **XTTS v2** model natively on the GPU to generate a cloned voice, and returns the cloned audio.
- **Interconnection:** Exposed to the internet via an `ngrok` tunnel. The Render Cloud API forwards requests to this tunnel.

---

## 3. Frontend (Flutter Mobile App)

### Code/mobile_app_fakeless/lib/main.dart & audio_page.dart
- **Purpose:** The user-facing mobile application.
- **Key Features:**
  - Audio recording and playback (`just_audio`, `record`).
  - End-to-end encryption of audio files before transmission.
  - Communicates with the Render `/protect` and `/clone` endpoints.
  - Allows users to compare original, protected, and cloned audio to verify the defense.

---

## 4. Testing & Benchmarking

### CloneShield/test_latency_render.py
- **Purpose:** Comprehensive latency benchmarking suite.
- **Functionality:** 
  - Measures Cold Start and Warm Server latency for both the Protection API and the Cloner API.
  - Calculates Time-To-First-Byte (TTFB) and handles timeouts gracefully.
  - Generates detailed CSV reports (`latency_results.csv` and `latency_results_aggregate.csv`).

---

## 5. Core Machine Learning Pipeline

### CloneShield/cloneshield/
- Contains the core model architecture (`model.py`), custom losses (`losses.py`), and dataset loaders. 
- Used to train the U-Net based perturbation generator that protects audio from XTTS v2.

### CloneShield/train.py & protect.py
- **Purpose:** Scripts for training the CloneShield model locally on a GPU and applying the protection manually to files for testing.

---

## 6. Workflow and Data Flow

1. **User Action:** The user records audio in the Flutter app.
2. **Protection Request:** The app encrypts and sends the audio to the Render API (`/protect`).
3. **Protection Inference:** Render runs the Perturbation Generator on CPU, adding imperceptible noise to the audio, and returns it.
4. **Cloning Request:** The app sends the original/protected audio to Render (`/clone`).
5. **Proxy to GPU:** Render proxies the request via `ngrok` to the Local Attacker API.
6. **Voice Cloning:** The Local API runs XTTS v2 on the GPU to generate a fake voice and sends it back up the chain.
7. **Verification:** The user plays the clones in the app to verify that the protected audio successfully degrades the cloning attempt.

---

## 7. Summary Table

| Component                          | Location                                | Role                                              |
|------------------------------------|-----------------------------------------|---------------------------------------------------|
| **Protection API (Cloud)**         | `CloneShield/render_deployment/app.py`  | Applies audio defense, proxies clone requests     |
| **Attacker API (Local GPU)**       | `CloneShield/attacker_api.py`           | Runs XTTS v2 voice cloning via ngrok              |
| **Mobile App (Frontend)**          | `Code/mobile_app_fakeless/`             | UI, recording, playback, API communication        |
| **Latency Benchmark**              | `CloneShield/test_latency_render.py`    | Evaluates API response times                      |
| **Model Training**                 | `CloneShield/train.py`                  | Trains the adversarial audio defense network      |