# FYP-FakeLess: Audio Deepfake Prevention System

FakeLess (CloneShield) is a comprehensive defense system designed to protect voice audio against unauthorized cloning by state-of-the-art Voice Conversion (VC) and Text-to-Speech (TTS) models like **XTTS v2**. 

By injecting carefully crafted, inaudible adversarial perturbations into the original audio, FakeLess disrupts the speaker embedding extraction process of cloning models, causing them to generate severely degraded or incorrect voices when attempting to clone the protected sample.

---

## 🌟 System Architecture

To ensure the system is accessible via a mobile application while overcoming the high computational requirements of Voice Cloning, we employ a **Hybrid Deployment Architecture**:

1. **Frontend (Flutter Mobile App):** Allows users to record audio, request protection, and play back side-by-side comparisons of original vs. cloned voices. End-to-end encryption secures the audio during transmission.
2. **Protection API (Render Cloud):** A FastAPI service hosted on Render that processes incoming audio, applies the perturbation filter (CPU inference), and securely proxies cloning requests.
3. **Attacker API (Local GPU via ngrok):** A local FastAPI server running on a machine with a dedicated GPU. It runs the heavy **XTTS v2** model natively and is exposed to the Render Cloud API via an `ngrok` secure tunnel.

---

## 🚀 Quick Start Guide

To run the entire ecosystem, follow these steps in order:

### 1. Start the Local Attacker API (GPU required)
The attacker API runs the XTTS v2 model. It must be run locally on a machine with a CUDA-enabled GPU.

```bash
# 1. Activate the environment
conda activate xtts

# 2. Navigate to the project folder
cd CloneShield

# 3. Run the Attacker API
uvicorn attacker_api:app --host 127.0.0.1 --port 8000
```

### 2. Expose the Attacker API via ngrok
In a new terminal window, expose the local port 8000 to the internet so Render can communicate with it:

```bash
ngrok http 8000
```
*Copy the generated forwarding URL (e.g., `https://xxxx.ngrok-free.app`).*

### 3. Deploy/Configure the Render Protection API
Ensure your `app.py` in `CloneShield/render_deployment/` is deployed to Render. 
In your Render Dashboard, add the following Environment Variable:
- `NGROK_URL` = `<your_ngrok_url_from_step_2>`

### 4. Run the Flutter Mobile App
Ensure the Flutter app (`Code/mobile_app_fakeless`) points to your Render API URL (e.g., `https://fakeless-api.onrender.com`).
Connect an Android device and run:
```bash
cd Code/mobile_app_fakeless
flutter run
```

---

## 📊 Latency Benchmarking

Because the Protection API runs on a Free-Tier CPU and the Attacker API relies on a remote GPU tunnel, we provide a benchmarking script to measure end-to-end latency.

### Running the Benchmark
```bash
conda activate xtts
cd CloneShield
python test_latency_render.py --audio sound_samples/input/speaker.wav --warm-runs 5
```

### Recent Benchmark Results
| Component | Scenario | Avg Response Time | Hardware |
| :--- | :--- | :--- | :--- |
| **Protection API (`/protect`)** | Cold Start | ~145.7s | Render Shared CPU |
| **Protection API (`/protect`)** | Warm Server | **~90.6s** | Render Shared CPU |
| **Attacker API (`/clone`)** | Cold Start | ~11.3s | Local RTX GPU via ngrok |
| **Attacker API (`/clone`)** | Warm Server | **~5.1s** | Local RTX GPU via ngrok |

> **Note:** The `~90s` protection latency is due to neural network inference running on a free-tier Render CPU. In a production environment with GPU-backed cloud instances, this would drop to 1-3 seconds.

---

## 🧠 Model Training (Optional)

If you wish to retrain the perturbation generator on your own GPU:

1. Install dependencies and PyTorch with CUDA support.
2. Prepare the dataset (e.g., LibriSpeech).
3. Run the training script:
```bash
cd CloneShield
python train.py --epochs 50 --batch-size 8 --device cuda
```
*For a detailed guide on GPU training and hyperparameters, see `CloneShield/gpu_run_guide.md`.*

---

## 📂 Project Structure Overview
For a detailed breakdown of the files and code interactions, please see [PROJECT_FILE_OVERVIEW.md](PROJECT_FILE_OVERVIEW.md).
