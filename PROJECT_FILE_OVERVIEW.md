# FYP-FakeLess Project: File Overview and Interconnection

This document explains the purpose of each major file in the FYP-FakeLess project and how they work together to accomplish the system's tasks. The project consists of a backend (Python/Flask) for audio protection and a frontend (Flutter mobile app) for user interaction.

---

## 1. Backend (Python)

### Code/defense_trainingv2.py
- **Purpose:** Implements the advanced audio protection pipeline, including HiFiGAN vocoder, adversarial training, and ensemble ASV models.
- **Key Classes:**
  - `HiFiGANVocoder`: Converts mel spectrograms to high-quality audio.
  - `EnsembleASVEmbedder`: Extracts robust speaker embeddings using multiple ASV models.
  - `AdvancedAudioProcessor`: Handles audio loading, saving, and processing.
- **Usage:** Provides core logic for training, protecting, and evaluating audio files. Used by the API to process audio.

### Code/protect_api.py
- **Purpose:** Flask API server that exposes endpoints for audio protection.
- **Key Functions:**
  - `/protect` endpoint: Receives encrypted audio, decrypts it, applies the protection pipeline (via `defense_trainingv2.py`), re-encrypts the result, and returns it.
- **Interconnection:** Imports and uses `AdvancedAudioProtector` and `AdvancedAudioProcessor` from `defense_trainingv2.py` to process audio files. Handles encryption/decryption to ensure secure file transfer with the mobile app.

---

## 2. Frontend (Flutter Mobile App)

### Code/mobile_app_fakeless/lib/main.dart
- **Purpose:** Entry point for the Flutter app. Sets up theming and navigation.
- **Key Classes:**
  - `MyApp`: Root widget, configures light/dark themes.
  - `SplashScreen`: Initial splash animation.
  - `MyHomePage`: Main screen with options to upload or record audio.
- **Interconnection:** Navigates to `AudioPage` for audio recording and protection.

### Code/mobile_app_fakeless/lib/audio_page.dart
- **Purpose:** Main UI for recording, uploading, and protecting audio.
- **Key Features:**
  - Audio recording and playback using `just_audio` and `record` packages.
  - Encrypts audio files before sending to the backend API.
  - Sends encrypted audio to `/protect` endpoint, receives and decrypts protected audio.
  - Plays both original and protected audio for user comparison.
- **Interconnection:**
  - Communicates with the Flask backend (`protect_api.py`) for audio protection.
  - Uses the same encryption key as the backend for secure transfer.

### Code/mobile_app_fakeless/lib/splashscreen.dart
- **Purpose:** Displays an animated splash screen on app launch.
- **Interconnection:** Navigates to `MyHomePage` after animation.

---

## 3. Supporting Files and Folders

- **checkpoints_advanced/**: Stores trained model checkpoints (e.g., universal delta files) used by the protection pipeline.
- **Cloud_files_test/**: Temporary storage for uploaded and processed audio files on the backend.
- **assets/models/**: Contains TFLite models for on-device inference (if implemented in the app).
- **requirements.txt**: Lists Python dependencies for backend.
- **pubspec.yaml** (not shown): Lists Flutter dependencies for the mobile app.

---

## 4. Workflow and Interconnection

1. **User records or uploads audio in the Flutter app (`audio_page.dart`).**
2. **App encrypts the audio and sends it to the Flask backend (`protect_api.py` via `/protect`).**
3. **Backend decrypts the audio, processes it using the advanced protection pipeline (`defense_trainingv2.py`), and re-encrypts the protected audio.**
4. **App receives the protected audio, decrypts it, and allows the user to play and compare.**

Encryption ensures secure transfer of audio files between the app and backend. The backend leverages advanced machine learning models for robust audio protection, while the app provides a user-friendly interface for interaction.

---

## 5. Summary Table

| File/Folder                        | Purpose/Role                                              | Interconnection                        |
|------------------------------------|-----------------------------------------------------------|----------------------------------------|
| defense_trainingv2.py              | Core audio protection logic, ML models                    | Used by protect_api.py                 |
| protect_api.py                     | Flask API for audio protection                            | Calls defense_trainingv2.py classes    |
| main.dart                          | Flutter app entry, theming, navigation                    | Loads splashscreen, home, audio page   |
| audio_page.dart                    | Audio record/upload, API communication, playback          | Calls backend API, handles encryption  |
| splashscreen.dart                  | Animated splash screen                                    | Navigates to home page                 |
| checkpoints_advanced/              | Model checkpoints for protection                          | Used by defense_trainingv2.py          |
| Cloud_files_test/                  | Temp storage for audio files on backend                   | Used by protect_api.py                 |
| assets/models/                     | (Optional) On-device TFLite models                        | For future on-device inference         |
| requirements.txt                   | Python dependencies                                       | For backend setup                      |

---

This document should help new developers understand the structure and data flow of the FYP-FakeLess project.