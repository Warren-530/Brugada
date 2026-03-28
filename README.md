# Brugada Syndrome Clinical AI Assistant

## Overview
This repository contains a Streamlit-based clinical decision-support application for Brugada syndrome risk triage from 12-lead ECG WFDB records.

The deployed pipeline combines:
- Four deep feature extractors (ResNet, EEGNet, BiLSTM, CWT-CNN)
- Handcrafted statistical and morphology features
- A trained meta-learner for final risk estimation
- Clinician-facing reporting and optional AI advisor support

This software is intended for triage support and workflow prioritization. It is not a standalone diagnostic tool.

## Repository Structure
- app.py: Streamlit web application entry point
- inference.py: End-to-end inference pipeline and decision logic
- file_utils.py: Upload processing and batch utilities
- ui_components.py: Report rendering components
- chatbot.py: Optional Gemini-based AI clinical advisor
- models/: Trained model artifacts
- requirements.txt: Python dependencies

## Requirements
- Python 3.12
- Git
- Internet access for initial dependency installation

## First-Time Local Setup (After Cloning)
Use these steps the first time you pull this project to your machine.

### Windows PowerShell
```powershell
git clone https://github.com/Warren-530/Brugada.git
cd Brugada

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If activation is blocked by PowerShell policy, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS/Linux
```bash
git clone https://github.com/Warren-530/Brugada.git
cd Brugada

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Verify Required Model Artifacts
Before running the app, confirm these files exist in models/:
- extractor_resnet.keras
- extractor_eegnet.keras
- extractor_bilstm.keras
- extractor_cwt_cnn.keras
- brugada_scaler.pkl
- brugada_selector.pkl
- brugada_meta_learner.pkl

## Optional: Configure Gemini API Key (AI Advisor)
The app can run without an API key. Configure this only if you want chatbot support.

Option 1: .streamlit/secrets.toml
```toml
GEMINI_API_KEY = "your-api-key"
```

Option 2: Environment variable (PowerShell)
```powershell
$env:GEMINI_API_KEY = "your-api-key"
```

## Run the Project
Start the web application:

```powershell
python -m streamlit run app.py
```

Open the local URL printed by Streamlit (usually http://localhost:8501).

## Daily Run (After Initial Setup)
For all future runs, use only:

### Windows PowerShell
```powershell
cd Brugada
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

### macOS/Linux
```bash
cd Brugada
source .venv/bin/activate
python -m streamlit run app.py
```

Optional cache reset if UI state appears stale:
```powershell
streamlit cache clear
```

## Input Requirements
- Upload matched WFDB file pairs: .hea and .dat
- Base filenames must match (example: 100.hea and 100.dat)
- Expected signal format: 12-lead ECG

## Troubleshooting
### Missing Python modules
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### TensorFlow / Keras model loading issues
Check TensorFlow version:
```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
```
Then reinstall dependencies if needed:
```powershell
pip install -r requirements.txt --force-reinstall
```

### App starts but diagnosis fails
- Confirm all artifacts in models/ are present
- Confirm each uploaded record includes both .hea and .dat
- Confirm filenames are correctly paired

## Clinical Disclaimer
This software supports clinical workflow and triage decisions. Final interpretation and diagnosis must be made by qualified clinicians.