# Brugada Syndrome Clinical AI Assistant

## 1. Project Summary
This project is a Streamlit-based clinical decision-support application for Brugada syndrome triage from 12-lead ECG WFDB records.

The current application is designed for high-sensitivity screening support and clinician workflow prioritization. It combines:
- Multi-view deep feature extraction
- Handcrafted statistical and morphology features
- Feature standardization and selection
- A trained ensemble meta-learner for final risk estimation
- Clinician-oriented report generation

This tool supports triage and does not replace physician diagnosis.

## 2. What Is Included
Core application files:
- app.py: Streamlit application entry point and interaction flow
- inference.py: Inference pipeline, model loading, feature extraction, decision logic
- file_utils.py: Upload grouping, file handling, and batch prediction utilities
- ui_components.py: Report components and visualization helpers
- chatbot.py: Optional Gemini-based AI advisor integration
- models/: All deployed model artifacts required for inference
- requirements.txt: Runtime dependency list

Training documentation and notebook:
- A dedicated training notebook exists in your project environment at:
  - ../submit/Brugada_Model_Training_MultiU.ipynb
- This notebook is a full end-to-end training reference for the stacked architecture and artifact generation workflow.
- It is a companion training document; the web app itself uses pre-trained artifacts in models/ for inference.

## 3. System Requirements
- Python 3.12
- Git
- Internet access for first-time dependency installation
- Sufficient memory for TensorFlow model loading and CWT processing

## 4. First-Time Setup After Cloning
Use this once on a new machine.

### 4.1 Windows PowerShell
```powershell
git clone https://github.com/Warren-530/Brugada.git
cd Brugada

python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

If script execution is blocked:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 4.2 macOS or Linux
```bash
git clone https://github.com/Warren-530/Brugada.git
cd Brugada

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 5. Verify Required Inference Artifacts
Before starting the app, confirm these files exist in models/:
- extractor_resnet.keras
- extractor_eegnet.keras
- extractor_bilstm.keras
- extractor_cwt_cnn.keras
- brugada_scaler.pkl
- brugada_selector.pkl
- brugada_meta_learner.pkl

If these files are missing, inference cannot run.

## 6. Run the Application
Start Streamlit:

```powershell
python -m streamlit run app.py
```

Then open the local URL shown in the console, typically http://localhost:8501.

## 7. Daily Run Workflow
After first-time setup, use this standard startup sequence.

### 7.1 Windows PowerShell
```powershell
cd Brugada
.\.venv\Scripts\Activate.ps1
python -m streamlit run app.py
```

### 7.2 macOS or Linux
```bash
cd Brugada
source .venv/bin/activate
python -m streamlit run app.py
```

Optional cache reset if state appears stale:
```powershell
streamlit cache clear
```

## 8. Input Data Requirements
- Upload paired WFDB files: one .hea and one .dat per record
- Base names must match, for example 100.hea and 100.dat
- Data is expected to be compatible with 12-lead ECG assumptions in preprocessing

## 9. New Training Notebook Section
### 9.1 File and Purpose
The companion notebook ../submit/Brugada_Model_Training_MultiU.ipynb documents the training pipeline that produced the deployed artifacts.

Its purpose is to provide:
- Reproducible training logic
- Architecture design details
- Out-of-fold feature extraction strategy
- Ensemble assembly and threshold optimization process
- Artifact generation steps for deployment

### 9.2 What the Notebook Contains
The notebook is structured as phased pipeline blocks:
- Phase 1: Raw data loading and preprocessing
- Phase 2: Statistical, expert morphology, and CWT view generation
- Phase 3: Deep model definitions and strict OOF latent extraction
- Phase 4: Feature stacking, selection, and cost-sensitive meta-learning
- Phase 5: Artifact export for deployment
- Phase 5A and 5B and 6: uncertainty analysis and explainability visualizations

### 9.3 Why It Matters to This Repository
The notebook explains how model artifacts were trained, while the app executes a fixed inference-only version of that logic for fast and stable deployment.

In short:
- Notebook: training and experimentation reference
- Web app: production-style inference and reporting path

## 10. Detailed Model Explanation
This section explains the full modeling strategy in depth, including design rationale, data flow, and deployment implications.

### 10.1 Problem Setting and Design Constraints
Brugada ECG classification in this project is treated as a high-dimensional, low-sample-size problem:
- Raw ECG contains rich temporal structure and lead interactions
- Pathology signatures can be subtle and localized, especially in right precordial leads V1 to V3
- Dataset size constraints increase overfitting risk for deep models
- Pure end-to-end deep classification can become unstable with small cohorts

To address these constraints, the project uses a hybrid architecture:
- Learn multiple latent representations with specialized deep models
- Combine these with handcrafted domain features
- Train a classical ensemble meta-learner on top of stacked features
- Use out-of-fold extraction to reduce leakage in stacking

### 10.2 End-to-End Pipeline View
At a high level, the model pipeline follows this sequence:
1. Load WFDB signal and metadata labels
2. Denoise and length-normalize ECG waveforms
3. Build multiple feature views
4. Extract deep latent features via OOF strategy
5. Concatenate handcrafted and deep features
6. Standardize and select informative dimensions
7. Train cost-sensitive ensemble meta-learner
8. Tune threshold with F2-oriented objective
9. Export scaler, selector, and models for deployment

### 10.3 Phase 1: Signal Preprocessing
The notebook performs the same conceptual preprocessing later mirrored in deployment:
- Band-pass filter using Butterworth design in a clinically relevant frequency range
- Sequence normalization to fixed length 1200 by truncation or zero-padding
- Stratified train-test split before feature extraction to preserve class distribution and reduce leakage

Why this is important:
- Fixed length is required by neural feature extractors
- Filtering suppresses baseline drift and high-frequency artifacts
- Stratification stabilizes minority-class evaluation

### 10.4 Phase 2A: Statistical Feature View
The statistical branch generates 84 handcrafted features:
- 12 leads x 7 statistics per lead
- Typical descriptors: max, min, standard deviation, variance, skewness, kurtosis, RMS

Role in the system:
- Captures broad waveform distribution shape across all leads
- Provides robust baseline descriptors independent of deep latent space
- Improves complementarity for stacked learning

### 10.5 Phase 2B: Expert Morphology Feature View
The morphology branch focuses on V1 to V3 and extracts 9 features total:
- For each of V1, V2, V3:
  - J-point height proxy
  - ST slope proxy
  - Curvature proxy

Rationale:
- Brugada-related signatures are commonly emphasized in right precordial leads
- Explicit morphology features introduce clinically interpretable constraints
- This branch supports both classification and report explanation layers

### 10.6 Phase 2C: Frequency View Through CWT
The frequency branch converts V1 to V3 waveforms to CWT scalograms:
- Continuous Wavelet Transform across scales
- Resized image-like representation per selected lead
- Input to 2D-CNN feature extractor

Rationale:
- Captures time-frequency patterns not explicit in time-only representations
- Sensitive to localized transient energy structures
- Complements pure temporal feature encoders

### 10.7 Phase 3: Deep Feature Extractors
Four deep models are used to learn complementary latent embeddings, each producing 32-dimensional features:
- 1D ResNet-style extractor for morphology-rich local patterns
- EEGNet-style separable-convolution extractor for compact temporal channel interactions
- Attention-BiLSTM extractor for temporal sequence dependencies
- CWT-CNN extractor for image-like time-frequency signatures

Key shared ideas:
- Lead-aware processing via attention-like mechanisms in temporal branches
- Dropout and normalization for regularization
- Latent layer extraction instead of end classifier logits for stacking

### 10.8 Out-of-Fold Latent Feature Generation
The notebook uses strict 5-fold OOF extraction for deep latent vectors:
- For each fold, train on fold-train and predict latent features on fold-validation
- Assemble OOF latent matrix for full training set without same-sample fit leakage
- Predict test latent features per fold and average across folds

Why OOF matters:
- Prevents optimistic leakage in second-level learner
- Reduces meta-learner overconfidence
- More realistic estimate of generalization behavior

### 10.9 Data Augmentation Strategy
During deep feature training, the notebook includes augmentation mechanisms such as:
- Physical signal augmentation (scale and baseline perturbation)
- Mixup to regularize decision boundaries

Expected effect:
- Improve robustness under small-sample variability
- Reduce sensitivity to minor acquisition differences

### 10.10 Feature Stacking Geometry
The full stacked vector is assembled in fixed order:
- Statistical features: 84
- Expert morphology features: 9
- Deep latent features: 32 x 4 = 128
- Total: 221 dimensions

This ordering consistency is important because scaler and selector artifacts are trained on this exact schema.

### 10.11 Standardization and Feature Selection
Post-stacking transformation uses:
- StandardScaler on the full 221-dimensional space
- Model-based feature selection to keep the most informative subset

Benefits:
- Controls feature scale differences across handcrafted and deep embeddings
- Reduces noise and dimensionality burden for final ensemble
- Improves stability in limited-data setting

### 10.12 Meta-Learner Design
The final layer is a cost-sensitive ensemble strategy combining multiple classical learners:
- XGBoost
- LightGBM
- CatBoost
- RBF-kernel SVM

Stacking with a logistic regression final estimator (class balancing enabled) is used to combine these learners.

Why this design works well in this context:
- Different tree and kernel models capture different nonlinear boundaries
- Classical learners can exploit curated latent spaces effectively
- Cost-sensitive weighting helps preserve minority-class recall

### 10.13 Threshold Optimization Strategy
Notebook evaluation performs threshold sweep and optimizes for F2-oriented behavior, prioritizing recall for positive pathological cases.

Important distinction:
- Training notebook demonstrates threshold optimization in one setting
- Deployment in inference.py applies a clinically conservative threshold policy tuned for triage workflow

This is expected and deliberate. Training-time analysis and deployment-time clinical policy can differ.

### 10.14 Exported Artifacts and Their Roles
Artifact export in training produces the deployment assets:
- brugada_scaler.pkl: feature standardization transform
- brugada_selector.pkl: feature selection transform
- brugada_meta_learner.pkl: final classifier
- extractor_resnet.keras: deep latent model 1
- extractor_eegnet.keras: deep latent model 2
- extractor_bilstm.keras: deep latent model 3
- extractor_cwt_cnn.keras: deep latent model 4

The app loads these artifacts during startup and runs deterministic inference with no retraining in production flow.

### 10.15 Inference-Time Reporting and Explainability
The deployed app augments raw prediction with clinician-facing context:
- Risk probability and threshold-relative interpretation
- Morphology evidence summaries focused on V1 to V3
- Decision stability and borderline handling logic
- Recommendation tiers and next-action guidance

This layer is not only for visualization; it is designed to support triage prioritization and safer handoff to physician review.

### 10.16 Why This Hybrid Architecture Is Appropriate
Compared with single-model alternatives, this architecture provides:
- Better representation diversity
- Lower leakage risk in stacked training
- More resilient behavior in HDLSS constraints
- Better practical interpretability through explicit morphology channels
- Flexible deployment with fixed reusable artifacts

### 10.17 Limitations and Responsible Use
Even with strong engineering choices, limitations remain:
- Dataset and cohort shift can reduce external generalization
- Morphology proxies are heuristic and not equivalent to full electrophysiology interpretation
- Explainability summaries are supportive signals, not causal proof
- Clinical context, physician judgment, and additional tests remain mandatory

## 11. Optional Gemini API Configuration
The app can run without chatbot features. Configure only if needed.

Option 1: .streamlit/secrets.toml
```toml
GEMINI_API_KEY = "your-api-key"
```

Option 2: environment variable in PowerShell
```powershell
$env:GEMINI_API_KEY = "your-api-key"
```

## 12. Troubleshooting
### 12.1 Missing Python Modules
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 12.2 TensorFlow or Keras Load Errors
```powershell
python -c "import tensorflow as tf; print(tf.__version__)"
pip install -r requirements.txt --force-reinstall
```

### 12.3 App Starts but Prediction Fails
Check the following:
- All files in models/ exist
- Uploaded records include both .hea and .dat files
- Filenames are correctly paired

## 13. Clinical Disclaimer
This software is intended for decision support and triage workflow assistance. It does not provide definitive diagnosis. Final interpretation must be performed by qualified clinicians within appropriate clinical context.