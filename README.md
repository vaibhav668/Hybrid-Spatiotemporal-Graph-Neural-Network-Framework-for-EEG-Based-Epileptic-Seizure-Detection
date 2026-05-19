#  GNN-LSTM Hybrid Architecture for EEG-Based Epileptic Seizure Detection

> **Intern:** Vaibhav Pokhriyal | B.Tech CSE (AI & ML), GBPIET Pauri Garhwal  
> **Supervisor:** Dr. Tapan Kumar Gandhi (PhD, FNAE, FNASc, SMIEEE) — Professor, Dept. of Electrical Engineering, IIT Delhi  
> **Duration:** 12 February 2026 – 12 May 2026 (3 Months)  
> **Domain:** Biomedical Signal Processing · Deep Learning · Graph Neural Networks

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Motivation](#-motivation)
- [Dataset](#-dataset)
- [Pipeline Architecture](#-pipeline-architecture)
- [Model Architecture](#-model-architecture)
- [Training Strategy](#-training-strategy)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Challenges & Solutions](#-challenges--solutions)
- [Limitations & Future Work](#-limitations--future-work)
- [References](#-references)

---

##  Overview

Epilepsy affects approximately **50 million people worldwide**. Automated, accurate detection of epileptic seizures from EEG signals is a critical clinical priority — particularly for the significant proportion of patients who remain refractory to antiepileptic medication.

This project presents a **novel GNN-LSTM hybrid deep learning pipeline** for automated EEG-based seizure detection. The architecture combines:

- A **Graph Attention Network (GAT)** encoder that models inter-electrode spatial relationships as graph structures
- A **Bidirectional LSTM (BiLSTM)** network that captures temporal dependencies across sequential EEG windows

The full pipeline is implemented and evaluated on the **CHB-MIT Scalp EEG Database** with rigorous patient-wise cross-validation.

---

##  Motivation

The fundamental insight driving this work: **epileptic seizures are not spatially localised events** — they propagate through interconnected brain regions via synchronised neural pathways. The spatial propagation pattern is as diagnostically relevant as the temporal waveform shape.

| Approach | Spatial Modelling | Temporal Modelling | Limitation |
|---|---|---|---|
| SVM / Random Forest | ❌ | ❌ | No deep feature learning |
| CNN on spectrograms | ⚠️ Partial | ❌ | Treats electrodes as independent channels |
| LSTM / GRU | ❌ | ✅ | Blind to inter-electrode topology |
| **GNN-LSTM (Ours)** | ✅ GAT | ✅ BiLSTM | — |

GNNs provide a natural framework: each electrode becomes a graph node, pairwise functional connectivity (Pearson correlation) becomes an edge, and the GNN learns to aggregate information respecting the graph topology. The LSTM then processes sequences of graph-level embeddings to model how spatial connectivity evolves over time.

---

##  Dataset

**CHB-MIT Scalp EEG Database** — the primary benchmark for seizure detection research.

| Property | Details |
|---|---|
| Number of Patients | 22 (chb01–chb24, excluding chb12 & chb16) |
| Sampling Rate | 256 Hz (uniform) |
| Electrode System | International 10-20, bipolar montage |
| File Format | European Data Format (.edf) |
| Total Seizures | ~182 annotated seizures |
| Recording Duration | 24+ hours per patient |
| Class Imbalance | Seizure windows < 1–2% of total data |

The extreme sparsity of seizure events — less than 1–2% of total recordings — is the central challenge of this project. A naive classifier predicting "non-seizure" for every input achieves >98% accuracy but **0% clinical utility**.

---

##  Pipeline Architecture

The complete pipeline consists of **10 sequential phases** from raw EDF ingestion to 5-fold cross-validated model evaluation.

```
Phase 1  →  Environment Setup
            Imports · Seeds · CONFIG · Device

Phase 2  →  Data Loading & Channel Standardisation
            EDF → MNE → 18-ch Bipolar Montage (zero-fill missing)

Phase 3  →  Windowing & Labelling
            4s / 50% overlap · Seizure labelling
            Near-seizure filter (±60s) · Baseline cap 200/300

Phase 4  →  Preprocessing
            Bandpass 0.5–40 Hz · Notch 60 Hz · Per-channel z-score normalisation

Phase 5  →  Feature Extraction & Graph Construction
            11-D node features · Top-K=6 edges (Pearson correlation)

Phase 6  →  Sequence Assembly & Class Balancing
            seq_len=15 graphs · Patient-tagged IDs
            Per-patient cap 400 · 2:1 non-seizure ratio

Phase 7  →  Train / Val / Test Split
            Patient-wise stratified (no leakage)

Phase 8  →  GNN-LSTM Model
            GATEncoder → BiLSTM → Classifier

Phase 9  →  Training Strategy
            FocalLoss · AdamW · Cosine Annealing · Grad clip · Early stopping

Phase 10 →  5-Fold GroupKFold Cross-Validation
            Patient-disjoint folds · Per-fold fresh model · Threshold search
```

### Key Configuration

```python
CONFIG = {
    "sampling_rate":        256,     # Hz — CHB-MIT hardware rate
    "window_size_sec":      4,       # seconds per EEG window
    "overlap":              0.5,     # 50% sliding window overlap
    "sequence_length":      15,      # windows per LSTM sequence (~37.5s)
    "correlation_threshold":0.6,     # edge threshold (superseded by Top-K)
    "batch_size":           16,
}
# Derived
window_samples = 1024   # 256 × 4
step_size      = 512    # 1024 × (1 − 0.5)
```

### Node Feature Vector (11-Dimensional)

Each EEG electrode node is described by 11 seizure-sensitive features:

| Feature | Type | Seizure Relevance |
|---|---|---|
| Standard Deviation | Time-domain | Amplitude proxy; elevated during ictal activity |
| Line Length | Time-domain | Signal complexity; markedly elevated at onset |
| RMS Energy | Time-domain | Elevated broadband power during seizure |
| Delta Power (1–4 Hz) | Spectral | Post-ictal diffuse slowing |
| Theta Power (4–8 Hz) | Spectral | Mesial temporal involvement in TLE |
| Alpha Power (8–13 Hz) | Spectral | Suppressed during ictal activity |
| Beta Power (13–30 Hz) | Spectral | Fast oscillations at seizure initiation |
| Gamma Power (30–40 Hz) | Spectral | High-frequency activity in onset zones |
| Spectral Entropy | Information | Low during seizure (energy concentration) |
| Hjorth Mobility | Complexity | Mean frequency proxy; increases during seizure |
| Hjorth Complexity | Complexity | Changes characteristically in ictal state |

---

##  Model Architecture

### Forward Pass Overview

```
Input: Sequence of 15 PyG graphs (each = one 4-second EEG window)
         ↓
GATEncoder × 15        (shared weights)
  Input Projection     Linear(11 → 256) + ReLU
  GAT Layer 1          GATConv(256 → 256, 4 heads) → 1024-dim + GroupNorm(4,1024) + ReLU + Dropout(0.3)
  GAT Layer 2          GATConv(1024 → 256, 1 head) → GroupNorm(1,256)
  Global Pooling       GlobalMeanPool ∥ GlobalMaxPool → [512]
         ↓
Embed Projection       Linear(512 → 512) + LayerNorm + ReLU
         ↓
BiLSTM                 2-layer Bidirectional LSTM (hidden=256)
                       Input [15 × 512] → fwd+bwd last hidden [512]
         ↓
Classifier             Linear(512 → 128) + LayerNorm + ReLU + Dropout(0.2)
                       → Linear(128 → 1) → Sigmoid
         ↓
Output: P(seizure) ∈ [0, 1]   (threshold optimised per epoch)
```

### Component Summary

| Component | Configuration | Notes |
|---|---|---|
| Input Projection | Linear(11→256) + ReLU | Node feature lifting |
| GAT Layer 1 | hidden=256, 4 heads, concat → 1024-dim | GroupNorm aligned with heads |
| GAT Layer 2 | 1024→256, 1 head, no concat | Head collapse, GroupNorm |
| Global Pooling | Mean + Max concat | → 512-dim graph embedding |
| Embed Projection | Linear(512→512), LayerNorm | Embedding refinement |
| BiLSTM | hidden=256, 2 layers, bidirectional | → 512-dim temporal feature |
| Classifier | 512→128→1, LayerNorm, Dropout(0.2) | Binary output with sigmoid |

### Design Decisions

- **GroupNorm** instead of BatchNorm — works correctly for any batch size including 1 (critical during single-sample evaluation)
- **Dual pooling** (Mean + Max) — captures both average electrode network state and most extreme inter-electrode activations
- **Top-K=6 graph construction** — guarantees no isolated nodes regardless of correlation level; prevents GAT message-passing degradation
- **BiLSTM bidirectionality** — forward LSTM tracks onset dynamics; backward LSTM incorporates future context

---

##  Training Strategy

| Component | Setting | Rationale |
|---|---|---|
| Loss Function | Focal Loss (α=0.75, γ=2.0) | Down-weights easy negatives; focuses on hard seizure examples |
| Optimiser | AdamW (lr=3e-4, wd=1e-4) | Decoupled weight decay; true L2 regularisation |
| LR Scheduler | CosineAnnealingWarmRestarts (T₀=10, T_mult=2) | Periodic restarts to escape local minima |
| LR Warmup | LinearLR (start_factor=0.1, 5 epochs) | Prevents destabilising large early gradient steps |
| Gradient Clipping | max_norm=1.0 | Prevents exploding gradients in BiLSTM |
| Gradient Accumulation | 16 steps | Simulates larger effective batch size |
| Dynamic Threshold | Grid search [0.10, 0.95] per epoch | Maximises F1 subject to precision>0.5 and recall>0.3 |
| Early Stopping | patience=5, max 50 epochs | Restores best validation F1 checkpoint |

### Multi-Stage Class Imbalance Mitigation

```
Level 1 — Near-Seizure Proximity Filter
          Retains non-seizure windows within ±60s of any seizure boundary
          → focuses training on clinically critical pre/post-ictal regions

Level 2 — Forced Baseline Inclusion
          Up to 300 random background non-seizure windows per patient
          → maintains diversity in background EEG patterns

Level 3 — Non-Seizure File Capping
          Max 200 windows from seizure-free files
          → prevents seizure-free patients from dominating

Level 4 — Per-Patient Sequence Balancing
          2:1 non-seizure:seizure ratio, capped at 200 per class per patient
          → prevents high-seizure patients from dominating

Level 5 — Focal Loss
          α=0.75, γ=2.0 — minority class weighting + hard example mining
```

---

##  Results

Evaluated under **patient-wise 5-fold GroupKFold cross-validation** — no patient appears in both training and test folds.

### Classification Performance

| Split | Accuracy | Macro F1 | Seizure Recall | Seizure F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Train** | 0.91 | 0.91 | 0.96 | 0.88 | 0.985 |
| **Test** | 0.84 | 0.82 | 0.78 | 0.76 | 0.914 |

### Test Confusion Matrix

```
                 Predicted
                 Non-seizure   Seizure
Actual  Non-sz      960          150
        Seizure     122          433
```

- **Seizure Recall (0.78):** 78% of true seizures correctly detected
- **Test ROC-AUC (0.914):** Exceeds the 0.85 threshold considered strong for patient-wise EEG evaluation
- **Train→Test gap:** Moderate, expected given severe class imbalance and patient-specific EEG variability

### Training Dynamics

- Training loss: 0.13 → 0.015 (stable convergence)
- Validation F1: rapid improvement in early epochs, stable convergence after ~epoch 15
- No significant divergence between train/val loss, confirming effective regularisation

---

##  Installation

### Prerequisites

- Python 3.10+
- CUDA-enabled GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/gnn-lstm-seizure-detection.git
cd gnn-lstm-seizure-detection

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
torch>=2.0.0
torch-geometric
mne
numpy
pandas
scipy
scikit-learn
networkx
tqdm
matplotlib
seaborn
pickle5
```

> **Note:** Install PyTorch Geometric following the [official guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) matching your CUDA version.

---

##  Usage

### 1. Dataset Download

Download the **CHB-MIT Scalp EEG Database** from PhysioNet:

```bash
# Using wget
wget -r -N -c -np https://physionet.org/files/chbmit/1.0.0/

# Or via PhysioNet web interface
# https://physionet.org/content/chbmit/1.0.0/
```

### 2. Configure Dataset Path

In the notebook (Phase 1.6), update the dataset path:

```python
DATASET_PATH = "/path/to/chb-mit-scalp-eeg-database-1.0.0"
```

### 3. Run the Pipeline

Open and run the notebook sequentially:

```bash
jupyter notebook gnn_lstm_combined_vaibhav.ipynb
```

Or execute all phases in order — each phase builds on the outputs of the previous.

### 4. Pipeline Phases Overview

```python
# Phase 1  — Environment, seeds, CONFIG
# Phase 2  — Load EDF files, standardise to 18-channel bipolar montage
# Phase 3  — Window (4s, 50% overlap), label, near-seizure filter
# Phase 4  — Bandpass + notch filter, z-score normalise
# Phase 5  — Extract 11-D node features, construct Top-K=6 graphs
# Phase 6  — Assemble sequences (len=15), per-patient balancing
# Phase 7  — Patient-wise train/val/test split
# Phase 8  — Define GNN_LSTM_Model
# Phase 9  — Train with FocalLoss, AdamW, dynamic threshold
# Phase 10 — 5-fold GroupKFold cross-validation
```

---

##  Project Structure

```
gnn-lstm-seizure-detection/
│
├── gnn_lstm_combined_vaibhav.ipynb   # Main pipeline notebook (all 10 phases)
│
├── data/
│   └── chb-mit/                      # CHB-MIT dataset (not included — download separately)
│       ├── chb01/
│       ├── chb02/
│       └── ...
│
├── cache/
│   └── *.pkl                         # Preprocessed graph sequences (auto-generated)
│
├── results/
│   ├── confusion_matrices/           # Per-fold confusion matrix plots
│   ├── roc_curves/                   # Train/test ROC curve plots
│   └── training_curves/              # Loss & F1 curves
│
├── requirements.txt
├── README.md
└── InternReport_Vaibhav.pdf          # Full technical internship report
```

---

##  Challenges & Solutions

### 1. Severe Class Imbalance (~1% seizure windows)
**Solution:** Five-level mitigation hierarchy — proximity filtering, forced baseline inclusion, file capping, per-patient balancing, and Focal Loss.

### 2. Inconsistent EEG Channel Configurations Across Patients
**Solution:** Fixed 18-channel bipolar montage enforced for all patients. Missing channels handled via adjacent fallback or cross-channel mean substitution.

### 3. Graph Connectivity / Isolated Nodes
**Solution:** Switched from hard correlation threshold to **Top-K=6 neighbourhood** strategy, guaranteeing every node has at least 6 edges regardless of signal correlation.

### 4. BatchNorm Instability at Small Batch Sizes
**Solution:** Replaced all BatchNorm layers with **GroupNorm** (4 groups, aligned with attention heads), which is invariant to batch size.

### 5. Summary File Parsing Failures
**Solution:** Implemented regex-based annotation parser handling variable whitespace, multi-digit seizure indices, and irregular annotation structures. Malformed lines are logged and skipped safely.

### 6. Memory / Computational Constraints
**Solution:** Patient-wise processing, non-seizure capping, per-patient sequence caps (MAX=400), and **pickle-based dataset persistence** for one-time preprocessing.

---

##  Limitations & Future Work

| Limitation | Proposed Direction |
|---|---|
| Evaluated only on CHB-MIT | Validate on TUH EEG Corpus, Bonn dataset, clinical recordings |
| Binary classification only | Extend to multi-class: pre-ictal / ictal / post-ictal / interictal |
| Pearson correlation edges (linear only) | Explore PLV, Granger Causality, Transfer Entropy for richer graphs |
| Fixed sequence length (15 windows) | Adaptive temporal modelling for variable seizure durations |
| Detection only (post-onset) | Develop pre-ictal prediction capability |
| Offline research pipeline | Optimise for real-time clinical deployment |
| EEG artefact sensitivity | Dedicated artefact rejection module (ICA, template matching) |

---

##  Acknowledgements

This work was conducted at the **Neuro Computing Laboratory, IIT Delhi** under the supervision of **Dr. Tapan Kumar Gandhi** (Professor, Department of Electrical Engineering, IIT Delhi). The internship was completed in partial fulfilment of the research requirements for the B.Tech programme at **GBPIET, Pauri Garhwal**.

The **CHB-MIT Scalp EEG Database** is publicly available via [PhysioNet](https://physionet.org/content/chbmit/1.0.0/).

