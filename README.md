# AG-BiLSTM: Attention-Gated Bidirectional LSTM for Collision-Risk Prediction

This repository implements an **Attention-Gated Bidirectional LSTM (AG-BiLSTM)** framework for **rear-end collision-risk prediction** using high-resolution vehicle trajectory data.  
The model is designed to achieve a **high efficiency–performance trade-off**, enabling accurate prediction of rare, safety-critical events while remaining computationally lightweight for real-time deployment.

---

## Key Features

- Vehicle-wise data splitting to guarantee zero data leakage  
- Causal temporal smoothing of kinematic features  
- Train-only label thresholding for unbiased evaluation  
- Future-aware sequence labeling (prediction horizon support)  
- Attention-gated temporal aggregation for enhanced interpretability  
- Comprehensive evaluation (ROC-AUC, confusion matrix, class-wise metrics)  
- High-quality visual diagnostics (600 DPI, IEEE-ready)

---

## Project Structure

AG-BiLSTM/
│
├── data/
│   └── NGSIM.csv                  # Raw trajectory data (not included)
│
├── figures/
│   ├── training_curves.png
│   ├── roc_curves.png
│   ├── class_distribution.png
│   └── correlation_matrix.png
│
├── ag_bilstm.py                   # Main implementation
├── requirements.txt               # Python dependencies
└── README.md                      # Project documentation

---

## Model Overview

The proposed AG-BiLSTM architecture consists of:
1. Stacked Bidirectional LSTM layers for forward–backward temporal modeling  
2. An attention-gating mechanism to emphasize critical time steps  
3. Lightweight fully connected layers for multi-class collision-risk prediction  

The model predicts three discrete risk states:
- Safe (0)
- Warning (1)
- Danger (2)

---

## Input Features

Each input sequence consists of **10 time steps**, sampled at **10 Hz**, with the following features per frame:

- Vehicle velocity (v_Vel)
- Space headway
- Time headway
- Vehicle length
- Vehicle width
- Lane ID

---

## Data Preprocessing Pipeline

1. Raw data loading  
2. Passenger vehicle filtering  
3. Vehicle-wise train/validation/test split (70/15/15)  
4. Causal rolling-window smoothing (per vehicle)  
5. Inverse time-gap computation (label-only feature)  
6. Train-only percentile-based risk thresholding  
7. Standard scaling (train-fit, global-transform)  
8. Sliding-window sequence generation  
9. Future-risk label assignment  

No resampling, oversampling, or class balancing is applied to validation or test sets.

---

## Training

Run the model using:

python ag_bilstm.py

Default settings:
- Epochs: 50  
- Batch size: 64  
- Optimizer: Adam (1e-4)  
- Early stopping and learning-rate reduction

---

## Evaluation Metrics

- Overall accuracy  
- Class-wise precision, recall, and F1-score  
- Macro and weighted averages  
- One-vs-rest ROC-AUC curves  
- Confusion matrix visualization

---

## Reproducibility

- Fixed random seeds (NumPy, TensorFlow)  
- Deterministic vehicle-wise splitting  
- Train-only normalization and labeling

---

## Requirements

python >= 3.8  
numpy  
pandas  
scikit-learn  
matplotlib  
seaborn  
tensorflow >= 2.x  

Install dependencies with:

pip install -r requirements.txt

---

## License

This project is released for research and academic use only.  
Please ensure compliance with dataset licensing terms (e.g., NGSIM).
