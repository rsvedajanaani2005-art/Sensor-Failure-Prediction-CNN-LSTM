# Sensor Failure Prediction in a Binary Distillation Column using CNN-LSTM

A deep learning–based approach for multi-class sensor fault diagnosis in a binary distillation column using a hybrid CNN-LSTM architecture.

The model predicts common sensor faults such as **drift**, **offset**, and **noise**, achieving over **95% test accuracy**.

---

## Problem Statement

In chemical process industries, faulty sensors can lead to:

- Poor process control  
- Safety hazards  
- Reduced product quality  
- Economic losses  

Manual monitoring is not scalable or real-time.  

This project automates sensor failure prediction using multivariate time-series data from a binary distillation column.

---

## Model Summary

| Feature | Details |
|----------|----------|
| Model Type | CNN-LSTM (Hybrid Architecture) |
| Input | Multivariate time-series data (14 sensors) |
| Output Classes | Healthy, Drift, Noise, Offset |
| Windowing Strategy | Sliding window (100 timesteps) |
| Lookahead | 20 timesteps |
| Framework | PyTorch |
| Test Accuracy | 95.60% |

---

## Dataset Information

The original industrial dataset is confidential and cannot be shared.

The dataset contains temperature readings from 14 sensors in a binary distillation column.

To simulate realistic conditions, synthetic faults were injected into healthy data:

- Drift → Gradual signal deviation over time  
- Offset → Sudden constant shift in signal  
- Noise → Random perturbations added to signal  

This allowed controlled training and evaluation.

---

## Methodology

### 1. Data Preprocessing

- Missing values filled using median
- Normalization of all sensor channels
- Creation of sliding windows for time-series sequences
- Stratified 80:20 train-test split

---

### 2. Model Architecture

- 1D Convolutional layer for feature extraction  
- Bidirectional LSTM for temporal modeling  
- Fully connected layer for classification  

Loss Function: CrossEntropyLoss  
Optimizer: Adam  

---

## Results

### Confusion Matrix

![Confusion Matrix](outputs/confusion_matrix.png)

---

### Classification Report

| Class     | Precision | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|----------|
| Healthy   | 0.90 | 0.90 | 0.90 | 21 |
| Drift     | 0.96 | 0.93 | 0.95 | 46 |
| Noise     | 1.00 | 0.96 | 0.98 | 46 |
| Offset    | 0.94 | 1.00 | 0.97 | 46 |
| **Accuracy** |  |  | **0.96** | **159** |
| **Macro Avg** | 0.95 | 0.95 | 0.95 | 159 |
| **Weighted Avg** | 0.96 | 0.96 | 0.96 | 159 |

---

## Tech Stack

- Python  
- PyTorch  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  

---

## How to Run

1. Clone the repository:
```
git clone https://github.com/your-username/Sensor-Failure-Prediction-CNN-LSTM.git
cd sensor-failure-prediction-cnn-lstm
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Run training script:
```
python train.py
```

---

## Future Work

- Real-time deployment for industrial monitoring  
- Multi-fault simultaneous detection  
- Integration with IIoT platforms  
- Validation on real-world faulty sensor data  
