# Deep Learning Network Intrusion Detection System (MLP)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Deep Learning](https://img.shields.io/badge/Model-Multilayer%20Perceptron%20(MLP)-orange.svg)](https://scikit-learn.org/)
[![Explainability](https://img.shields.io/badge/XAI-SHAP%20Interpretability-green.svg)](https://shap.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Dataset-UNSW--NB15-purple.svg)](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

A deep learning framework for robust network intrusion detection, feature-tier benchmarking, and open-set zero-day threat recognition on high-throughput network traffic.

---

## Key Highlights & Technical Contributions
- **Deep MLP Architecture**: Implemented a 3-layer neural network `(256 -> 128 -> 64)` with ReLU activations, Adam optimization, adaptive learning rates, and L2 regularization to classify multi-class cyber attacks on the 250k+ record UNSW-NB15 dataset.
- **Severe Class Imbalance Mitigation**: Resolved extreme attack distribution skew across 9 intrusion categories via custom probability-based sample weight resampling.
- **5-Tier Dimensionality Reduction Benchmark**: Systematically evaluated performance across 48, 40, 30, 20, and 15 feature subsets. The **30-feature tier achieved peak multi-class accuracy (70.2%) and Macro-F1 (0.665)** while slashing feature dimensionality by 37.5%.
- **Zero-Day Attack Recognition (Open-Set Detection)**: Implemented prediction confidence threshold sweeps to detect unknown/zero-day attacks (*Worms* and *Shellcode*) without model retraining, reaching **0.814 AUROC**.
- **Model Calibration & Reliability**: Assessed prediction confidence using Expected Calibration Error (ECE ~0.13–0.17) to prevent overconfident classification in SOC environments.
- **Explainable AI (XAI)**: Deployed a SHAP bootstrap interpretability pipeline to extract global feature rankings and verify explanation stability across dimensions.

---

## Experimental Results

### Feature-Tier Benchmarking (UNSW-NB15)
| Feature Set | Features | Accuracy | Precision | Recall | Macro F1 | Attack Detection Rate | False Positive Rate | ROC-AUC | ECE | Unknown (Zero-Day) AUROC |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Full Tier** | 48 | 67.52% | 0.6157 | 0.7081 | 0.6334 | 98.28% | 35.82% | 0.9541 | 0.1735 | 0.7899 |
| **Tier 1** | 40 | 67.50% | 0.6283 | 0.7183 | 0.6383 | 98.96% | 36.95% | **0.9605** | 0.1620 | **0.8136** |
| **Tier 2** | 30 | **70.16%** | **0.6605** | 0.7267 | **0.6646** | 95.93% | **33.40%** | 0.9470 | 0.1623 | 0.7891 |
| **Tier 3** | 20 | 68.28% | 0.6697 | 0.7443 | 0.6695 | 97.96% | 36.91% | 0.9523 | 0.1442 | 0.7417 |
| **Tier 4** | 15 | 65.91% | 0.6482 | **0.7422** | 0.6510 | **99.67%** | 41.27% | 0.9513 | **0.1314** | 0.6991 |

---

## Project Architecture
```text
Deep-Learning-NIDS-MLP/
├── README.md                          # Project documentation and benchmark report
├── requirements.txt                   # Dependency specifications
├── config.yaml                        # Architecture and experiment configurations
├── src/
│   ├── model.py                       # Deep MLP architecture with custom resampling
│   ├── preprocessing.py               # Feature selection, scaling, and data loading
│   ├── train.py                       # End-to-end model training workflow
│   ├── evaluate_tiers.py              # 5-tier dimensionality benchmark runner
│   ├── openset.py                     # Zero-day detection & threshold sweep
│   └── shap_explain.py                # SHAP attribution & feature importance
└── results/
    ├── feature_tier_metrics.csv       # Multi-tier evaluation metrics
    ├── shap_importance_48f.csv        # Global SHAP feature ranking dataset
    ├── open_set_sweep_*.csv           # Open-set sweep metrics (15f, 20f, 30f, 40f, 48f)
    └── plots/                         # ROC curves, confusion matrices, and sweep plots
```

---

##  Quickstart

### 1. Installation
```bash
git clone https://github.com/AnshuRathore13/Deep-Learning-NIDS-MLP.git
cd Deep-Learning-NIDS-MLP
pip install -r requirements.txt
```

### 2. Train MLP
```bash
python src/train.py --config config.yaml --tier 40
```

### 3. Run Open-Set Zero-Day Detection
```bash
python src/openset.py --config config.yaml
```

### 4. Run SHAP Interpretability Analysis
```bash
python src/shap_explain.py --config config.yaml
```
