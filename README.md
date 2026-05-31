# PIWON: Physics-Informed Washoff Neural Network

A hybrid deep learning model for predicting urban stormwater total suspended solids (TSS) , combining physics-based washoff equations with data-driven neural networks.

## Overview

This repository contains the source code for the paper:

> **[Non-Point Source Pollution Prediction and Dynamics Simulation in Urban Runoff: A Physics-Informed Neural Network Approach]**  
> *[Sijie TANG, Jiping JIANG, Shuo WANG, Yi ZHENG, Dragan SAVIC, and Aijie ZHANG]*  
> *[Water Research], [2026]*

The model (PIWON) integrates:
- **GRU-based Encoder–Decoder** for rainfall-driven runoff estimation
- **Physics-based Washoff Module** governed by the exponential washoff equation
- **BuildUp Module** to estimate initial pollutant load before each rain event
- **WashOffPara Module** to predict spatially varying washoff parameters from catchment features

## Repository Structure

```
.
├── PIWON.py               # Core model architecture
├── data_processor.py      # Data loading, transformation, and preprocessing utilities
├── main.ipynb             # Training, evaluation, and sensitivity analysis
├── figures.ipynb          # Figure generation for the paper
├── data/
│   ├── NSQD_data.csv      # National Stormwater Quality Database (NSQD) dataset
│   ├── guangming.csv      # Guangming watershed validation data
│   ├── longgang.csv       # Longgang watershed validation data
│   ├── train_set.csv      # Pre-split training set
│   ├── test_set.csv        # Pre-split test set
│   ├── my_transformer     # Fitted AutoML feature pipeline (pickle)
│   └── obj_dict           # Category label mapping (pickle)
├── checkpoint/
│   ├── v2_seed_0_Oct19.pkl  # Trained model (seed 0)
│   ├── v2_seed_1_Oct19.pkl  # Trained model (seed 1)
│   ├── v2_seed_2_Oct19.pkl  # Trained model (seed 2)
│   ├── v2_seed_3_Oct19.pkl  # Trained model (seed 3)
│   └── v2_seed_4_Oct19.pkl  # Trained model (seed 4)
├── shap_ffi/              # SHAP values for FFI prediction
├── shap_tss/              # SHAP values for TSS prediction
└── log/                   # Training logs
```

## Requirements

See `requirements.txt` for the full dependency list.

Key dependencies:
- Python >= 3.8
- PyTorch >= 1.12
- AutoGluon >= 0.7
- scikit-learn
- SHAP
- pandas, numpy, matplotlib, seaborn

## Installation

```bash
pip install -r requirements.txt
```

> **Note:** AutoGluon may require additional setup. See the [AutoGluon documentation](https://auto.gluon.ai/) for platform-specific instructions.

## Usage

### 1. Data Preprocessing

Run the data preprocessing cells at the top of `main.ipynb`, or use `data_processor.py` directly:

```python
import data_processor as dp

data = dp.raw_data('data/NSQD_data.csv')
converted_data, obj_dict = dp.data_transform(data)
```

### 2. Model Training

Execute the **Training** section in `main.ipynb`. Seeds 0–4 are used for ensemble runs to assess reproducibility.

```python
from PIWON import HybridModel

net = HybridModel(input_size=18, hidden_size=64)
train_losses, valid_losses, test_losses, best_state = train(
    net, train_set, valid_set, test_set,
    batch_size=32, num_epochs=1000
)
```

### 3. Evaluation & Figures

Run `figures.ipynb` to reproduce all figures in the paper.

## Model Architecture

```
Input (catchment features + synthetic rainfall time series)
    │
    ├── EncoderRNN (GRU)  →  Latent state
    │
    ├── DecoderRNN (GRU)  →  Runoff time series (q_t)
    │
    ├── BuildUp (MLP)     →  Initial pollutant mass (M_0)
    │
    ├── WashOffPara (MLP) →  Washoff parameters (k, β)
    │
    └── WashOff (Physics) →  Pollutant mass over time
                              TSS = (M_0 − M_T) / Total_Runoff
```
