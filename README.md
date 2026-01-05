# AI-Powered Intelligent Threat Detection and Response System

## Project Structure

```
CIP-PROJ/
├── data/                          # Datasets
│   ├── CIC_IDS_2017.csv
│   ├── NSL_KDD.csv
│   └── UNSW_NB15.csv
├── models/                        # Trained models (created at runtime)
│   ├── network_rf.pkl
│   ├── wids_iforest.pkl
│   ├── malware_lstm.h5
│   └── malware_tokenizer.pkl
├── src/                           # Source code
│   ├── network_traffic/           # Module 1: Network Traffic Classification
│   │   ├── __init__.py
│   │   └── network_classifier.py
│   ├── web_intrusion/             # Module 2: Web Intrusion Detection
│   │   ├── __init__.py
│   │   └── wids.py
│   ├── malware_analysis/          # Module 3: Malware Analysis
│   │   ├── __init__.py
│   │   └── malware_detector.py
│   └── main.py                    # Orchestration pipeline
└── requirements.txt               # Dependencies
```

## Module Responsibilities

### 1. Network Traffic Classification (`src/network_traffic/network_classifier.py`)
- **Purpose**: Binary classification of network traffic (Normal vs Attack)
- **Datasets**: NSL-KDD, CIC-IDS-2017, UNSW-NB15
- **Algorithm**: Random Forest Classifier
- **Key Functions**:
  - `train_network_classifier()`: Main entry point
  - Loads and preprocesses datasets (max 50K rows each)
  - Encodes categorical features
  - Scales numerical features
  - Trains and evaluates model
- **Output**: `models/network_rf.pkl`

### 2. Web Intrusion Detection System (`src/web_intrusion/wids.py`)
- **Purpose**: Anomaly detection for web traffic
- **Data**: Simulated web logs based on CIC-IDS features
- **Algorithm**: Isolation Forest
- **Key Functions**:
  - `train_wids()`: Main entry point
  - Generates synthetic web log data
  - Detects anomalies (expected 20% contamination)
- **Output**: `models/wids_iforest.pkl`

### 3. Malware Analysis (`src/malware_analysis/malware_detector.py`)
- **Purpose**: Binary classification of behavior sequences (Benign vs Malware)
- **Data**: Synthetic malware behavior sequences
- **Algorithm**: LSTM Neural Network (Keras/TensorFlow)
- **Key Functions**:
  - `train_malware_detector()`: Main entry point
  - Generates synthetic behavior data
  - Tokenizes and pads sequences
  - Trains LSTM for 3 epochs
- **Output**: `models/malware_lstm.h5`, `models/malware_tokenizer.pkl`

### 4. Main Pipeline (`src/main.py`)
- **Purpose**: Orchestration only - NO ML logic
- **Functions**:
  - Imports and calls module functions
  - Collects results
  - Runs demonstration predictions
  - Prints final system status

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd src
python main.py
```

## Requirements
- Python 3.9+
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- tensorflow >= 2.8.0
- joblib >= 1.1.0

## Architecture Principles

✓ **Separation of Concerns**: Each module has a single responsibility
✓ **Modularity**: Modules can be run independently for testing
✓ **Clean API**: Each module exposes one main function
✓ **No Globals**: Configuration passed through constructors
✓ **Error Handling**: Graceful degradation if datasets missing
✓ **Scalability**: Easy to add new detection modules

## Expected Output

```
================================================================================
AI-POWERED THREAT DETECTION AND RESPONSE SYSTEM
================================================================================

[Module 1 trains...]
✓ Network Traffic Classification completed successfully

[Module 2 trains...]
✓ Web Intrusion Detection completed successfully

[Module 3 trains...]
✓ Malware Analysis completed successfully

[Demo predictions run...]

================================================================================
FINAL SYSTEM STATUS
================================================================================

✓ Network attack detection: OPERATIONAL
✓ Web anomaly detection: OPERATIONAL
✓ Malware detection: OPERATIONAL

SUCCESS: All 3 modules trained and ready!
```
