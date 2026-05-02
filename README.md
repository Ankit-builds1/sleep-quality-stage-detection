# Sleep Quality and Stage Detection using Multi-Modal Deep Learning with Generative AI

This project detects sleep stages (Wake, NREM, REM) from raw polysomnography (PSG) signals using a custom Temporal Convolutional Network (TCN), and generates clinical sleep reports using Mistral 7B.

## Dataset

- **SHHS1** — 100 subjects (training)
- **SHHS2** — 52 subjects (validation)
- Signals used: EEG, EOG(L), EOG(R)
- Source: [National Sleep Research Resource (NSRR)](https://sleepdata.org/datasets/shhs)

## Pipeline

1. Load EDF files and extract EEG/EOG channels
2. Resample to 125 Hz, apply Butterworth bandpass filter (0.5–45 Hz)
3. Segment into 30-second epochs → reshape to (10, 3, 300)
4. Train SleepTCN model for 3-class classification (Wake / NREM / REM)
5. Compute sleep quality metrics (efficiency, WASO, TST, REM%)
6. Generate clinical report using Mistral 7B Instruct (4-bit quantized)

## Results

| Metric | Value |
|---|---|
| Validation Accuracy | 87% |
| Wake F1 | 0.91 |
| NREM F1 | 0.90 |
| REM F1 | 0.70 |
| Best Epoch | 4 / 14 |
| Model Parameters | 314,019 |

## Requirements

```bash
pip install mne pyEDFlib scipy numpy torch transformers bitsandbytes accelerate scikit-learn matplotlib seaborn
```

## How to Run

1. Open the notebook on Kaggle
2. Add SHHS1 and SHHS2 datasets
3. Enable GPU (T4 or P100)
4. Run all cells

## Tech Stack

- Python, PyTorch
- MNE-Python, SciPy
- Mistral 7B Instruct v0.2 (HuggingFace)
- BitsAndBytes (4-bit quantization)
- Scikit-learn, Matplotlib, Seaborn

## Author

Ankit Dash — B.Tech CSE, Centurion University of Technology and Management  
