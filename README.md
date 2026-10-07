# DualTL-rPPG: Remote Photoplethysmography using Deep Learning

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

## Description
This repository contains a complete deep learning pipeline for estimating heart rate and vital signs directly from facial video streams using **Remote Photoplethysmography (rPPG)**. 

The project leverages Computer Vision (for facial mesh tracking and region-of-interest extraction) and Deep Learning (1D-CNNs, GRU, and DualTL) to analyze subtle color variations in the human face and predict the underlying pulse. It also features live webcam inference and Explainable AI (XAI) techniques, such as Grad-CAM and occlusion mapping, to visualize exactly how the models interpret facial regions to detect heart rate.

*(Placeholder: Add a screenshot of the live webcam inference or a performance graph here!)*
`![Demo](path/to/your/image.png)`

## Key Features
- **Live Inference:** Real-time heart rate estimation directly from a webcam feed using facial landmarks.
- **Multiple Architectures:** Implements and evaluates several models including custom 1D-CNNs, Bi-directional GRUs, and Dual Transfer Learning (DualTL) architectures.
- **Explainable AI (XAI):** Includes Grad-CAM and Occlusion visualization scripts to understand model focus and build trust in predictions.
- **Comprehensive Pipeline:** End-to-end scripts for data preprocessing, training, ensembling, and rigorous evaluation (e.g., Leave-One-Subject-Out cross-validation).

## Directory Structure
The repository is modular and organized for maintainability:

```text
DualTL-rPPG/
├── src/                    # Core library (datasets, model architectures, utilities)
├── scripts/
│   ├── train/              # Training and baseline scripts
│   ├── eval/               # Evaluation, scoring, and ensemble testing
│   ├── data/               # Data preprocessing and split generation
│   ├── explain/            # XAI scripts (Grad-CAM, Occlusion)
│   ├── visualization/      # Graph generation, statistical reports, and result visualizations
│   ├── check/              # Diagnostics, label checking, and sanity tests
│   └── live/               # Live webcam inference scripts
├── configs.yaml            # Project configurations
├── split.json              # Train/Test splits
└── requirements.txt        # Python dependencies
```

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/kaviraj-debug/DualTL-rPPG.git
   cd DualTL-rPPG
   ```

2. **Create a virtual environment (Optional but recommended):**
   ```bash
   python -m venv dualtl_env
   # On Windows:
   dualtl_env\Scripts\activate
   # On Mac/Linux:
   source dualtl_env/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage Examples

**1. Live Webcam Inference:**
To test the model in real-time using your webcam:
```bash
python scripts/live/live_hr.py
```

**2. Visualizing Results:**
To generate statistical comparisons and performance graphs:
```bash
python scripts/visualization/make_comparison.py
```

**3. Model Training:**
To train the baseline models on your preprocessed dataset:
```bash
python scripts/train/train_baselines.py
```

## Contributing
Feel free to open issues or submit pull requests if you have suggestions or improvements!

## License
This project is licensed under the MIT License.
