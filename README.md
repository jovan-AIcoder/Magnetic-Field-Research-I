# Prediction of Daily Variations of the Earth's Magnetic Field Using Deep Neural Networks

This repository contains a TensorFlow-based deep learning implementation to predict the Earth's magnetic field components (Bx, By, Bz) from time of day. The model is intended to explore daily variation dynamics influenced by ionospheric currents, global current systems, and other electromagnetic processes.

## Introduction

The Earth's magnetic field exhibits daily variations with relatively regular patterns, primarily driven by ionospheric Sq currents, solar radiation, and magnetospheric plasma interactions. These variations are typically on the order of thousands of nanoTesla (nT) and can be modeled empirically using machine learning approaches.
The model in this repository uses a small dataset (48 samples per day) with 30-minute time resolution to learn the daily periodic patterns for the three magnetic field components.

## Model Structure
The model is built as a multilayer feed-forward neural network with the following characteristics:

Architecture:
64 → 32 → 16 → 8 → 4 → 2 → 1

Activation function:
All hidden layers use Mish/Swish, offering smooth non-monotonic behavior and often more stability compared to ReLU.

Output layer: Linear (no activation), suitable for regression.

Data scaling: MinMaxScaler is used to stabilize training.

Epochs: 300

Optimizer:
Adamax or Nadam were chosen for stability on small datasets.

Loss function:
Mean Squared Error (MSE)

Typical loss range observed:

0.0021 – 0.0041 (in scaled space)

which indicates reasonably good fitting given the limited dataset size.

## Dataset

Dataset details:

- 48 samples per day
- 30-minute time resolution (0.5 hours)
- Full 24-hour coverage
- Magnetic field component values in nanoTesla (nT)

Source: magnetometer observations (location not specified)

## Project Goals

- Test whether a simple neural network can learn daily geomagnetic variations.
- Evaluate deep learning performance for geophysical modeling with small datasets.
- Develop a baseline for further research, such as:
  - geomagnetic storm prediction
  - correlation with Kp/Dst indices
  - comparison of physics-based vs data-driven models



## File	Description
`GenerateModel.py`	Main script to train models
`Model_Bx.h5`	Trained model for Bx component
`Model_By.h5`	Trained model for By component
`Model_Bz.h5`	Trained model for Bz component
`README.md`	Project documentation
`datamagnet.csv` Dataset fot training
`scaler_Bx.pkl` Scaler for Bx component
`scaler_By.pkl` Scaler for By component
`scaler_Bz.pkl` Scaler for Bz component
`scaler_Time.pkl` Scaler for time
`viewgraph.py` Show the graph of magnetic variation over time
`predictmagnet.py` You can input time (in decimal hours) to predict the magnetic field
`MagneticProperties.py` Show all information about geomagnetic condition in Earth such as magnetization, magnetic dipole, inclination, etc



## Usage

1. Install dependencies

```bash
pip install tensorflow numpy scikit-learn
```

2. Run predictmagnet.py

```bash
python predictmagnet.py
```
3. Input time to console



## Potential Improvements

- Add data from international geomagnetic observatories (INTERMAGNET).
- Add features such as:
  - date
  - solar activity indices
  - ionospheric conditions
- Use LSTM/Transformer architectures to capture deeper temporal dynamics.
- Compare the model with physics-based simulations (IGRF, CHAOS models).



## License

This repository is provided under a permissive modification-friendly license.



## Contributors

Created by: Jovan Alcoder

Topic: AI for Geophysics / Earth's Magnetic Field Modeling

Made in Indonesia