# Seismic Fault Detection using CNN

A hands-on machine learning project that **detects geological faults in seismic sections using a convolutional neural network (CNN), built end-to-end on synthetic SEGY data**.

## Overview

Seismic exploration produces images of underground rock layers by recording sound wave reflections beneath the earth's surface. Interpreters look for faults — breaks or offsets in these layers — because they are important for locating oil and gas reservoirs, assessing geological hazards, and ensuring drilling safety. Manually scanning large seismic datasets for faults is slow and inconsistent, which motivates automating the task with machine learning.

This project builds a complete pipeline that mirrors the real-world workflow used in the seismic industry:

1. Generate synthetic seismic data representing rock layers, with and without faults.
2. Write the data to disk in SEGY format, the standard file format used in seismic exploration, including trace headers and geometry.
3. Read the SEGY files back and extract header information into a structured geometry table.
4. Train a CNN to classify each seismic section as containing a fault or not.
5. Evaluate the model and inspect misclassified examples.

The goal is to practice the full pipeline — SEGY file I/O, trace headers and geometry, and a PyTorch training loop — on a small, fully controlled synthetic dataset before applying the same skills to real seismic data.

## Goal

Build a classification model using a CNN that looks at a seismic 2D image — a picture of underground rock layers — and automatically answers: is there a fault (a break in the layers) in this section, or not?

## Pipeline

### 1. Synthetic Data Generation
Generate a set of synthetic seismic lines using NumPy, modeling horizontal rock layers as wave patterns. For a subset of the lines, introduce a fault by shifting the layers along part of the section.

### 2. SEGY Export
Write each synthetic line to an individual SEGY file using `segyio`, including standard trace headers such as inline number, crossline range, and CMP X/Y coordinates.

### 3. SEGY Reading and Geometry Table
Read the SEGY files back, extract the trace headers, and organize the results into a pandas DataFrame containing inline number, coordinates, and fault labels. Plot the spatial layout to confirm the geometry is consistent.

### 4. Dataset, DataLoader, and Model Training
Implement a PyTorch `Dataset` that loads a seismic section and its label from the geometry table, and a `DataLoader` for batching. Train a small CNN (two to three convolutional layers) to classify each section as fault or no fault, split by inline number rather than randomly to reflect realistic validation practice.

### 5. Evaluation
Report training loss and validation accuracy. Inspect misclassified sections to assess whether errors stem from subtle faults or ambiguous synthetic data.

## Tech Stack

- Python
- segyio — SEGY file reading and writing
- NumPy — synthetic data generation
- pandas — geometry and metadata handling
- Matplotlib — visualization
- PyTorch — CNN model and training loop

## Repository Structure

```
seismic-fault-detection-cnn/
├── data/               # Generated SEGY files
├── src/
│   ├── generate.py     # Synthetic data generation and SEGY writing
│   ├── geometry.py     # SEGY reading and geometry table construction
│   ├── dataset.py       # PyTorch Dataset and DataLoader
│   ├── model.py          # CNN architecture
│   └── train.py          # Training and evaluation loop
├── notebooks/          # Exploration and visualization notebooks
├── requirements.txt
└── README.md
```

## Status

This is a learning project built to practice seismic data handling and applied deep learning on a synthetic dataset. It is not intended for use on real field data in its current form.

## License

MIT
