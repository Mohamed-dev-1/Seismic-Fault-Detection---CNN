# Seismic Fault Detection with CNN

An end-to-end machine learning project for detecting geological faults in seismic sections using **SEG-Y data, seismic trace geometry, NumPy, Pandas, and PyTorch**.

The project combines seismic data processing and deep learning into a single practical workflow:

**Synthetic seismic generation → SEG-Y writing → SEG-Y reading → Header extraction → Geometry mapping → Dataset construction → CNN training → Fault classification**

The primary goal of this project is to build practical experience with both **seismic data processing** and **deep learning workflows** rather than treating them as separate exercises.

---

## Project Overview

Seismic data is commonly stored in the **SEG-Y (SEG-Y)** format, which contains seismic traces together with metadata describing their acquisition geometry.

In this project, synthetic seismic sections are generated to represent geological layers. Some sections contain an artificial geological fault, while others contain continuous layers.

The generated data is then written to SEG-Y files and processed through a complete machine learning pipeline.

The final model is a convolutional neural network (CNN) that learns to classify seismic sections into two categories:

* `Fault`
* `No Fault`

This creates a controlled environment where the entire dataset is understood and reproducible while still using realistic seismic data structures.

---

## Learning Objectives

This project is designed to combine three practical learning objectives.

### 1. SEG-Y Data Processing

Learn how to:

* Open SEG-Y files using `segyio`
* Read seismic traces
* Understand traces and time samples
* Convert traces into NumPy arrays
* Display seismic sections with `matplotlib`
* Control amplitude visualization
* Understand the basic structure of SEG-Y files

### 2. SEG-Y Headers and Geometry

Learn how to:

* Read trace headers
* Extract CMP X/Y coordinates
* Extract inline and crossline information
* Build a geometry table using Pandas
* Map seismic lines into spatial coordinates
* Visualize the spatial distribution of seismic lines
* Understand the relationship between seismic data and acquisition geometry

### 3. PyTorch Deep Learning

Learn how to:

* Create a custom PyTorch `Dataset`
* Use `DataLoader`
* Prepare tensors from seismic data
* Build a convolutional neural network
* Implement a training loop
* Use loss functions and optimizers
* Train on a GPU with CUDA
* Separate training and validation data
* Monitor training loss
* Evaluate classification performance

---

# Pipeline

The complete workflow is:

```text
Synthetic Geological Model
          |
          v
Generate Seismic Sections
          |
          v
Write SEG-Y Files
          |
          v
Read SEG-Y with segyio
          |
          +----------------------+
          |                      |
          v                      v
Read Seismic Traces       Read Trace Headers
          |                      |
          v                      v
Build Seismic Images      Build Geometry Table
          |                      |
          +----------+-----------+
                     |
                     v
              Build Dataset
                     |
                     v
             Train / Validation
                     |
                     v
              PyTorch CNN
                     |
                     v
              Fault Prediction
                     |
                     v
            Evaluate Predictions
```

---

# Project Structure

```text
Seismic-Fault-Detection---CNN/
│
├── data/
│   ├── raw/
│   │   └── synthetic SEG-Y files
│   │
│   └── processed/
│       └── dataset metadata
│
├── src/
│   ├── generate_data.py
│   ├── write_segy.py
│   ├── read_segy.py
│   ├── geometry.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── notebooks/
│   ├── seismic_visualization.ipynb
│   └── model_experiments.ipynb
│
├── outputs/
│   ├── seismic_sections/
│   ├── geometry_maps/
│   ├── training_curves/
│   └── predictions/
│
├── requirements.txt
├── README.md
└── LICENSE
```

The exact structure may evolve as the project develops.

---

# Dataset

Instead of relying on a pre-existing labeled seismic dataset, this project generates a controlled synthetic dataset.

This approach provides several advantages:

* The geological structures are known.
* Labels are automatically generated.
* The dataset can be reproduced.
* Different fault characteristics can be introduced.
* The difficulty of the classification problem can be controlled.
* The complete SEG-Y workflow can be practiced without depending on an external dataset.

## Synthetic Classes

### No Fault

Synthetic seismic layers remain approximately continuous across the section.

```text
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
```

### Fault

A displacement is introduced into the seismic layers.

```text
~~~~~~~~~~~~~~~~~~
~~~~~~~~~~~~~~~~~~
              ~~~~~~~~~~~~~
              ~~~~~~~~~~~~~
              ~~~~~~~~~~~~~
```

The exact synthetic structure can be modified to produce different fault orientations, offsets, amplitudes, and noise levels.

---

# Stage 1 — Synthetic Seismic Generation

The first stage generates artificial seismic sections using NumPy.

The sections represent simplified geological layers with controlled characteristics such as:

* Layer continuity
* Amplitude
* Frequency
* Noise
* Layer displacement
* Fault location
* Fault offset

Approximately 40 seismic lines are generated for the initial experiment.

A subset contains synthetic faults while the remaining sections contain continuous geological layers.

The generated sections are then exported as SEG-Y files.

---

# Stage 2 — SEG-Y I/O

The generated SEG-Y files are read using `segyio`.

A seismic line can be represented as a matrix:

```text
              Traces
        0   1   2   3   4   ...
      +-------------------------
  0   |   .   .   .   .   .
  1   |   .   .   .   .   .
  2   |   .   .   .   .   .
Time  |   .   .   .   .   .
Samples
      |   .   .   .   .   .
      +-------------------------
```

The resulting NumPy array has the general structure:

```text
[time samples, traces]
```

This representation can then be visualized as an image.

Example:

```python
with segyio.open("line.sgy", "r", ignore_geometry=True) as f:
    data = np.stack(
        [f.trace[i] for i in range(f.tracecount)]
    ).T
```

The seismic amplitudes are displayed using `matplotlib`.

---

# Stage 3 — Trace Headers and Geometry

SEG-Y traces contain metadata in their headers.

The project extracts spatial information such as:

* Inline number
* Crossline number
* CMP X coordinate
* CMP Y coordinate

This information is converted into a Pandas DataFrame.

Example conceptual structure:

| line     | inline |      x |       y | label |
| -------- | -----: | -----: | ------: | ----- |
| line_001 |    100 | 500000 | 4020000 | 0     |
| line_002 |    101 | 500025 | 4020000 | 1     |
| line_003 |    102 | 500050 | 4020000 | 0     |

This table becomes the connection between:

**seismic data → spatial geometry → machine learning dataset**

The X/Y coordinates can then be plotted to visualize the spatial arrangement of the seismic lines.

---

# Stage 4 — PyTorch Dataset

Instead of loading the entire dataset into memory at once, a custom PyTorch `Dataset` loads seismic sections when they are requested.

Conceptually:

```python
class SeismicDataset(Dataset):

    def __getitem__(self, index):
        # Load SEG-Y section
        # Convert to tensor
        # Return image and label
        return seismic_tensor, label
```

This allows the machine learning pipeline to work directly with the SEG-Y files.

The resulting data flow is:

```text
Geometry DataFrame
        |
        v
     Dataset
        |
        v
   DataLoader
        |
        v
     CNN Model
```

---

# Stage 5 — Train / Validation Split

The dataset is divided into training and validation sets.

The split is performed by seismic line rather than randomly mixing individual pixels or samples.

This is important because neighboring seismic data can be highly correlated.

A conceptual split is:

```text
Training Lines
      |
      +-- Line 001
      +-- Line 002
      +-- Line 003
      +-- ...

Validation Lines
      |
      +-- Line 031
      +-- Line 032
      +-- Line 033
      +-- ...
```

This provides a more meaningful test of whether the model can recognize the fault pattern on seismic lines it has not seen during training.

---

# Stage 6 — Convolutional Neural Network

A small CNN is used as the initial model.

The architecture is intentionally simple because the main objective is to understand the complete training pipeline.

Conceptually:

```text
Input Seismic Section
        |
        v
Convolution
        |
        v
Activation
        |
        v
Pooling
        |
        v
Convolution
        |
        v
Activation
        |
        v
Pooling
        |
        v
Fully Connected Layer
        |
        v
Fault / No Fault
```

The network learns spatial patterns associated with the synthetic geological structures.

---

# Stage 7 — Training

The model is trained using a standard PyTorch training loop.

The training process includes:

1. Load a batch from the `DataLoader`
2. Move tensors to the selected device
3. Perform a forward pass
4. Calculate the loss
5. Clear previous gradients
6. Perform backpropagation
7. Update model parameters
8. Repeat for all batches
9. Evaluate on the validation set

The project also introduces GPU training through:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

The model and tensors are then moved to the selected device.

---

# Evaluation

The initial evaluation focuses on classification accuracy and training behavior.

The project records:

* Training loss
* Validation loss
* Validation accuracy
* Correct predictions
* Incorrect predictions

Training curves can be visualized to identify whether the model is learning effectively.

Example:

```text
Epoch
 |
 |\
 | \
 |  \
 |   \
 |    \________
 |
 +----------------
       Loss
```

---

# Prediction Inspection

Numerical accuracy alone is not enough for seismic interpretation.

For selected validation samples, the project compares:

```text
Seismic Section
      |
      +---- True Label
      |
      +---- Predicted Label
```

For example:

```text
True:      Fault
Predicted: Fault
```

or:

```text
True:      Fault
Predicted: No Fault
```

Incorrect predictions are particularly useful because they can reveal limitations in:

* Synthetic data generation
* Fault visibility
* Noise levels
* Model architecture
* Dataset size
* Generalization

---

# Technologies

| Technology   | Purpose                                         |
| ------------ | ----------------------------------------------- |
| Python       | Main programming language                       |
| NumPy        | Numerical and synthetic seismic data generation |
| Pandas       | Geometry and dataset metadata                   |
| Matplotlib   | Seismic and geometry visualization              |
| segyio       | SEG-Y file reading and writing                  |
| PyTorch      | CNN and deep learning pipeline                  |
| CUDA         | GPU acceleration                                |
| Jupyter      | Experiments and visualization                   |
| Git / GitHub | Version control                                 |

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Mohamed-dev-1/Seismic-Fault-Detection---CNN.git

cd Seismic-Fault-Detection---CNN
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

The intended workflow is:

### 1. Generate the synthetic seismic dataset

```bash
python src/generate_data.py
```

### 2. Write the generated data to SEG-Y

```bash
python src/write_segy.py
```

### 3. Read and visualize SEG-Y sections

```bash
python src/read_segy.py
```

### 4. Extract geometry

```bash
python src/geometry.py
```

### 5. Train the CNN

```bash
python src/train.py
```

### 6. Evaluate the model

```bash
python src/evaluate.py
```

The exact commands may change as the implementation evolves.

---

# Key Concepts Practiced

This project is intentionally designed as a learning project.

## Seismic Data

* SEG-Y
* Traces
* Time samples
* Trace headers
* CMP
* Inline
* Crossline
* Coordinates
* Seismic amplitudes
* Seismic sections

## Data Processing

* NumPy arrays
* Pandas DataFrames
* Data normalization
* Visualization
* Dataset indexing
* File-based datasets

## Deep Learning

* Tensors
* Datasets
* DataLoaders
* CNNs
* Forward propagation
* Backpropagation
* Loss functions
* Optimizers
* Training loops
* Validation
* GPU acceleration

---

# Learning-First Development Strategy

This project is intentionally developed incrementally.

Rather than starting with a complete implementation, each stage is implemented and tested independently.

```text
Phase 1
Understand seismic arrays
        |
        v
Phase 2
Read SEG-Y files
        |
        v
Phase 3
Understand trace headers
        |
        v
Phase 4
Build geometry table
        |
        v
Phase 5
Create PyTorch Dataset
        |
        v
Phase 6
Build CNN
        |
        v
Phase 7
Write training loop
        |
        v
Phase 8
Train and evaluate
        |
        v
Phase 9
Analyze incorrect predictions
```

The goal is not simply to obtain a working model.

The goal is to understand what happens at every stage of the pipeline.

---

# Challenges and Experiments

After the basic pipeline works, the dataset and model can be progressively improved.

Possible experiments include:

### Data

* Increase the number of seismic lines
* Add Gaussian noise
* Change fault displacement
* Change fault position
* Generate dipping layers
* Generate multiple faults
* Change seismic frequencies
* Introduce amplitude variations

### Model

* Change the number of convolutional layers
* Change kernel sizes
* Add Batch Normalization
* Add Dropout
* Compare different optimizers
* Experiment with learning rates
* Increase the number of filters

### Training

* Compare CPU and GPU training
* Experiment with batch sizes
* Experiment with different epoch counts
* Track training and validation loss
* Investigate overfitting

---

# Future Extensions

The classification problem is only the first step.

Possible future extensions include:

## 1. Fault Localization

Instead of predicting whether a section contains a fault, predict where the fault occurs.

```text
Input:
Seismic Section

Output:
Fault Location
```

## 2. Semantic Segmentation

Train a CNN to classify individual pixels or regions as:

```text
Background
Fault
```

This would move the project from classification toward seismic interpretation.

## 3. 3D Seismic Volumes

Extend the pipeline from 2D seismic lines to 3D seismic volumes.

```text
2D Section
     |
     v
Multiple Sections
     |
     v
3D Seismic Volume
```

## 4. Real Seismic Data

Replace the synthetic dataset with publicly available seismic datasets.

This introduces additional challenges such as:

* Missing metadata
* Different coordinate systems
* Noise
* Data normalization
* Large file sizes
* Class imbalance
* Complex geological structures

## 5. Seismic Interpolation

A longer-term extension is to investigate machine-learning approaches for reconstructing missing seismic traces.

This transforms the problem from:

```text
Classification
```

into:

```text
Seismic Reconstruction / Interpolation
```

---

# What I Learned

This project is intended to demonstrate practical understanding of a complete data-to-model workflow:

```text
Raw Data
   ↓
Data Format
   ↓
Metadata
   ↓
Geometry
   ↓
Preprocessing
   ↓
Dataset
   ↓
Deep Learning
   ↓
Training
   ↓
Evaluation
   ↓
Analysis
```

The important lesson is that machine learning is not only about designing a neural network.

A successful ML system also requires understanding how data is stored, how metadata describes the data, how the dataset is constructed, and how the model's predictions are evaluated.

---

# Project Status

**Status:** In Development

Current focus:

* [ ] Synthetic seismic generation
* [ ] SEG-Y writing
* [ ] SEG-Y reading
* [ ] Seismic visualization
* [ ] Trace header extraction
* [ ] Geometry mapping
* [ ] PyTorch Dataset
* [ ] DataLoader
* [ ] CNN implementation
* [ ] GPU training
* [ ] Validation
* [ ] Prediction visualization
* [ ] Error analysis

---

# Author

**Mohamed-dev-1**

Computer Science student interested in:

* Artificial Intelligence
* Machine Learning
* Deep Learning
* Data Science
* Software Development
* Scientific Computing

GitHub:
https://github.com/Mohamed-dev-1

---

# License

This project is intended primarily for educational and experimental purposes.
