# Project 4 — EMNIST Image Classification with ConvNN

**Notebook:** `Arash_Ganjouri_Project4.ipynb`  
**Author:** *Arash Ganjouri*  
**Course:** *Machine Learning & Deep Learning with Python*

---

## 📌 Project Description
This project trains and evaluates a **Convolutional Neural Network (ConvNN)** to classify handwritten characters from the **EMNIST Balanced** dataset, which includes both digits and uppercase/lowercase letters (47 total classes).

The work focuses on data preprocessing, CNN design, and performance evaluation using modern deep learning techniques in **PyTorch**.

---

## 📊 Dataset
- **Dataset:** EMNIST (Balanced split)  
- **Classes:** 47 (digits + uppercase/lowercase letters)  
- **Image Size:** 28×28 grayscale  
- **Samples:** ~131,600 training images, ~22,400 testing images  
- **Source:** [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset) (via `torchvision.datasets`)

**Preprocessing Steps:**
- Rotation and horizontal flipping to fix EMNIST’s orientation  
- Normalization with mean `0.1307`, std `0.3081` (same as MNIST)  
- Random 90%/10% split of training data into **train** and **validation** sets

---

## 🧠 Model Architecture — ConvNN
A compact, custom **Convolutional Neural Network** designed for EMNIST’s 28×28 grayscale inputs.

### **Feature Extractor (Convolutional Blocks)**
1. `Conv2d(1→32, 3×3)` → `ReLU` → `MaxPool2d(2)`  
2. `Conv2d(32→64, 3×3)` → `ReLU` → `MaxPool2d(2)`

### **Classifier (Fully Connected Layers)**
1. `Linear(3136→128)` → `ReLU` → `Dropout(0.20)`  
2. `Linear(128→128)` → `BatchNorm1d` → `ReLU` → `Dropout(0.20)`  
3. `Linear(128→64)` → `BatchNorm1d` → `ReLU` → `Dropout(0.20)`  
4. `Linear(64→64)` → `BatchNorm1d` → `ReLU` → `Dropout(0.15)`  
5. `Linear(64→47)` (Output logits for each class)

**Design Rationale:**
- Convolutions capture local shape/stroke features.  
- Batch normalization stabilizes deeper fully connected layers.  
- Dropout reduces overfitting.

---

## ⚙️ Training & Evaluation
**Loss Function:** `CrossEntropyLoss`  
**Optimizers:**
- Default: **Adam** (lr = 1e-3)  
- Optionally: **SGD** with momentum  

**Hyperparameters:**
- Epochs: 5 (baseline)  
- Batch size: 64  
- Random seed: 42 for reproducibility  

**Metrics:**
- Training/validation loss  
- Accuracy per epoch  
- Final test accuracy  
- Confusion matrix & classification report (precision, recall, F1-score)

---

## 🧩 Workflow Summary
1. Install dependencies and import libraries (PyTorch, Torchvision, scikit-learn, matplotlib, W&B).  
2. Load EMNIST (Balanced) dataset.  
3. Apply rotation, flipping, and normalization transforms.  
4. Split into train/validation/test loaders.  
5. Define CNN (`ConvNN`) model.  
6. Train the model for several epochs while tracking validation metrics.  
7. Evaluate on the held-out test set and generate a confusion matrix.

---

## 🧰 Tools & Libraries
- **Python 3.x**  
- **PyTorch** (`torch`, `torchvision`, `torchaudio`)  
- **NumPy**, **Matplotlib**  
- **scikit-learn** (for evaluation metrics)  
- **Weights & Biases (wandb)** — for experiment tracking

---

## 📈 Results Overview
- Achieved strong baseline accuracy after 5 epochs with minimal overfitting.  
- Validation accuracy improved with dropout and batch normalization.  
- Orientation correction was critical — without it, accuracy degraded significantly.  
- The model effectively differentiates digits and most alphabetic classes, though confusion exists between visually similar characters (e.g., “O” vs “0”, “l” vs “I”).

---

## 🎯 Learning Outcomes
Through this project, we learn to:
- Build and train CNNs in **PyTorch** for real-world image datasets.  
- Apply effective preprocessing for dataset quirks (rotation, normalization).  
- Use validation splits to tune hyperparameters.  
- Interpret classification metrics and confusion matrices.

---
