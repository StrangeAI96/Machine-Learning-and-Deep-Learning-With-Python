# Final Project — MNIST Classification with Modified VGG16

**Notebook:** `Arash_Ganjouri_Final_Project.ipynb`  
**Author:** *Arash Ganjouri*  
**Course:** *Machine Learning & Deep Learning with Python*

---

## 📌 Project Description
This project implements a **handwritten digit classification system** using the **MNIST dataset** and a **modified VGG16** architecture.  
It focuses on feature fusion from multiple convolutional layers, optimization through random search, and performance evaluation using **macro F1-scores**.

The work demonstrates deep learning best practices including preprocessing, model customization, regularization, learning rate scheduling, and early stopping — all implemented in **PyTorch** and executed in **Google Colab** with GPU acceleration.

---

## 📊 Dataset
- **Dataset:** MNIST  
- **Classes:** 10 (digits 0–9)  
- **Image Size:** 28×28 grayscale  
- **Samples:** 60,000 training, 10,000 test images  
- **Source:** [MNIST Dataset (torchvision)](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)

**Dataset Split:**
- 70% (42,000 samples) → Training  
- 20% (12,000 samples) → Validation  
- 10% (6,000 samples) → Extra Test  
- + 10,000 official test images

**Preprocessing:**
- Conversion to tensors with `ToTensor()`  
- Normalization using mean `0.1307`, std `0.3081`  
- No rotation or augmentation needed (properly aligned images)

---

## 🧠 Model Architecture — Modified VGG16
A custom **VGG16-based convolutional neural network** that combines features from multiple depths for improved performance on MNIST.

### **Feature Extractor (5 Blocks)**
1. **Block 1:** 2×`Conv2d(1→64, 3×3, padding=1)` → ReLU → `MaxPool2d(2)` → Output: [64, 14, 14]  
2. **Block 2:** 2×`Conv2d(64→128, 3×3, padding=1)` → ReLU → `MaxPool2d(2)` → Output: [128, 7, 7]  
3. **Block 3:** 3×`Conv2d(128→256, 3×3, padding=1)` → ReLU → `MaxPool2d(2)` → Output: [256, 3, 3]  
4. **Block 4:** 3×`Conv2d(256→512, 3×3, padding=1)` → ReLU → `MaxPool2d(2)` → Output: [512, 1, 1]  
5. **Block 5:** 3×`Conv2d(512→512, 3×3, padding=1)` → ReLU → Output: [512, 1, 1]

### **Feature Fusion**
Feature maps from layers **4**, **21**, and **29** are concatenated:  
- Layer 4: 12,544 features (64×14×14)  
- Layer 21: 4,608 features (512×3×3)  
- Layer 29: 512 features (512×1×1)  
**→ Total Input to Classifier:** 17,664 features

### **Classifier**
- `Linear(17,664→4,096)` → ReLU → Dropout(p)  
- `Linear(4,096→4,096)` → ReLU → Dropout(p)  
- `Linear(4,096→10)` → Logits output  

**Dropout:** 0.3–0.5 (tuned)  
**Total Parameters:** ~30M (depending on configuration)

---

## ⚙️ Training & Evaluation
**Loss Function:** `CrossEntropyLoss`  
**Optimizers:**  
- **Adam**  
- **SGD** (momentum 0.9 or 0.95)

**Schedulers:**
- `ReduceLROnPlateau` (patience=2, factor=0.3)

**Training Protocol:**
- Up to **30 epochs** with **early stopping** (patience=5)  
- Batch sizes: [128, 256]  
- Evaluation metric: **Macro F1-score**  

**Validation & Test Evaluation:**
- The best model (highest validation F1) is saved.  
- Evaluated on both the **official test set (10k)** and the **extra test subset (6k)**.  

**Visualization:**
- 10 random samples plotted for sanity check and qualitative assessment.  

---

## 🔍 Hyperparameter Optimization
A **random search (5 iterations)** explores key hyperparameters:

| Parameter | Search Range |
|------------|---------------|
| Learning Rate | [0.00005, 0.0001, 0.0005, 0.001, 0.005] |
| Batch Size | [128, 256] |
| Dropout | [0.3, 0.4, 0.5] |
| Weight Decay | [1e-5, 5e-5, 1e-4] |
| Optimizer | [Adam, SGD] |
| Momentum (SGD only) | [0.9, 0.95] |

Each trial runs full training with early stopping.  
The configuration with the **best validation F1-score** is used for final testing.

---

## 🧩 Workflow Summary
1. Load and preprocess MNIST dataset.  
2. Split into training, validation, and test subsets.  
3. Define modified VGG16 model (`VGG16Last3`).  
4. Train model using selected optimizer and scheduler.  
5. Apply early stopping and learning rate reduction.  
6. Evaluate model on test sets (macro F1, accuracy).  
7. Run random search for hyperparameter tuning.  
8. Report and visualize best results.

---

## 🧰 Tools & Libraries
- **Python 3.x**  
- **PyTorch** (`torch`, `torchvision`, `torchaudio`)  
- **NumPy**, **Matplotlib**  
- **scikit-learn** (for F1-score and metrics)  
- **Google Colab GPU runtime**

---

## 📈 Results Overview
- Achieved high validation and test **macro F1-scores (>0.95)**.  
- **Early stopping** effectively prevented overfitting.  
- **Feature fusion** from multiple VGG16 layers improved representation quality.  
- **Adam optimizer** generally outperformed SGD for smaller batch sizes.  
- The model reliably distinguished all 10 digits, with minimal confusion.

---

## 🎯 Learning Outcomes
Through this project, we gain experience in:
- Building **custom CNNs** by modifying classical architectures (VGG16).  
- Applying **feature fusion** to enrich representations.  
- Using **F1-score** as a robust metric for multi-class classification.  
- Performing **hyperparameter tuning** under computational constraints.  
- Implementing **early stopping** and **LR scheduling** in PyTorch.

---

