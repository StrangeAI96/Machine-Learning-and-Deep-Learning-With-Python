# Project 2 — Linear Regression on California Housing Dataset  

**Notebook:** `Machine_&_Deep_Learning_Course_project_2.ipynb`  
**Folder:** `Project 2`  

---

## 📌 Project Description  
This project applies **Linear Regression** to the **California Housing dataset** using two approaches:  
1. **Analytical solution** (closed-form normal equations)  
2. **Gradient Descent optimization**  

The goal is to compare the efficiency and accuracy of both methods under different preprocessing and learning rate settings.  

---

## 📊 Dataset  
- **Source:** California Housing dataset (from `sklearn.datasets`)  
- **Samples:** 20,640  
- **Features:** 8 (e.g., median income, house age, average rooms, etc.)  
- **Target:** Median house value  

---

## 🛠️ Tools & Libraries  
- Python 3.x  
- NumPy, Pandas  
- Matplotlib  
- scikit-learn (datasets, train_test_split, preprocessing)  

---

## 🚀 Workflow  
1. Load dataset and split into training (80%) and testing (20%) sets  
2. Preprocess features with **StandardScaler** and **MinMaxScaler**  
3. Implement Linear Regression:  
   - Analytical solution (closed-form)  
   - Gradient Descent optimization  
4. Evaluate models using **Root Mean Square Error (RMSE)**  
5. Visualize RMSE behavior across epochs and learning rates  

---

## 📈 Results  
- **Analytical solution** provides exact regression weights with minimal computation.  
- **Gradient Descent** performance is sensitive to scaling and learning rates.  
- RMSE plots show how preprocessing impacts convergence speed and accuracy.  

---

## 🎯 Learning Outcome  
Through this project, we gain experience with:  
- Comparing analytical and iterative approaches to Linear Regression  
- Understanding the role of preprocessing (StandardScaler vs. MinMaxScaler)  
- Evaluating models using RMSE and convergence analysis  

---
