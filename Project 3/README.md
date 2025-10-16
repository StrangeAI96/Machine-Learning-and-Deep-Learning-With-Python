# 🩺 Project 3 — Medical Text Classification  
**Author:** [Arash Ganjouri](https://github.com/arash-ganjouri)  
**Course:** *Comprehensive Machine Learning & Deep Learning with Python*  
**Notebook:** `Arash_Ganjouri_Project3.ipynb`  
**Folder:** `Project 3`  

---

## 📌 Project Description  
This project develops and evaluates a **medical text classification system** that categorizes clinical transcripts into one of four classes:  
1. **Surgery**  
2. **Medical Records**  
3. **Internal Medicine**  
4. **Other**  

Two text vectorization strategies are implemented:  
- **Binary Bag-of-Words (BBoW)**  
- **Frequency Bag-of-Words (FBoW)**  

Multiple machine learning models are trained and evaluated on both representations to determine which yields better classification performance using **weighted F1-score**.

---

## 📊 Dataset  
- **Type:** Medical text corpus (CSV format)  
- **Samples:**  
  - Training: 4000  
  - Validation: 500  
  - Test: 500  
- **Features:** Transcript text  
- **Target:** Class label (1–4, adjusted to 0–3 for model compatibility)

---

## 🧠 Models Implemented  
- Logistic Regression  
- Decision Tree Classifier  
- Random Forest Classifier  
- XGBoost Classifier  

Each model is trained and tested using both **BBoW** and **FBoW** features for comparative analysis.

---

## 🛠️ Tools & Libraries  
- Python 3.x  
- NumPy, Pandas  
- scikit-learn (`CountVectorizer`, `metrics`, `model_selection`, `linear_model`, `tree`, `ensemble`)  
- XGBoost  
- Google Colab (for execution and file handling)

---

## 🚀 Workflow  
1. **Load data** — Import `train.csv`, `valid.csv`, and `test.csv` into Colab  
2. **Preprocess text** — Lowercasing and punctuation removal  
3. **Vectorization**  
   - **BBoW:** Binary word presence vectors  
   - **FBoW:** Frequency-based word count vectors, normalized  
4. **Model training** — Fit Logistic Regression, Decision Tree, Random Forest, and XGBoost  
5. **Evaluation** — Compute **weighted F1-score** for validation and test sets  
6. **Comparison** — Analyze performance difference between BBoW and FBoW models  

---

## 📈 Results  
- **BBoW**: Simpler representation, faster to train  
- **FBoW**: Richer feature representation but required normalization for stability  
- **XGBoost** achieved the **highest weighted F1-score** overall  
- Demonstrates the trade-off between feature complexity and generalization performance  

---

## 🎯 Learning Outcomes  
- Implementing **Bag-of-Words** models for NLP tasks  
- Comparing text feature representations (binary vs. frequency)  
- Training and evaluating traditional ML models on text data  
- Building an end-to-end classification pipeline in Google Colab  

---

## 📂 Output Files  
- `bbow_predictions.csv` — Model predictions using Binary BoW  
- `fbow_predictions.csv` — Model predictions using Frequency BoW  
- Weighted F1-scores for model comparison  

---

