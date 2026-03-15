# 💳 Credit Card Fraud Detection Using Machine Learning

> A supervised machine learning project to detect fraudulent credit card transactions using Random Forest Classification on a highly imbalanced dataset.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Workflow](#workflow)
- [Key Results](#key-results)
- [Visualizations](#visualizations)
- [How to Run](#how-to-run)
- [Possible Improvements](#possible-improvements)
- [Author](#author)

---

## 📖 Project Overview

Credit card fraud is a major global financial threat. This project builds a machine learning pipeline to automatically classify transactions as **legitimate (0)** or **fraudulent (1)** using historical transaction data. The core challenge is the extreme class imbalance — fraud cases account for only **~0.17%** of all transactions.

**Goal:** Maximize fraud detection (Recall) while keeping false alarms (False Positives) low.

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| Total Records | 284,807 transactions |
| Features | 31 columns (Time, V1–V28, Amount, Class) |
| Target Column | `Class` → 0 = Legitimate, 1 = Fraud |
| Fraud Cases | 492 (~0.17%) |
| Legitimate Cases | 284,315 (~99.83%) |
| Missing Values | None |
| Duplicates | 1,081 rows (removed during cleaning) |

> **Note:** Features V1–V28 are the result of PCA transformation to protect user confidentiality. Only `Time` and `Amount` are in their original form.

---

## 📁 Project Structure

```
credit-card-fraud-detection/
│
├── creditcard.csv                          # Raw dataset (download from Kaggle)
├── Credit_Card_Fraud_Detection.ipynb       # Main Jupyter Notebook
├── README.md                               # Project documentation
└── requirements.txt                        # Python dependencies
```

---

## 🛠️ Tech Stack

| Category | Libraries / Tools |
|---|---|
| Language | Python 3.10+ |
| Data Manipulation | `pandas`, `numpy` |
| Visualization | `matplotlib`, `seaborn` |
| Machine Learning | `scikit-learn` |
| Environment | Jupyter Notebook / Google Colab |

---

## 🔄 Workflow

```
1. Import Libraries
       ↓
2. Load & Explore Dataset
   └── shape, head, tail, info, describe
       ↓
3. Data Cleaning
   ├── 3a. Handle Missing Values  → None found
   └── 3b. Remove Duplicate Rows → 1,081 removed
       ↓
4. Exploratory Data Analysis (EDA)
   ├── Q1: Class distribution (Fraud vs Legitimate %)
   ├── Q2: Fraud transaction amount distribution
   └── Q3: Amount comparison across both classes
       ↓
5. Model Development
   ├── 5a. Feature/Target split (X, y)
   ├── 5b. Train-Test split (80/20, stratified)
   └── 5c. Train Random Forest Classifier
       ↓
6. Model Evaluation
   ├── Classification Report
   ├── Confusion Matrix (numeric + heatmap)
   └── Feature Importance Plot
       ↓
7. Summary & Conclusions
```

---

## 📈 Key Results

### Model: Random Forest Classifier (`n_estimators=100`, `random_state=41`)

| Metric | Legitimate (0) | Fraud (1) |
|---|---|---|
| Precision | 1.00 | 0.90 |
| Recall | 1.00 | 0.80 |
| F1-Score | 1.00 | 0.85 |
| Support | 56,671 | 75 |

**Overall Accuracy: ~99.97%**

### Confusion Matrix

```
                  Predicted
                  Legit   Fraud
Actual  Legit  [ 56664     7  ]
        Fraud  [    15    60  ]
```

| | Count |
|---|---|
| True Negatives (Legit correctly identified) | 56,664 |
| False Positives (Legit flagged as Fraud) | 7 |
| False Negatives (Fraud missed) | 15 |
| True Positives (Fraud correctly detected) | 60 |

> Out of 75 fraud cases in the test set, **60 were correctly detected** and only 15 were missed.

---

## 📊 Visualizations

The notebook includes the following plots:

- **Pie Chart** — Class distribution (Fraud vs. Legitimate %)
- **Histogram** — Fraud transaction amount distribution
- **Side-by-side Histograms** — Amount comparison (Fraud vs. Legitimate)
- **Confusion Matrix Heatmap** — Visual breakdown of predictions
- **Feature Importance Bar Chart** — Top 15 most predictive features

---

## ▶️ How to Run

### 1. Clone or Download the Repository

```bash
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Download the Dataset

Download `creditcard.csv` from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and place it in the project root directory.

### 4. Launch the Notebook

```bash
jupyter notebook Credit_Card_Fraud_Detection.ipynb
```

---

## 🚀 Possible Improvements

| Technique | Purpose |
|---|---|
| **SMOTE / Undersampling** | Handle severe class imbalance |
| **XGBoost / LightGBM** | Potentially better performance |
| **Hyperparameter Tuning** | `GridSearchCV` or `RandomizedSearchCV` |
| **ROC-AUC Curve** | Better evaluation metric for imbalanced data |
| **Precision-Recall Curve** | Ideal metric for fraud detection |
| **Cross-Validation** | More robust model evaluation |
| **Feature Scaling** | Normalize `Amount` and `Time` columns |

---

## 👨‍💻 Author

**Rushikesh Sangamnere**

**Email: rushikeshsangamnere4561@gmail.com**

**Phone: +91 9096506345**


---

## 📄 License

This project is for educational purposes. The dataset is publicly available on Kaggle under its own license terms.
