# 🔥 Wildfire Risk Classification from Satellite and Weather Data

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-1.2+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/XGBoost-1.7+-006600?style=for-the-badge&logo=xgboost&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.28+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Status-Complete-22C55E?style=for-the-badge" />
</p>

---

## 👥 Team Members

| # | Name | Reg | Responsibilities |
|---|------|------|-----------------|
| 1 | Anagha C | 253126 | Data preprocessing, cleaning, and Exploratory Data Analysis |
| 2 | Savin Jees |  | Model implementation — SVM, Random Forest, XGBoost |
| 3 | V A Sreehari | 253313 | Visualization, documentation, Streamlit deployment |

---

## 📌 Problem Statement

Wildfires pose a severe threat to ecosystems, human settlements, and biodiversity worldwide. Early and accurate prediction of wildfire risk is critical for disaster management and resource allocation. This project leverages **machine learning** to classify wildfire risk levels (**Low**, **Medium**, **High**) using meteorological and vegetation-related features from the Algerian Forest Fire Dataset.

---

## 📊 Dataset Description

| Property | Details |
|----------|---------|
| **Name** | Algerian Forest Fire Dataset |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/547/algerian+forest+fires+dataset) |
| **Samples** | 244 instances (2 regions: Bejaia & Sidi Bel-Abbes) |
| **Features** | 13 meteorological & fire weather index attributes |
| **Target** | Binary classification — Fire / Not Fire |

### 🔑 Key Features

| Feature | Description |
|---------|-------------|
| `Temperature` | Maximum temperature (°C) |
| `RH` | Relative Humidity (%) |
| `Ws` | Wind Speed (km/h) |
| `Rain` | Total rainfall (mm) |
| `FFMC` | Fine Fuel Moisture Code |
| `DMC` | Duff Moisture Code |
| `DC` | Drought Code |
| `ISI` | Initial Spread Index |
| `BUI` | Build Up Index |
| `FWI` | Fire Weather Index |

---

## 🛠️ Technologies Used

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.9+ |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | scikit-learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Web Application** | Streamlit |
| **Model Serialization** | Joblib |
| **IDE** | VS Code |

---

## 🤖 Machine Learning Models

### 1️⃣ Support Vector Machine (SVM)
- Kernel: RBF (Radial Basis Function)
- Regularization parameter C = 1.0
- Probability estimates enabled
- Best for high-dimensional feature spaces

### 2️⃣ Random Forest
- 200 decision trees (estimators)
- Max depth: 10
- Provides feature importance scores
- Robust against overfitting

### 3️⃣ XGBoost
- 200 boosting rounds
- Learning rate: 0.1
- Max depth: 6
- State-of-the-art gradient boosting

---

## 🔄 Workflow

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Dataset    │───▶│  Preprocessing   │───▶│  Feature Eng.   │
│  (Algerian FF)   │    │  & Cleaning      │    │  & Selection    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                       │
                              ┌─────────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  Train/Test Split │
                    │   (80/20)        │
                    └──────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐   ┌──────────────┐   ┌──────────────┐
   │     SVM     │   │ Random Forest│   │   XGBoost    │
   └─────────────┘   └──────────────┘   └──────────────┘
          │                   │                   │
          └───────────────────┼───────────────────┘
                              ▼
                    ┌──────────────────┐
                    │   Evaluation     │
                    │  & Comparison    │
                    └──────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
           ┌──────────────┐   ┌──────────────┐
           │ Visualizations│   │  Streamlit   │
           │  (PNG plots)  │   │  Web App     │
           └──────────────┘   └──────────────┘
```

**Step-by-step:**
1. **Data Collection** — Algerian Forest Fire Dataset (CSV)
2. **Preprocessing** — Null handling, encoding, standardization (Member 1)
3. **Model Training** — SVM, Random Forest, XGBoost (Member 2)
4. **Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrices
5. **Visualization** — Professional plots and charts (Member 3)
6. **Deployment** — Interactive Streamlit web application (Member 3)

---

## 📈 Visualization Results

The following visualizations are generated automatically by `main.py`:

| # | Visualization | Description |
|---|--------------|-------------|
| 1 | `accuracy_comparison.png` | Bar chart comparing model accuracies |
| 2 | `metrics_comparison.png` | Grouped bars for all metrics across models |
| 3 | `feature_importance_*.png` | Feature importance (RF & XGBoost) |
| 4 | `risk_distribution.png` | Pie + bar chart of risk levels |
| 5 | `confusion_matrix_*.png` | Per-model confusion matrix heatmaps |
| 6 | `confusion_matrices_all.png` | Side-by-side comparison |
| 7 | `prediction_probs_*.png` | Prediction probability distributions |
| 8 | `correlation_heatmap.png` | Feature correlation matrix |
| 9 | `class_distribution.png` | Binary target class balance |
| 10 | `model_radar_chart.png` | Radar/spider chart model comparison |

All plots are saved to `outputs/plots/` in high-resolution PNG format.

---

## 🚀 How to Run the Project

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Wildfire-Risk-Classification.git
cd Wildfire-Risk-Classification

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# Generate predictions, metrics, and all visualizations
python main.py
```
