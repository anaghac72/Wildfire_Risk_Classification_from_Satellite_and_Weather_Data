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
| 1 | Anagha C |  | Data preprocessing, cleaning, and Exploratory Data Analysis |
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

