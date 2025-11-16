
# ⭐ **Predictive Analytics for Mental Health Wellness Programs**

*AI for Social Good — Data Science, Machine Learning & Behavioral Insights*

---

## 🧠 Overview

This project builds a **full end-to-end predictive analytics system** to understand and forecast **psychological health outcomes** across Indian states. Using a national-scale mental-health dataset (2017–2021), advanced ML models, Bayesian reasoning, and rich Tableau visualizations, the project discovers **what psychosocial, demographic, and lifestyle factors most strongly influence mental wellness**.

The system is designed to help public health organizations and wellness centers:

* Identify **high-risk populations**
* Allocate **healthcare resources**
* Plan **state-level interventions**
* Support **branch expansion decisions**
* Provide **data-driven personalized care**

This project won **1st Place at the DataQuezt 2024 International Business Analytics Conference**.

---

## 🚀 Features

### **1. Full Data Science Pipeline**

✔ Data ingestion & preprocessing
✔ Missing-value handling
✔ Feature engineering (Age buckets, psychosocial segmentation, sleep clusters)
✔ Automatic label encoding & cleaning

### **2. Machine Learning Models**

* **XGBoost Classifier** (best performing)
* **RandomForest Classifier**
* Evaluation using **ROC-AUC, Accuracy, Precision/Recall**

### **3. Explainable AI (XAI) with SHAP**

* Global feature importance
* Local explanations for individual predictions
* Identifies top drivers:
  **sleep duration, psychosocial habits, income, chronic diseases, physical activity**

### **4. Probabilistic Graphical Model**

A Bayesian Network predicts outcome probabilities under hypothetical “what-if” scenarios:

> *"What is the probability of good mental health if sleep improves to 7–9 hours?"*
> *"How does alcohol/smoking influence risk for different income buckets?"*

### **5. Production-Ready Architecture**

* **FastAPI inference service**
* **Dockerized deployment**
* Pipeline stored as reusable models (`joblib`)
* Clean modular code structure

### **6. Interactive Tableau Dashboard**

Rich visual storytelling showcasing:

* Gender-wise visit patterns
* State-wise psychosocial distribution
* Chronic disease trends
* Sleep behavior across states
* Mental health scores over time

<img width="3998" height="1998" alt="Dashboard 1" src="https://github.com/user-attachments/assets/3f172fb8-1f25-4dc8-8d58-e6d1afb1240a" />


---

## 🗂️ Repository Structure

```
mental-health-predictive-analytics/
│
├─ data/                    # raw + processed data
├─ notebooks/               # EDA, ML models, Bayesian Model, Survey analysis
├─ dashboards/              # Tableau screenshots
├─ src/                     # preprocessing + training pipeline
├─ api/                     # FastAPI prediction service
├─ models/                  # serialized ML pipelines
├─ requirements.txt
├─ Dockerfile
└─ README.md
```

---

## 📊 Dataset Summary

The dataset includes:

* 100+ Indian states/UTs
* 50,000+ records (simulated + survey-based)
* Variables such as:

| Category     | Features                                             |
| ------------ | ---------------------------------------------------- |
| Demographics | Age, Gender, Education, Income, Family Structure     |
| Health       | Chronic Diseases, Sleep Duration, Physical Activity  |
| Lifestyle    | Psychosocial Factors (Alcohol, Smoking, Screen Time) |
| Behavior     | Healthcare Visit Frequency, Diet Plan Adherence      |
| Target       | Psychological Health (Good / Bad)                    |

---

## 🛠️ Tech Stack

* **Python** (Pandas, NumPy, Scikit-learn, XGBoost)
* **SHAP** for explainability
* **pgmpy** for Probabilistic Graph Models
* **FastAPI** + **Uvicorn**
* **Docker**
* **Tableau Public**
* **Jupyter Notebooks**

---

## ⚙️ Setup & Usage

### **Install dependencies**

```bash
pip install -r requirements.txt
```

### **Run preprocessing**

```bash
python -m src.preprocess
```

### **Train ML models**

```bash
python -m src.train_ml_models
```

### **Run Bayesian Network**

```bash
python -m src.train_bayesian_network
```

### **Start API service**

```bash
uvicorn api.serve_model:app --reload
```

### **Sample API Request**

```json
{
  "Year": 2021,
  "Indian_States": "Maharashtra",
  "Chronic_Diseases": "Asthma",
  "GENDER": "Female",
  "Age_in_years": 29,
  "Income": 150000,
  "Psychosocial_Factors": "Alcohol and Smoking",
  "Sleep_duration_Hrs": 9,
  "Physical_Activity": 0
}
```

---

## 🧩 Key Insights Discovered

From the ML + Bayesian + Tableau analysis:

### 🔹 **Top Predictors of Good Psychological Health**

1. **Sleep duration (7–9 hours)**
2. **Lower psychosocial risk factors (alcohol, smoking)**
3. **Higher income stability**
4. **Consistent physical activity**
5. **Chronic disease load**

### 🔹 **Behavior Patterns**

* Women tend to visit healthcare facilities more frequently.
* Certain states show disproportionally high psychosocial risk factors.
* Sleep deprivation strongly correlates with “Bad” psychological health.

### 🔹 **Actionable Recommendations**

* Introduce **sleep hygiene programs** in high-risk states.
* Launch **behavioral therapy programs** focusing on psychosocial dependencies.
* Personalize interventions by state, lifestyle, and chronic disease history.

---

## 🌍 Real-World Impact

This project demonstrates how **data science can enable accessible mental wellness solutions**. It can help:

* NGOs
* Wellness centers
* Public health agencies
* Mental health startups
* Policy researchers

to design **targeted, evidence-based interventions**.

---

## 🏆 Acknowledgements

Special thanks to the DataQuezt 2024 organizing committee, Ayur Chikitsa experts, and peers who supported the model validation.

---
