# Employee Attrition Prediction – Machine Learning Pipeline + Streamlit UI

This project implements a complete end-to-end machine learning workflow for predicting employee attrition using the IBM HR Analytics dataset. It includes data preprocessing, feature engineering, model training, evaluation, and an interactive Streamlit application for real-time predictions.

---

## 📊 Project Overview

Organizations face costly turnover. This tool provides a reproducible ML pipeline that predicts the likelihood of voluntary attrition based on employee characteristics such as age, job role, income, satisfaction scores, and overtime status.

The included Streamlit dashboard allows HR reviewers or analysts to:

- Adjust employee attributes through UI sliders and dropdowns  
- Generate model-driven attrition risk predictions  
- Visualize workforce-level attrition trends  
- Explore department-level and demographic insights  

---

## 🧠 Machine Learning Pipeline

### **Models Implemented**
- Logistic Regression  
- Random Forest Classifier  
- Gradient Boosting Classifier  

### **Methods & Engineering Steps**
- Missing-value handling  
- Categorical encoding (OneHotEncoder)  
- ColumnTransformer for mixed features  
- Train/test split  
- Hyperparameter search (grid/random tuning)  
- Model comparison using:
  - Accuracy  
  - Precision  
  - Recall  
  - Confusion matrix  
  - ROC-AUC  

### **Artifacts Produced**
- Preprocessed dataset  
- Trained model saved via `joblib`  
- Modular Python scripts (`app.py` and `train_model.py`)  

---

## 🖥️ Streamlit Application (UI Preview)

Below is an actual screenshot of the running interface:

![Streamlit Interface](Attrition%20Predictor%20Interface%20Screenshot.png)

The dashboard provides:

- **Employee Profile Controls** (age, income, job role, satisfaction, travel, etc.)  
- **Real-time attrition prediction probability**  
- **Workforce analytics visualizations**, including:
  - Attrition rate by department  
  - Age distribution (attrition vs. stay)  
  - Overall attrition benchmarks  

---

## 📂 Repository Structure

employee-attrition-prediction-ML-trainer/
│
├── app.py # Streamlit interface
├── train_model.py # ML training pipeline
├── data/ # Folder for local dataset (ignored by Git)
├── Attrition Predictor Interface Screenshot.png
├── .gitignore
└── README.md

yaml
Copy code

**Note:** The Kaggle dataset is intentionally excluded from GitHub for size and licensing reasons. Add it locally inside the `/data` folder to run the project.

---

## 🚀 How to Run Locally

Install requirements:

```bash
pip install -r requirements.txt
Train the model:

bash
Copy code
python train_model.py
Launch the Streamlit app:

bash
Copy code
streamlit run app.py
🔍 Intended Use
This repository is designed for:

Hiring managers reviewing ML engineering skills

Technical recruiters evaluating Python + ML capability

Data science portfolio demonstration

HR analytics exploratory modeling

It highlights clean code, reproducible ML workflow, and practical model deployment.

📧 Contact
Created by Jesse McClure
For professional review and evaluation.

yaml
Copy code
