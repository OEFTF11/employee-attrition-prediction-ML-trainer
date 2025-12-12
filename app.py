# app.py
#
# Streamlit app: TechNova HR Analytics – Employee Attrition Predictor
# Uses the trained pipeline saved by train_model.py and the IBM HR dataset
# to provide both descriptive analytics and a predictive attrition risk score.

import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

DATA_PATH = os.path.join("data", "WA_Fn-UseC_-HR-Employee-Attrition.csv")
MODEL_PATH = os.path.join("models", "employee_attrition_model.joblib")


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Attrition_Flag"] = df["Attrition"].map({"Yes": 1, "No": 0})
    return df


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    return model


def main():
    st.set_page_config(
        page_title="TechNova HR Analytics – Employee Attrition Predictor",
        layout="wide",
    )

    st.title("TechNova HR Analytics – Employee Attrition Predictor")

    st.markdown(
        """
        This tool uses a machine-learning model trained on IBM's HR Analytics Employee Attrition &
        Performance dataset to estimate whether an employee is likely to **stay** or **voluntarily leave (attrition)**.

        Use the controls in the sidebar to enter an employee profile,
        then click **Predict Attrition Risk**.
        """
    )

    # Load data & model
    df = load_data()
    model = load_model()

    # -----------------------------
    # Sidebar – employee profile
    # -----------------------------
    st.sidebar.header("Employee Profile")

    age = st.sidebar.slider("Age", int(df["Age"].min()), int(df["Age"].max()), 35)
    monthly_income = st.sidebar.slider(
        "Monthly Income",
        int(df["MonthlyIncome"].min()),
        int(df["MonthlyIncome"].max()),
        5000,
        step=100,
    )
    distance_from_home = st.sidebar.slider(
        "Distance From Home (miles)",
        int(df["DistanceFromHome"].min()),
        int(df["DistanceFromHome"].max()),
        5,
    )
    years_at_company = st.sidebar.slider(
        "Years at Company",
        int(df["YearsAtCompany"].min()),
        int(df["YearsAtCompany"].max()),
        3,
    )

    business_travel = st.sidebar.selectbox(
        "Business Travel",
        sorted(df["BusinessTravel"].unique()),
    )

    department = st.sidebar.selectbox(
        "Department",
        sorted(df["Department"].unique()),
    )

    job_role = st.sidebar.selectbox(
        "Job Role",
        sorted(df["JobRole"].unique()),
    )

    marital_status = st.sidebar.selectbox(
        "Marital Status",
        sorted(df["MaritalStatus"].unique()),
    )

    overtime = st.sidebar.selectbox(
        "OverTime",
        sorted(df["OverTime"].unique()),
    )

    job_satisfaction = st.sidebar.slider(
        "Job Satisfaction (1 = low, 4 = high)",
        int(df["JobSatisfaction"].min()),
        int(df["JobSatisfaction"].max()),
        3,
    )

    if st.sidebar.button("Predict Attrition Risk"):
        input_row = pd.DataFrame(
            [
                {
                    "Age": age,
                    "MonthlyIncome": monthly_income,
                    "DistanceFromHome": distance_from_home,
                    "YearsAtCompany": years_at_company,
                    "BusinessTravel": business_travel,
                    "Department": department,
                    "JobRole": job_role,
                    "MaritalStatus": marital_status,
                    "OverTime": overtime,
                    "JobSatisfaction": job_satisfaction,
                }
            ]
        )

        proba = model.predict_proba(input_row)[0][1]  # probability of attrition = 1
        pred = model.predict(input_row)[0]

        risk_pct = proba * 100

        if risk_pct < 20:
            risk_label = "Low"
        elif risk_pct < 50:
            risk_label = "Moderate"
        else:
            risk_label = "High"

        st.subheader("Predicted Attrition Risk")
        st.write(f"**Probability of Attrition:** {risk_pct:.1f}%")
        st.write(f"**Risk Level:** {risk_label}")

        status_text = "Attrition (likely to leave)" if pred == 1 else "Likely to stay"
        st.write(f"**Predicted Class:** {status_text}")

    # -----------------------------
    # Descriptive analytics section
    # -----------------------------
    st.markdown("---")
    st.subheader("Workforce Attrition Overview")

    col1, col2, col3 = st.columns(3)

    overall_attrition_rate = df["Attrition_Flag"].mean() * 100
    with col1:
        st.metric("Overall Attrition Rate", f"{overall_attrition_rate:.1f}%")

    # Attrition rate by Department
    with col2:
        st.markdown("**Attrition Rate by Department**")
        dept_attrition = (
            df.groupby("Department")["Attrition_Flag"].mean().sort_values() * 100
        )
        fig, ax = plt.subplots()
        dept_attrition.plot(kind="barh", ax=ax)
        ax.set_xlabel("Attrition Rate (%)")
        st.pyplot(fig)

    # Age distribution: Attrition vs No Attrition
    with col3:
        st.markdown("**Age Distribution – Attrition vs Stay**")
        fig2, ax2 = plt.subplots()
        df[df["Attrition_Flag"] == 1]["Age"].plot(
            kind="hist", alpha=0.6, label="Attrition", ax=ax2
        )
        df[df["Attrition_Flag"] == 0]["Age"].plot(
            kind="hist", alpha=0.6, label="Stay", ax=ax2
        )
        ax2.set_xlabel("Age")
        ax2.legend()
        st.pyplot(fig2)

    st.markdown(
        """
        **Note:** These charts are based on the historical IBM HR dataset and provide descriptive insight into
        where attrition is most common (e.g., departments, age ranges). The prediction above uses the same
        underlying model to score individual employee profiles.
        """
    )


if __name__ == "__main__":
    main()
