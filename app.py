"""Streamlit portfolio application for employee attrition exploration."""

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from src.config import MODEL_FEATURES, MODEL_PATH, REPORTS_DIR
from src.data import load_dataset

st.set_page_config(page_title="Attrition Intelligence Lab", page_icon="📊", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background: linear-gradient(180deg, #07111f 0%, #0b1727 35%, #101a2a 100%);}
    [data-testid="stSidebar"] {background: #0a1423; border-right: 1px solid #22324a;}
    .hero {padding: 2rem; border: 1px solid #263954; border-radius: 20px;
           background: linear-gradient(135deg, rgba(24,59,92,.95), rgba(12,27,45,.95)); margin-bottom: 1rem;}
    .eyebrow {color:#62d9c6; font-weight:700; letter-spacing:.12em; text-transform:uppercase; font-size:.8rem;}
    .hero h1 {font-size:2.6rem; margin:.35rem 0; color:#f6f8fb;}
    .hero p {color:#bdcad9; max-width:850px; font-size:1.05rem;}
    .risk-card {padding:1.4rem; border-radius:16px; background:#122238; border:1px solid #2a4261;}
    .small-note {color:#9fb0c4; font-size:.88rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def data() -> pd.DataFrame:
    return load_dataset()


@st.cache_resource
def artifact() -> dict[str, object]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("The trained model is missing. Run `python train_model.py` first.")
    loaded = joblib.load(MODEL_PATH)
    if not isinstance(loaded, dict) or "model" not in loaded:
        raise ValueError("The model artifact is outdated. Run `python train_model.py` again.")
    return loaded


def report_csv(name: str) -> pd.DataFrame:
    path = REPORTS_DIR / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def employee_form(frame: pd.DataFrame) -> tuple[bool, pd.DataFrame]:
    st.sidebar.header("Employee profile")
    st.sidebar.caption("Choose values or load the included example profile.")
    example = st.sidebar.toggle("Use higher-risk example", value=False)
    defaults = {"Age": 28 if example else 35, "MonthlyIncome": 2800 if example else 5000,
                "DistanceFromHome": 18 if example else 5, "YearsAtCompany": 1 if example else 3,
                "JobSatisfaction": 1 if example else 3, "OverTime": "Yes" if example else "No"}
    with st.sidebar.form("employee"):
        age = st.slider("Age", int(frame.Age.min()), int(frame.Age.max()), defaults["Age"])
        income = st.slider("Monthly income", int(frame.MonthlyIncome.min()), int(frame.MonthlyIncome.max()), defaults["MonthlyIncome"], step=100)
        distance = st.slider("Distance from home (miles)", int(frame.DistanceFromHome.min()), int(frame.DistanceFromHome.max()), defaults["DistanceFromHome"])
        tenure = st.slider("Years at company", int(frame.YearsAtCompany.min()), int(frame.YearsAtCompany.max()), defaults["YearsAtCompany"])
        satisfaction = st.slider("Job satisfaction", 1, 4, defaults["JobSatisfaction"], help="1 = low, 4 = very high")
        travel = st.selectbox("Business travel", sorted(frame.BusinessTravel.unique()))
        department = st.selectbox("Department", sorted(frame.Department.unique()))
        role = st.selectbox("Job role", sorted(frame.JobRole.unique()))
        marital = st.selectbox("Marital status", sorted(frame.MaritalStatus.unique()))
        overtime_options = sorted(frame.OverTime.unique())
        overtime = st.selectbox("Overtime", overtime_options, index=overtime_options.index(defaults["OverTime"]))
        submitted = st.form_submit_button("Estimate attrition risk", use_container_width=True, type="primary")
    row = pd.DataFrame([{"Age": age, "MonthlyIncome": income, "DistanceFromHome": distance,
                         "YearsAtCompany": tenure, "JobSatisfaction": satisfaction,
                         "BusinessTravel": travel, "Department": department, "JobRole": role,
                         "MaritalStatus": marital, "OverTime": overtime}])[MODEL_FEATURES]
    return submitted, row


def prediction_panel(bundle: dict[str, object], row: pd.DataFrame) -> None:
    probability = float(bundle["model"].predict_proba(row)[0, 1])
    threshold = float(bundle["threshold"])
    higher = probability >= threshold
    level, color = ("Higher", "#ff886e") if higher else ("Lower", "#62d9c6")
    st.markdown(
        f"""<div class="risk-card"><div class="eyebrow">Model estimate</div>
        <h2 style="color:{color}; margin:.35rem 0">{probability:.1%} · {level} modeled risk</h2>
        <div class="small-note">Decision threshold: {threshold:.1%}. This is a statistical estimate, not a determination about an employee.</div></div>""",
        unsafe_allow_html=True,
    )
    st.progress(min(probability, 1.0))
    with st.expander("Profile used for this estimate"):
        st.dataframe(row.T.rename(columns={0: "Value"}), use_container_width=True)


def main() -> None:
    st.markdown(
        """<section class="hero"><div class="eyebrow">Machine learning portfolio project</div>
        <h1>Attrition Intelligence Lab</h1>
        <p>Explore workforce patterns and a calibrated employee-attrition model built from the IBM HR Analytics sample dataset. The project compares multiple algorithms, tunes the selected model, and documents performance limitations.</p>
        </section>""", unsafe_allow_html=True,
    )
    try:
        frame, bundle = data(), artifact()
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        st.code("python -m pip install -r requirements.txt\npython train_model.py\nstreamlit run app.py", language="bash")
        st.stop()

    submitted, row = employee_form(frame)
    overview, model_tab, transparency = st.tabs(["Workforce overview", "Model performance", "Responsible use"])
    with overview:
        c1, c2, c3 = st.columns(3)
        c1.metric("Employees in sample", f"{len(frame):,}")
        c2.metric("Observed attrition", f"{frame.Attrition_Flag.mean():.1%}")
        c3.metric("Departments", frame.Department.nunique())
        if submitted:
            prediction_panel(bundle, row)
        else:
            st.info("Complete the employee profile in the sidebar and select **Estimate attrition risk**.")
        chart1, chart2 = st.columns(2)
        department = frame.groupby("Department", as_index=False).Attrition_Flag.mean()
        department["Attrition rate"] = department.Attrition_Flag * 100
        chart1.plotly_chart(px.bar(department, x="Attrition rate", y="Department", orientation="h",
            title="Observed attrition by department", color="Attrition rate", color_continuous_scale="Teal"), use_container_width=True)
        chart2.plotly_chart(px.histogram(frame, x="Age", color="Attrition", barmode="overlay", opacity=.65,
            title="Age distribution by outcome", color_discrete_map={"Yes": "#ff886e", "No": "#62d9c6"}), use_container_width=True)

    with model_tab:
        metadata = bundle.get("metadata", {})
        metrics = metadata.get("test_metrics", {})
        st.subheader(str(metadata.get("selected_model", "Selected model")))
        cols = st.columns(4)
        for column, key, label in zip(
            cols,
            ["roc_auc", "pr_auc", "recall", "precision"],
            ["ROC-AUC", "PR-AUC", "Recall", "Precision"],
            strict=True,
        ):
            column.metric(label, f"{float(metrics.get(key, 0)):.3f}")
        comparison, importance = report_csv("model_comparison.csv"), report_csv("feature_importance.csv")
        if not comparison.empty:
            st.markdown("#### Candidate model comparison")
            st.dataframe(comparison, hide_index=True, use_container_width=True)
        if not importance.empty:
            top = importance.sort_values("importance_mean").tail(10)
            st.plotly_chart(px.bar(top, x="importance_mean", y="feature", orientation="h", error_x="importance_std",
                title="Permutation importance on the held-out test set"), use_container_width=True)

    with transparency:
        st.subheader("Responsible-use boundary")
        st.warning("This demonstration is for education and portfolio review. It must not be used to make employment decisions or label real employees.")
        st.markdown("""
        The dataset is a small public sample and does not represent a specific workforce. Predictions show correlations, not causes. Age and marital status are retained to reproduce the original project design; because demographic inputs can encode unequal patterns, the project reports subgroup performance and requires human review in any hypothetical application.

        False positives may unfairly characterize employees, while false negatives may miss people who need support. A real deployment would require legal review, employee consultation, stronger data governance, external validation, and ongoing bias monitoring.
        """)
        subgroup = report_csv("subgroup_metrics.csv")
        if not subgroup.empty:
            st.markdown("#### Held-out subgroup diagnostics")
            st.dataframe(subgroup, hide_index=True, use_container_width=True)
    st.caption("Built with Python, scikit-learn, Streamlit, and Plotly · IBM HR Analytics sample data")


if __name__ == "__main__":
    main()
