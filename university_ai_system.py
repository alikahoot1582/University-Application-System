# =====================================================
# 🎓 Global University Application & Management System
# =====================================================
# MVP • Streamlit • ML-powered • Admin-controlled
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ------------------ CONFIG ------------------
st.set_page_config(page_title="University AI System", layout="wide")

st.title("🎓 Global University Application & Management System")
st.caption("Structured like real institutions • AI advises, humans decide")

# =====================================================
# 🌍 UNIVERSITY MASTER LIST
# =====================================================
UNIVERSITIES = [
    "University of Tokyo", "Kyoto University", "Osaka University",
    "Seoul National University", "KAIST",
    "Tsinghua University", "Peking University",
    "MIT", "Harvard University", "Stanford University",
    "University of Oxford", "University of Cambridge",
    "University of Melbourne",
    "Technical University of Munich",
    "NUMS", "NUML",
    "Other / Partner University"
]

# =====================================================
# SESSION-BASED MOCK DATABASE
# =====================================================
if "students" not in st.session_state:
    np.random.seed(42)
    st.session_state.students = pd.DataFrame({
        "student_id": range(1, 101),
        "attendance": np.random.randint(50, 100, 100),
        "grades": np.random.randint(40, 100, 100),
        "discipline_score": np.random.randint(0, 10, 100),
        "engagement": np.random.randint(1, 10, 100),
        "placed": np.random.choice([0, 1], 100)
    })

if "applications" not in st.session_state:
    st.session_state.applications = pd.DataFrame({
        "applicant_id": range(1, 51),
        "name": [f"Applicant {i}" for i in range(1, 51)],
        "target_university": np.random.choice(UNIVERSITIES, 50),
        "gpa": np.round(np.random.uniform(2.0, 4.0, 50), 2),
        "test_score": np.random.randint(900, 1600, 50),
        "status": ["Pending"] * 50
    })

if "courses" not in st.session_state:
    st.session_state.courses = pd.DataFrame({
        "course": ["CS", "AI", "Data Science", "Engineering", "Business"],
        "current_enrollment": np.random.randint(50, 300, 5)
    })

students = st.session_state.students
applications = st.session_state.applications
courses = st.session_state.courses

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
menu = st.sidebar.radio(
    "System Navigation",
    [
        "Dashboard",
        "Student Application Portal",
        "Admin – Applications Review",
        "Admissions AI",
        "Student Management",
        "Student Risk Prediction",
        "Resource Optimization",
        "Placement Prediction"
    ]
)

# =====================================================
# DASHBOARD
# =====================================================
if menu == "Dashboard":
    st.subheader("📊 System Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Students", len(students))
    col2.metric("Total Applications", len(applications))
    col3.metric("Courses Offered", len(courses))
    st.bar_chart(courses.set_index("course"))

# =====================================================
# STUDENT APPLICATION PORTAL
# =====================================================
elif menu == "Student Application Portal":
    st.subheader("🎓 University Application Portal")

    with st.form("application_form"):
        name = st.text_input("Full Name")
        university = st.selectbox("Target University", UNIVERSITIES)
        gpa = st.slider("GPA", 2.0, 4.0, 3.0)
        test_score = st.slider("Test Score", 900, 1600, 1100)
        submit = st.form_submit_button("Submit Application")

    if submit and name:
        new_id = applications["applicant_id"].max() + 1
        applications.loc[len(applications)] = [
            new_id, name, university, gpa, test_score, "Pending"
        ]
        st.success("✅ Application submitted")

# =====================================================
# ADMIN – APPLICATION REVIEW
# =====================================================
elif menu == "Admin – Applications Review":
    st.subheader("🛡️ Admin Application Review Panel")

    pending = applications[applications["status"] == "Pending"]
    st.dataframe(pending, use_container_width=True)

    if not pending.empty:
        applicant_id = st.selectbox("Select Applicant ID", pending["applicant_id"])
        decision = st.radio("Final Decision", ["Accept", "Reject"])

        if st.button("Finalize Decision"):
            applications.loc[
                applications["applicant_id"] == applicant_id, "status"
            ] = decision
            st.success(f"Application {decision}ed")

# =====================================================
# ADMISSIONS AI (SAFE)
# =====================================================
elif menu == "Admissions AI":
    st.subheader("🧠 Admissions AI – Advisory Engine")

    X = applications[["gpa", "test_score"]]
    y = (applications["status"] == "Accept").astype(int)

    if y.nunique() < 2:
        st.warning("AI needs at least one ACCEPT and one REJECT decision.")
    else:
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        gpa = st.slider("Applicant GPA", 2.0, 4.0, 3.2)
        test_score = st.slider("Test Score", 900, 1600, 1200)

        probs = model.predict_proba([[gpa, test_score]])[0]
        accept_idx = list(model.classes_).index(1)

        st.success(f"AI Acceptance Likelihood: {probs[accept_idx] * 100:.2f}%")

# =================================
