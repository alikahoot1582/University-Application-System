# =====================================================
# 🎓 Global University Application & Management System
# =====================================================
# Production-Safe • Streamlit • ML-Powered • Human-In-Control
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ------------------ CONFIG ------------------
st.set_page_config(page_title="University AI System", layout="wide")

st.title("🎓 Global University Application & Management System")
st.caption("AI advises • Humans decide • Systems stay safe")

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
# SESSION DATA INITIALIZATION
# =====================================================
np.random.seed(42)

if "students" not in st.session_state:
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

    st.divider()
    st.subheader("📚 Course Enrollment Snapshot")
    st.bar_chart(courses.set_index("course"))

# =====================================================
# STUDENT APPLICATION PORTAL
# =====================================================
elif menu == "Student Application Portal":
    st.subheader("🎓 University Application Portal")
    st.info("This portal submits applications only. Decisions are made by admins.")

    with st.form("application_form"):
        name = st.text_input("Full Name")
        university = st.selectbox("Target University", UNIVERSITIES)
        gpa = st.slider("GPA", 2.0, 4.0, 3.0)
        test_score = st.slider("Test Score", 900, 1600, 1100)
        submitted = st.form_submit_button("Submit Application")

    if submitted:
        if not name.strip():
            st.error("Name cannot be empty.")
        else:
            new_id = applications["applicant_id"].max() + 1
            applications.loc[len(applications)] = [
                new_id, name, university, gpa, test_score, "Pending"
            ]
            st.success("✅ Application successfully submitted")

# =====================================================
# ADMIN – APPLICATION REVIEW
# =====================================================
elif menu == "Admin – Applications Review":
    st.subheader("🛡️ Admin Review Panel")
    st.info("Admins make final decisions. AI does not auto-approve.")

    pending = applications[applications["status"] == "Pending"]

    if pending.empty:
        st.success("No pending applications 🎉")
    else:
        st.dataframe(pending, use_container_width=True)
        applicant_id = st.selectbox("Select Applicant ID", pending["applicant_id"])
        decision = st.radio("Decision", ["Accept", "Reject"])

        if st.button("Finalize Decision"):
            applications.loc[
                applications["applicant_id"] == applicant_id, "status"
            ] = decision
            st.success(f"Application {decision}ed successfully")

# =====================================================
# ADMISSIONS AI
# =====================================================
elif menu == "Admissions AI":
    st.subheader("🧠 Admissions AI (Advisory Only)")
    st.caption("This model assists judgment — it never decides.")

    X = applications[["gpa", "test_score"]]
    y = (applications["status"] == "Accept").astype(int)

    st.write(f"Accepted cases: {y.sum()} | Total reviewed: {len(y)}")

    if y.nunique() < 2:
        st.warning("AI needs both ACCEPTED and REJECTED examples to learn.")
    else:
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        gpa = st.slider("Applicant GPA", 2.0, 4.0, 3.2)
        test_score = st.slider("Test Score", 900, 1600, 1200)

        probs = model.predict_proba([[gpa, test_score]])[0]
        accept_idx = list(model.classes_).index(1)

        st.success(f"Estimated Acceptance Likelihood: {probs[accept_idx] * 100:.2f}%")

# =====================================================
# STUDENT MANAGEMENT
# =====================================================
elif menu == "Student Management":
    st.subheader("👨‍🎓 Student Management System")
    st.write(f"Total student records: {len(students)}")
    st.dataframe(students, use_container_width=True)

# =====================================================
# STUDENT RISK PREDICTION
# =====================================================
elif menu == "Student Risk Prediction":
    st.subheader("⚠️ At-Risk Student Detection")
    st.caption("Risk defined as academic performance below threshold.")

    X = students[["attendance", "grades", "discipline_score", "engagement"]]
    y = (students["grades"] < 50).astype(int)

    st.write(f"At-risk students in dataset: {y.sum()}")

    if y.nunique() < 2:
        st.warning("Insufficient variation to train risk model.")
    else:
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        sid = st.selectbox("Select Student ID", students["student_id"])
        row = students.loc[students["student_id"] == sid, X.columns]
        risk = model.predict(row)[0]

        if risk == 1:
            st.error("⚠️ Student is AT RISK")
        else:
            st.success("✅ Student is Performing Normally")

# =====================================================
# RESOURCE OPTIMIZATION
# =====================================================
elif menu == "Resource Optimization":
    st.subheader("📈 Course Demand Forecasting")
    st.caption("Forecasts future enrollment based on historical patterns.")

    X = np.arange(len(courses)).reshape(-1, 1)
    y = courses["current_enrollment"]

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    courses["forecasted_enrollment"] = model.predict(X).astype(int)

    st.dataframe(courses, use_container_width=True)

# =====================================================
# PLACEMENT PREDICTION
# =====================================================
elif menu == "Placement Prediction":
    st.subheader("💼 Placement Probability Estimator")
    st.caption("Predicts likelihood of placement based on engagement & performance.")

    X = students[["grades", "engagement", "attendance"]]
    y = students["placed"]

    st.write(f"Placed students in dataset: {y.sum()}")

    if y.nunique() < 2:
        st.warning("Not enough placement variation to train model.")
    else:
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        grade = st.slider("Grades", 40, 100, 75)
        engagement = st.slider("Engagement", 1, 10, 7)
        attendance = st.slider("Attendance (%)", 50, 100, 85)

        probs = model.predict_proba([[grade, engagement, attendance]])[0]
        placed_idx = list(model.classes_).index(1)

        st.success(f"Estimated Placement Probability: {probs[placed_idx] * 100:.2f}%")
