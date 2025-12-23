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
st.set_page_config(
    page_title="University AI System",
    layout="wide"
)

st.title("🎓 Global University Application & Management System")
st.caption("Structured like real institutions • AI advises, humans decide")

# =====================================================
# 🌍 UNIVERSITY MASTER LIST
# =====================================================
UNIVERSITIES = [
    # 🇯🇵 Japan
    "University of Tokyo", "Kyoto University", "Osaka University",
    "Tohoku University", "Keio University", "Waseda University",

    # 🇰🇷 South Korea
    "Seoul National University", "KAIST", "POSTECH", "Yonsei University", "Korea University",

    # 🇨🇳 China
    "Tsinghua University", "Peking University", "Fudan University", "Shanghai Jiao Tong University",

    # 🇺🇸 USA
    "MIT", "Harvard University", "Stanford University", "NYU", "NYU Shanghai",
    "UC Berkeley", "UCLA",

    # 🇬🇧 UK
    "University of Oxford", "University of Cambridge", "Imperial College London",
    "UCL", "King's College London",

    # 🇦🇺 Australia
    "University of Melbourne", "Australian National University", "University of Sydney", "Monash University",

    # 🇩🇪 Germany
    "Technical University of Munich", "Heidelberg University", "RWTH Aachen",

    # 🇵🇰 Pakistan
    "NUMS", "NUML", "Riphah International University",

    # 🌍 Catch-all
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
    st.caption("Students apply. No decisions made here.")
    with st.form("application_form"):
        name = st.text_input("Full Name")
        university = st.selectbox("Target University", UNIVERSITIES)
        gpa = st.slider("GPA", 2.0, 4.0, 3.0)
        test_score = st.slider("Test Score", 900, 1600, 1100)
        submit = st.form_submit_button("Submit Application")
    if submit:
        new_id = applications["applicant_id"].max() + 1
        applications.loc[len(applications)] = {
            "applicant_id": new_id,
            "name": name,
            "target_university": university,
            "gpa": gpa,
            "test_score": test_score,
            "status": "Pending"
        }
        st.success("✅ Application submitted. Await admin decision.")

# =====================================================
# ADMIN – APPLICATION REVIEW
# =====================================================
elif menu == "Admin – Applications Review":
    st.subheader("🛡️ Admin Application Review Panel")
    st.caption("Authority lives here. AI advises. Admin decides.")
    pending = applications[applications["status"] == "Pending"]
    st.dataframe(pending, use_container_width=True)
    if not pending.empty:
        applicant_id = st.selectbox("Select Applicant ID", pending["applicant_id"])
        decision = st.radio("Final Decision", ["Accept", "Reject"])
        if st.button("Finalize Decision"):
            applications.loc[applications["applicant_id"] == applicant_id, "status"] = decision
            st.success(f"Application {decision}ed successfully.")
    else:
        st.info("No pending applications.")

# =====================================================
# ADMISSIONS AI (ADVISORY ONLY)
# =====================================================
elif menu == "Admissions AI":
    st.subheader("🧠 Admissions AI – Advisory Engine")
    st.caption("This does NOT auto-accept. It informs judgment.")
    X = applications[["gpa", "test_score"]]
    y = (applications["status"] == "Accept").astype(int)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    gpa = st.slider("Applicant GPA", 2.0, 4.0, 3.2)
    test_score = st.slider("Test Score", 900, 1600, 1200)
    prob = model.predict_proba([[gpa, test_score]])[0][1]
    st.success(f"AI Acceptance Likelihood: {prob * 100:.2f}%")

# =====================================================
# STUDENT MANAGEMENT
# =====================================================
elif menu == "Student Management":
    st.subheader("👨‍🎓 Student Information System")
    st.dataframe(students, use_container_width=True)

# =====================================================
# STUDENT RISK PREDICTION
# =====================================================
elif menu == "Student Risk Prediction":
    st.subheader("⚠️ At-Risk Student Detection")
    X = students[["attendance", "grades", "discipline_score", "engagement"]]
    y = (students["grades"] < 50).astype(int)
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    sid = st.selectbox("Select Student ID", students["student_id"])
    row = students[students["student_id"] == sid][X.columns]
    risk = model.predict(row)[0]
    if risk:
        st.error("⚠️ Student is AT RISK")
    else:
        st.success("✅ Student is Performing Normally")

# =====================================================
# RESOURCE OPTIMIZATION
# =====================================================
elif menu == "Resource Optimization":
    st.subheader("📈 Course Demand Forecast")
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
    st.subheader("💼 Placement Probability Engine")
    X = students[["grades", "engagement", "attendance"]]
    y = students["placed"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    grade = st.slider("Grades", 40, 100, 75)
    engagement = st.slider("Engagement", 1, 10, 7)
    attendance = st.slider("Attendance (%)", 50, 100, 85)
    prob = model.predict_proba([[grade, engagement, attendance]])[0][1]
    st.success(f"Placement Probability: {prob * 100:.2f}%")

