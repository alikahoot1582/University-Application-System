# =====================================================
# 🎓 Global University Application System
# =====================================================
# Clean MVP • Streamlit • AI-Advisory • Admin-Controlled
# =====================================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="University Application System",
    layout="wide"
)

st.title("🎓 Global University Application System")
st.caption("AI advises • Humans decide • Decisions stay accountable")

# =====================================================
# 🌍 UNIVERSITY MASTER LIST
# =====================================================
UNIVERSITIES = [
    "University of Tokyo", "Kyoto University",
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
# SESSION DATA (SAFE INIT)
# =====================================================
np.random.seed(42)

if "applications" not in st.session_state:
    st.session_state.applications = pd.DataFrame({
        "applicant_id": range(1, 31),
        "name": [f"Applicant {i}" for i in range(1, 31)],
        "target_university": np.random.choice(UNIVERSITIES, 30),
        "gpa": np.round(np.random.uniform(2.0, 4.0, 30), 2),
        "test_score": np.random.randint(900, 1600, 30),
        "status": ["Pending"] * 30
    })

applications = st.session_state.applications

# =====================================================
# SIDEBAR NAVIGATION (ONLY 4 PAGES)
# =====================================================
menu = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Student Application Portal",
        "Admin – Applications Review",
        "Admissions AI (Advisory)"
    ]
)

# =====================================================
# DASHBOARD
# =====================================================
if menu == "Dashboard":
    st.subheader("📊 System Overview")

    col1, col2 = st.columns(2)
    col1.metric("Total Applications", len(applications))
    col2.metric(
        "Pending Applications",
        (applications["status"] == "Pending").sum()
    )

    st.divider()
    st.subheader("📄 Application Status Breakdown")
    status_counts = applications["status"].value_counts()
    st.bar_chart(status_counts)

# =====================================================
# STUDENT APPLICATION PORTAL
# =====================================================
elif menu == "Student Application Portal":
    st.subheader("📝 Apply to a University")
    st.info("Submit your application. Final decisions are made by administrators.")

    with st.form("application_form"):
        name = st.text_input("Full Name")
        university = st.selectbox("Target University", UNIVERSITIES)
        gpa = st.slider("GPA", 2.0, 4.0, 3.0)
        test_score = st.slider("Test Score", 900, 1600, 1100)
        submit = st.form_submit_button("Submit Application")

    if submit:
        if not name.strip():
            st.error("Full name is required.")
        else:
            new_id = applications["applicant_id"].max() + 1
            applications.loc[len(applications)] = [
                new_id, name, university, gpa, test_score, "Pending"
            ]
            st.success("✅ Application submitted successfully")

# =====================================================
# ADMIN – APPLICATION REVIEW
# =====================================================
elif menu == "Admin – Applications Review":
    st.subheader("🛡️ Admin Review Panel")
    st.caption("Admins make the final decision. AI does not auto-accept.")

    pending = applications[applications["status"] == "Pending"]

    if pending.empty:
        st.success("No pending applications.")
    else:
        st.dataframe(pending, use_container_width=True)

        applicant_id = st.selectbox(
            "Select Applicant ID",
            pending["applicant_id"]
        )
        decision = st.radio("Decision", ["Accept", "Reject"])

        if st.button("Finalize Decision"):
            applications.loc[
                applications["applicant_id"] == applicant_id, "status"
            ] = decision
            st.success(f"Application {decision}ed")

# =====================================================
# ADMISSIONS AI (ADVISORY ONLY)
# =====================================================
elif menu == "Admissions AI (Advisory)":
    st.subheader("🧠 Admissions AI – Advisory Only")
    st.caption(
        "This model provides guidance based on historical decisions. "
        "It does not make final decisions."
    )

    X = applications[["gpa", "test_score"]]
    y = (applications["status"] == "Accept").astype(int)

    st.write(
        f"Accepted: {y.sum()} | "
        f"Reviewed: {(applications['status'] != 'Pending').sum()}"
    )

    if y.nunique() < 2:
        st.warning(
            "AI requires both accepted and rejected cases to learn. "
            "Please review more applications."
        )
    else:
        model = RandomForestClassifier(random_state=42)
        model.fit(X, y)

        gpa = st.slider("Applicant GPA", 2.0, 4.0, 3.2)
        test_score = st.slider("Test Score", 900, 1600, 1200)

        probs = model.predict_proba([[gpa, test_score]])[0]
        accept_idx = list(model.classes_).index(1)

        st.success(
            f"Estimated Acceptance Likelihood: "
            f"{probs[accept_idx] * 100:.2f}%"
        )
