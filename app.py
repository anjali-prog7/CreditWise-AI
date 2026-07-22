import streamlit as st
import pandas as pd
import numpy as np
import joblib


# ===============================
# LOAD MODEL FILES
# ===============================

model = joblib.load("credit_model.pkl")
scaler = joblib.load("scaler.pkl")


# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="CreditWise AI",
    page_icon="🏦",
    layout="wide"
)


# ===============================
# CUSTOM CSS
# ===============================

st.markdown("""
<style>

body{
    background:#f4f7fb;
}

.main{
    background:#f4f7fb;
}


.title{
    text-align:center;
    font-size:45px;
    font-weight:800;
    color:#123C8C;
}


.subtitle{
    text-align:center;
    font-size:20px;
    color:#666;
}


.card{

    background:white;
    padding:25px;
    border-radius:18px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.08);
    text-align:center;

}


.section{

    font-size:25px;
    font-weight:bold;
    color:#123C8C;

}


.stButton button{

    width:100%;
    height:55px;
    border-radius:12px;
    background:#123C8C;
    color:white;
    font-size:20px;
    font-weight:bold;

}


.stButton button:hover{

    background:#0b2557;
}


</style>

""",unsafe_allow_html=True)



# ===============================
# HEADER
# ===============================


st.markdown(
"""
<h1 class="title">
🏦 CreditWise AI
</h1>

<p class="subtitle">
Smart Loan Approval Prediction System
</p>

""",
unsafe_allow_html=True
)


st.divider()



# ===============================
# SIDEBAR
# ===============================

with st.sidebar:


    st.image(
        "https://cdn-icons-png.flaticon.com/512/2830/2830284.png",
        width=120
    )


    st.title("CreditWise")


    st.info(
    """
    AI based loan analysis platform.

    ✔ Applicant Analysis

    ✔ Risk Assessment

    ✔ Approval Prediction

    """
    )


    st.write("---")


    st.write("Model : Logistic Regression")

    st.write("Domain : Banking AI")



# ===============================
# DASHBOARD CARDS
# ===============================


c1,c2,c3 = st.columns(3)


with c1:

    st.markdown(
    """
    <div class="card">

    <h2>👤</h2>

    <h3>Applicant</h3>

    Customer Information

    </div>
    """,
    unsafe_allow_html=True
    )


with c2:

    st.markdown(
    """
    <div class="card">

    <h2>🤖</h2>

    <h3>AI Model</h3>

    ML Based Decision

    </div>
    """,
    unsafe_allow_html=True
    )


with c3:

    st.markdown(
    """
    <div class="card">

    <h2>📊</h2>

    <h3>Risk Score</h3>

    Approval Probability

    </div>
    """,
    unsafe_allow_html=True
    )


st.write("")



# ===============================
# APPLICATION FORM
# ===============================


st.markdown(
"<p class='section'>📝 Loan Application Details</p>",
unsafe_allow_html=True
)



left,right = st.columns(2)



with left:


    applicant_income = st.number_input(
        "Applicant Income",
        min_value=0.0,
        value=50000.0
    )


    coapplicant_income = st.number_input(
        "Coapplicant Income",
        min_value=0.0,
        value=10000.0
    )


    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )


    dependents = st.number_input(
        "Dependents",
        min_value=0,
        max_value=10
    )


    existing_loans = st.number_input(
        "Existing Loans",
        min_value=0
    )


    savings = st.number_input(
        "Savings",
        min_value=0.0
    )


    collateral = st.number_input(
        "Collateral Value",
        min_value=0.0
    )



with right:


    loan_amount = st.number_input(
        "Loan Amount",
        min_value=1000.0
    )


    loan_term = st.number_input(
        "Loan Term (Months)",
        min_value=1
    )


    credit_score = st.slider(
        "Credit Score",
        300,
        900,
        700
    )


    dti_ratio = st.slider(
        "DTI Ratio",
        0.0,
        2.0,
        0.30
    )


    education = st.selectbox(
        "Education Level",
        [
            "Graduate",
            "Not Graduate"
        ]
    )


    employment = st.selectbox(
        "Employment Status",
        [
            "Salaried",
            "Self-employed",
            "Unemployed"
        ]
    )


    marital = st.selectbox(
        "Marital Status",
        [
            "Married",
            "Single"
        ]
    )


    gender = st.selectbox(
        "Gender",
        [
            "Female",
            "Male"
        ]
    )



col1,col2,col3 = st.columns(3)


with col1:

    purpose = st.selectbox(
        "Loan Purpose",
        [
            "Car",
            "Education",
            "Home",
            "Personal"
        ]
    )


with col2:

    property_area = st.selectbox(
        "Property Area",
        [
            "Rural",
            "Semiurban",
            "Urban"
        ]
    )


with col3:

    employer = st.selectbox(
        "Employer Category",
        [
            "Government",
            "MNC",
            "Private",
            "Unemployed"
        ]
    )



st.write("")


predict = st.button(
    "🔍 Analyze Loan Application"
)

# ==========================================
# PREDICTION LOGIC
# ==========================================


if predict:


    # -------------------------------
    # Feature Engineering
    # -------------------------------

    dti_ratio_sq = dti_ratio ** 2

    credit_score_sq = credit_score ** 2

    applicant_income_log = np.log1p(applicant_income)



    # -------------------------------
    # Education Encoding
    # Same as notebook LabelEncoder
    # -------------------------------

    if education == "Graduate":

        education_encoded = 0

    else:

        education_encoded = 1



    # -------------------------------
    # One Hot Encoding
    # Same columns as training
    # -------------------------------


    employment_salaried = (
        1 if employment=="Salaried" else 0
    )


    employment_self = (
        1 if employment=="Self-employed" else 0
    )


    employment_unemployed = (
        1 if employment=="Unemployed" else 0
    )



    marital_single = (
        1 if marital=="Single" else 0
    )



    purpose_car = (
        1 if purpose=="Car" else 0
    )


    purpose_education = (
        1 if purpose=="Education" else 0
    )


    purpose_home = (
        1 if purpose=="Home" else 0
    )


    purpose_personal = (
        1 if purpose=="Personal" else 0
    )



    property_semiurban = (
        1 if property_area=="Semiurban" else 0
    )


    property_urban = (
        1 if property_area=="Urban" else 0
    )



    gender_male = (
        1 if gender=="Male" else 0
    )



    employer_government = (
        1 if employer=="Government" else 0
    )


    employer_mnc = (
        1 if employer=="MNC" else 0
    )


    employer_private = (
        1 if employer=="Private" else 0
    )


    employer_unemployed = (
        1 if employer=="Unemployed" else 0
    )



    # -------------------------------
    # Final DataFrame
    # SAME ORDER AS TRAINING
    # -------------------------------


    input_data = pd.DataFrame([{

        "Applicant_Income":
        applicant_income,


        "Coapplicant_Income":
        coapplicant_income,


        "Age":
        age,


        "Dependents":
        dependents,


        "Existing_Loans":
        existing_loans,


        "Savings":
        savings,


        "Collateral_Value":
        collateral,


        "Loan_Amount":
        loan_amount,


        "Loan_Term":
        loan_term,


        "Education_Level":
        education_encoded,


        "Employment_Status_Salaried":
        employment_salaried,


        "Employment_Status_Self-employed":
        employment_self,


        "Employment_Status_Unemployed":
        employment_unemployed,


        "Marital_Status_Single":
        marital_single,


        "Loan_Purpose_Car":
        purpose_car,


        "Loan_Purpose_Education":
        purpose_education,


        "Loan_Purpose_Home":
        purpose_home,


        "Loan_Purpose_Personal":
        purpose_personal,


        "Property_Area_Semiurban":
        property_semiurban,


        "Property_Area_Urban":
        property_urban,


        "Gender_Male":
        gender_male,


        "Employer_Category_Government":
        employer_government,


        "Employer_Category_MNC":
        employer_mnc,


        "Employer_Category_Private":
        employer_private,


        "Employer_Category_Unemployed":
        employer_unemployed,


        "DTI_Ratio_sq":
        dti_ratio_sq,


        "Credit_Score_sq":
        credit_score_sq,


        "Applicant_Income_log":
        applicant_income_log

    }])



    # -------------------------------
    # Scaling
    # -------------------------------


    scaled_data = scaler.transform(
        input_data
    )



    # -------------------------------
    # Prediction
    # -------------------------------


    prediction = model.predict(
        scaled_data
    )[0]



    probability = model.predict_proba(
        scaled_data
    )[0][1]



    st.divider()



    # -------------------------------
    # Result UI
    # -------------------------------


    if prediction == 1:


        st.success(
            "🎉 Loan Approved"
        )


        st.metric(
            "Approval Probability",
            f"{probability*100:.2f}%"
        )


        st.balloons()



    else:


        st.error(
            "❌ Loan Rejected"
        )


        st.metric(
            "Approval Probability",
            f"{probability*100:.2f}%"
        )


    st.write("")


    st.info(
        """
        AI Analysis Completed.

        The decision is generated
        using Machine Learning model.
        """
    )