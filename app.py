import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# ================= PAGE CONFIGURATION =================
st.set_page_config(
    page_title="Customer Churn Intelligence Portal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================= PROFESSIONAL STYLING =================
st.markdown("""
    <style>
    .stApp {
        background: #f8fafc;
        color: #1e293b;
    }
    .portal-title {
        font-size: 2.8rem !important;
        font-weight: 900 !important;
        color: #1e3a8a !important;
        margin-bottom: 0px;
    }
    .section-heading {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3a8a;
        margin-bottom: 0.3rem;
    }
    .sub-heading {
        color: #475569;
        font-size: 1.15rem;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }
    .banner-danger {
        background-color: #fee2e2;
        border-left: 6px solid #ef4444;
        padding: 18px 22px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .banner-success {
        background-color: #dcfce7;
        border-left: 6px solid #10b981;
        padding: 18px 22px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    .stTextInput label, .stNumberInput label, .stSelectbox label {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    .stTextInput input, .stNumberInput input, div[data-baseweb="select"] > div, div[data-baseweb="input"] {
        border-radius: 8px !important;
        border: 1px solid #94a3b8 !important;
        background-color: #ffffff !important;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        border: none;
        box-shadow: 0 2px 6px rgba(59, 130, 246, 0.3);
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# ================= LOAD DATA & MODEL =================
@st.cache_data
def load_data():
    if os.path.exists("WA_Fn-UseC_-Telco-Customer-Churn.csv"):
        df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        return df
    return None

@st.cache_resource
def load_model():
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            return pickle.load(f)
    return None

data = load_data()
model = load_model()

# ================= NAVIGATION =================
if 'page' not in st.session_state:
    st.session_state.page = "Home"

def set_page(p):
    st.session_state.page = p

col_logo, col_nav1, col_nav2, col_nav3 = st.columns([2.4, 1.2, 1.2, 1.3])
with col_logo:
    st.markdown('<p class="portal-title">⚡ Telecom Churn Portal</p>', unsafe_allow_html=True)
with col_nav1:
    if st.button("🏠 Home", use_container_width=True):
        set_page("Home")
        st.rerun()
with col_nav2:
    if st.button("🔮 Prediction", use_container_width=True):
        set_page("Prediction Engine")
        st.rerun()
with col_nav3:
    if st.button("📊 EDA Dashboard", use_container_width=True):
        set_page("EDA Dashboard")
        st.rerun()

st.markdown("<hr style='margin-top: 10px; margin-bottom: 25px; border-color: #cbd5e1;'>", unsafe_allow_html=True)

# ================= PAGES =================
if st.session_state.page == "Home":
    st.markdown('<p class="section-heading">Customer Churn Prediction System</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-heading">A machine learning platform designed to analyze subscriber behavior and predict customer churn with high precision.</p>', unsafe_allow_html=True)
    
    if data is not None:
        # 1. Project Highlights & Metrics Box
        with st.container(border=True):
            st.markdown("### 📊 Project Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Subscribers", f"{len(data):,}")
            m2.metric("Features Analyzed", f"{data.shape[1]}")
            m3.metric("ML Algorithm", "Random Forest")
            m4.metric("System Status", "Operational ⚡")
        
        # 2. Action Buttons Right After the Metrics Box
        st.markdown("<br>", unsafe_allow_html=True)
        hb1, hb2 = st.columns(2)
        with hb1:
            if st.button("🔮 Launch Prediction Engine", use_container_width=True):
                set_page("Prediction Engine")
                st.rerun()
        with hb2:
            if st.button("📊 Explore EDA Dashboard", use_container_width=True):
                set_page("EDA Dashboard")
                st.rerun()
        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Dataset Preview Box
        with st.container(border=True):
            st.subheader("📋 Dataset Preview")
            st.dataframe(data.head(100), use_container_width=True)

elif st.session_state.page == "Prediction Engine":
    st.markdown('<p class="section-heading">🔮 Customer Churn Prediction Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-heading">Enter customer specifications below to evaluate real-time attrition risk accurately.</p>', unsafe_allow_html=True)

    if model is not None and data is not None:
        temp_data = data.copy()
        if hasattr(model, 'feature_names_in_'):
            expected_cols = model.feature_names_in_
        else:
            expected_cols = temp_data.drop("Churn", axis=1).columns

        for col in temp_data.select_dtypes(include='object').columns:
            if col != 'Churn':
                le = LabelEncoder()
                temp_data[col] = le.fit_transform(temp_data[col].astype(str))

        with st.container(border=True):
            with st.form("prediction_form"):
                st.subheader("Customer Specification Form")
                c1, c2, c3 = st.columns(3)
                with c1:
                    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=72, value=1)
                    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=95.0)
                with c2:
                    total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=95.0)
                    senior_citizen = st.selectbox("Senior Citizen (0: No, 1: Yes)", [0, 1])
                with c3:
                    paperless = st.selectbox("Paperless Billing (0: No, 1: Yes)", [0, 1])
                    contract_type = st.selectbox("Contract Type (0: Month-to-month, 1: One year, 2: Two year)", [0, 1, 2])

                submitted = st.form_submit_button("Predict Churn Status 🚀")

        if submitted:
            new_customer = temp_data.drop("Churn", axis=1).iloc[[0]].copy()
            if 'tenure' in new_customer.columns: new_customer['tenure'] = tenure
            if 'MonthlyCharges' in new_customer.columns: new_customer['MonthlyCharges'] = monthly_charges
            if 'TotalCharges' in new_customer.columns: new_customer['TotalCharges'] = total_charges
            if 'SeniorCitizen' in new_customer.columns: new_customer['SeniorCitizen'] = senior_citizen
            if 'PaperlessBilling' in new_customer.columns: new_customer['PaperlessBilling'] = paperless
            if 'Contract' in new_customer.columns: new_customer['Contract'] = contract_type

            for col in expected_cols:
                if col not in new_customer.columns:
                    new_customer[col] = 0
            new_customer = new_customer[expected_cols]

            prediction = model.predict(new_customer)
            probability = model.predict_proba(new_customer)
            
            churn_prob = probability[0][1]
            stay_prob = probability[0][0]
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 📈 Executive Prediction Results & Analytics")
            
            if prediction[0] == 1 or churn_prob >= 0.40:
                st.markdown("""
                <div class="banner-danger">
                    <h3 style="color: #b91c1c; margin: 0 0 5px 0;">⚠️ Churn Prediction: Customer Likely to Leave</h3>
                    <p style="color: #7f1d1d; margin: 0; font-size: 1.05rem;">This subscriber exhibits critical behavioral indicators pointing toward potential service cancellation.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="banner-success">
                    <h3 style="color: #047857; margin: 0 0 5px 0;">✅ Retention Prediction: Customer Likely to Stay</h3>
                    <p style="color: #065f46; margin: 0; font-size: 1.05rem;">This subscriber demonstrates stable engagement patterns and strong account retention.</p>
                </div>
                """, unsafe_allow_html=True)
                
            res_c1, res_c2 = st.columns(2)
            with res_c1:
                with st.container(border=True):
                    st.markdown("<h4 style='color: #1e3a8a; margin-top: 0;'>Probability Breakdown</h4>", unsafe_allow_html=True)
                    mc1, mc2 = st.columns(2)
                    mc1.metric("Churn Probability", f"{churn_prob*100:.2f}%")
                    mc2.metric("Stay Probability", f"{stay_prob*100:.2f}%")
                    st.progress(float(churn_prob))
            with res_c2:
                with st.container(border=True):
                    st.markdown("<h4 style='color: #1e3a8a; margin-top: 0;'>Strategic Recommendation</h4>", unsafe_allow_html=True)
                    if prediction[0] == 1 or churn_prob >= 0.40:
                        st.markdown("<ul><li>Offer targeted retention discounts & loyalty benefits.</li><li>Provide proactive customer service outreach.</li></ul>", unsafe_allow_html=True)
                    else:
                        st.markdown("<ul><li>Maintain standard engagement and touchpoints.</li><li>Explore cross-selling premium add-on services.</li></ul>", unsafe_allow_html=True)

elif st.session_state.page == "EDA Dashboard":
    st.markdown('<p class="section-heading">📊 Exploratory Data Analysis Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-heading">Visualizing key trends, distributions, and behavioral patterns from the telco dataset.</p>', unsafe_allow_html=True)
    
    if data is not None:
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("<h4 style='color: #1e3a8a; margin-top: 0; margin-bottom: 15px;'>Overall Churn Distribution</h4>", unsafe_allow_html=True)
                fig1, ax1 = plt.subplots(figsize=(6, 4))
                sns.countplot(x='Churn', data=data, palette=['#3b82f6', '#ef4444'], ax=ax1)
                st.pyplot(fig1)
                plt.close(fig1)
            
        with col2:
            with st.container(border=True):
                st.markdown("<h4 style='color: #1e3a8a; margin-top: 0; margin-bottom: 15px;'>Churn by Contract Type</h4>", unsafe_allow_html=True)
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                sns.countplot(x='Contract', hue='Churn', data=data, palette=['#10b981', '#f59e0b'], ax=ax2)
                st.pyplot(fig2)
                plt.close(fig2)
            
        col3, col4 = st.columns(2)
        with col3:
            with st.container(border=True):
                st.markdown("<h4 style='color: #1e3a8a; margin-top: 0; margin-bottom: 15px;'>Churn by Payment Method</h4>", unsafe_allow_html=True)
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                sns.countplot(x='PaymentMethod', hue='Churn', data=data, palette=['#6366f1', '#ec4899'], ax=ax3)
                plt.xticks(rotation=20, ha='right')
                st.pyplot(fig3)
                plt.close(fig3)
            
        with col4:
            with st.container(border=True):
                st.markdown("<h4 style='color: #1e3a8a; margin-top: 0; margin-bottom: 15px;'>Tenure vs Monthly Charges</h4>", unsafe_allow_html=True)
                fig4, ax4 = plt.subplots(figsize=(6, 4))
                sns.scatterplot(x='tenure', y='MonthlyCharges', hue='Churn', data=data, palette=['#3b82f6', '#ef4444'], alpha=0.6, ax=ax4)
                st.pyplot(fig4)
                plt.close(fig4)