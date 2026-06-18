"""
Loan Default Prediction Web Application
Main Streamlit app with user authentication, transactions, and predictions
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from mongo_manager import MongoDBManager
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(
    page_title="Loan Default Prediction System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
        .main { padding: 2rem; }
        .metric-card { 
            padding: 20px; 
            background-color: #f0f2f6; 
            border-radius: 10px;
            margin: 10px 0;
        }
        .success-box { 
            padding: 15px; 
            background-color: #d4edda; 
            color: #155724;
            border-radius: 5px;
            margin: 10px 0;
        }
        .error-box { 
            padding: 15px; 
            background-color: #f8d7da; 
            color: #721c24;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.user_email = None
    st.session_state.user_data = None

# Initialize MongoDB Manager
@st.cache_resource
def init_mongodb():
    return MongoDBManager()

db = init_mongodb()

# Load Model
@st.cache_resource
def load_model():
    try:
        with open('best_loan_model.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found! Please run model training first.")
        return None


def show_login_page():
    """Display login and registration page"""
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("## 📝 Register New Account")
        with st.form("register_form"):
            email = st.text_input("Email", key="reg_email")
            username = st.text_input("Username", key="reg_username")
            password = st.text_input("Password", type="password", key="reg_password")
            confirm_password = st.text_input("Confirm Password", type="password")
            full_name = st.text_input("Full Name", key="reg_fullname")
            phone = st.text_input("Phone Number", key="reg_phone")
            
            if st.form_submit_button("Register"):
                if password != confirm_password:
                    st.error("Passwords do not match!")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters!")
                else:
                    result = db.register_user(email, username, password, full_name, phone)
                    if result['success']:
                        st.success("✓ Registration successful! Please login.")
                    else:
                        st.error(f"Registration failed: {result['error']}")
    
    with col2:
        st.markdown("## 🔐 Login")
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            
            if st.form_submit_button("Login"):
                result = db.login_user(email, password)
                if result['success']:
                    st.session_state.logged_in = True
                    st.session_state.user_id = result['user_id']
                    st.session_state.user_email = email
                    st.session_state.user_data = result['user_data']
                    st.success("✓ Login successful!")
                    st.rerun()
                else:
                    st.error(f"Login failed: {result['error']}")


def show_dashboard():
    """Display main dashboard"""
    st.markdown(f"### Welcome, {st.session_state.user_data['full_name']} 👋")
    
    # Sidebar navigation
    st.sidebar.title("🏦 Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Dashboard", "Loan Application", "Transactions", "Predictions", "Forecasting", "Account Settings"]
    )
    
    if st.sidebar.button("🚪 Logout"):
        st.session_state.logged_in = False
        st.session_state.user_id = None
        st.session_state.user_email = None
        st.session_state.user_data = None
        st.rerun()
    
    if page == "Dashboard":
        show_home_dashboard()
    elif page == "Loan Application":
        show_loan_application()
    elif page == "Transactions":
        show_transactions_page()
    elif page == "Predictions":
        show_predictions_page()
    elif page == "Forecasting":
        show_forecasting_page()
    elif page == "Account Settings":
        show_account_settings()


def show_home_dashboard():
    """Home dashboard with balance and quick actions"""
    st.markdown("## 💼 Dashboard")
    
    # Refresh user data
    user_info = db.get_user(st.session_state.user_id)
    if user_info['success']:
        balance = user_info['user']['balance']
    else:
        balance = 0
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💵 Account Balance", f"${balance:,.2f}")
    with col2:
        st.metric("📧 Email", st.session_state.user_data['email'][:20] + "...")
    with col3:
        st.metric("👤 Username", st.session_state.user_data['username'])
    with col4:
        st.metric("📱 Member Since", "2024")
    
    st.divider()
    
    # Quick Actions
    st.markdown("### ⚡ Quick Actions")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if st.button("💰 Deposit", use_container_width=True):
            st.session_state.show_deposit = True
    
    with col2:
        if st.button("💸 Withdraw", use_container_width=True):
            st.session_state.show_withdraw = True
    
    with col3:
        if st.button("🔄 Transfer", use_container_width=True):
            st.session_state.show_transfer = True
    
    with col4:
        if st.button("📱 Recharge", use_container_width=True):
            st.session_state.show_recharge = True
    
    with col5:
        if st.button("🎯 Apply Loan", use_container_width=True):
            st.session_state.show_loan = True
    
    st.divider()
    
    # Transaction Actions
    if st.session_state.get('show_deposit', False):
        with st.container(border=True):
            st.markdown("### 💰 Deposit Money")
            amount = st.number_input("Amount ($)", min_value=0.01, step=0.01, key="deposit_amount")
            if st.button("Confirm Deposit"):
                result = db.deposit(st.session_state.user_id, amount)
                if result['success']:
                    st.success(result['message'])
                    st.session_state.show_deposit = False
                    st.rerun()
                else:
                    st.error(result['error'])
    
    if st.session_state.get('show_withdraw', False):
        with st.container(border=True):
            st.markdown("### 💸 Withdraw Money")
            amount = st.number_input("Amount ($)", min_value=0.01, step=0.01, key="withdraw_amount")
            if st.button("Confirm Withdrawal"):
                result = db.withdraw(st.session_state.user_id, amount)
                if result['success']:
                    st.success(result['message'])
                    st.session_state.show_withdraw = False
                    st.rerun()
                else:
                    st.error(result['error'])
    
    if st.session_state.get('show_transfer', False):
        with st.container(border=True):
            st.markdown("### 🔄 Transfer Money")
            recipient_email = st.text_input("Recipient Email")
            amount = st.number_input("Amount ($)", min_value=0.01, step=0.01, key="transfer_amount")
            if st.button("Confirm Transfer"):
                result = db.transfer(st.session_state.user_id, recipient_email, amount)
                if result['success']:
                    st.success(result['message'])
                    st.session_state.show_transfer = False
                    st.rerun()
                else:
                    st.error(result['error'])
    
    if st.session_state.get('show_recharge', False):
        with st.container(border=True):
            st.markdown("### 📱 Mobile Recharge")
            phone = st.text_input("Phone Number")
            amount = st.number_input("Amount ($)", min_value=0.01, step=0.01, key="recharge_amount")
            if st.button("Confirm Recharge"):
                result = db.recharge(st.session_state.user_id, phone, amount)
                if result['success']:
                    st.success(result['message'])
                    st.session_state.show_recharge = False
                    st.rerun()
                else:
                    st.error(result['error'])
    
    # Recent Transactions
    st.markdown("### 📊 Recent Transactions")
    trans_result = db.get_user_transactions(st.session_state.user_id, limit=10)
    if trans_result['success'] and trans_result['transactions']:
        trans_df = pd.DataFrame(trans_result['transactions'])
        st.dataframe(trans_df[['type', 'amount', 'description', 'timestamp']], use_container_width=True)
    else:
        st.info("No transactions yet")


def show_loan_application():
    """Loan application page with risk assessment"""
    st.markdown("## 🎯 Loan Application & Risk Assessment")
    
    model = load_model()
    if not model:
        return
    
    with st.form("loan_application_form"):
        st.markdown("### Applicant Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=35)
            income = st.number_input("Monthly Income ($)", min_value=100, value=5000)
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)
        
        with col2:
            loan_amount = st.number_input("Loan Amount ($)", min_value=100, value=10000)
            loan_term = st.number_input("Loan Term (months)", min_value=3, max_value=60, value=24)
            employment = st.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed"])
        
        col1, col2 = st.columns(2)
        with col1:
            repayment_history = st.number_input("Repayment History Score", min_value=0, max_value=5, value=3)
        with col2:
            collateral_value = st.number_input("Collateral Value ($)", min_value=0, value=5000)
        
        purpose = st.text_input("Loan Purpose (e.g., Business, Education)")
        
        if st.form_submit_button("📊 Assess Risk & Apply"):
            # Prepare data for prediction
            employment_map = {'Employed': 1, 'Self-employed': 2, 'Unemployed': 0}
            
            input_data = np.array([[
                age, income, loan_amount, loan_term, credit_score,
                repayment_history, employment_map[employment], collateral_value
            ]])
            
            # Scale input (assuming scaler was saved)
            input_scaled = model['scaler'].transform(input_data)
            
            # Make prediction
            prediction_proba = model['model'].predict_proba(input_scaled)[0]
            default_probability = prediction_proba[1]
            
            # Save prediction
            prediction_data = {
                'age': age, 'income': income, 'loan_amount': loan_amount,
                'loan_term': loan_term, 'credit_score': credit_score,
                'employment': employment, 'collateral_value': collateral_value
            }
            db.save_prediction(st.session_state.user_id, prediction_data, 
                             default_probability, model['model_name'])
            
            # Display results
            st.divider()
            st.markdown("### 📈 Risk Assessment Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Default Risk Score", f"{default_probability:.2%}")
            with col2:
                risk_level = "🔴 HIGH" if default_probability > 0.6 else "🟡 MEDIUM" if default_probability > 0.3 else "🟢 LOW"
                st.metric("Risk Level", risk_level)
            with col3:
                approval_prob = 1 - default_probability
                st.metric("Approval Probability", f"{approval_prob:.2%}")
            
            # Detailed insights
            with st.container(border=True):
                st.markdown("### 💡 Risk Insights")
                if default_probability > 0.6:
                    st.warning(f"⚠️ High default risk ({default_probability:.2%}). Recommend additional collateral or co-signer.")
                elif default_probability > 0.3:
                    st.info(f"⏳ Moderate risk ({default_probability:.2%}). Standard approval with rate adjustment.")
                else:
                    st.success(f"✓ Low default risk ({default_probability:.2%}). Recommended for approval.")
            
            # Apply for loan
            if st.button("✅ Apply for Loan", use_container_width=True):
                result = db.apply_for_loan(st.session_state.user_id, loan_amount, loan_term, purpose)
                if result['success']:
                    st.success(f"Loan application submitted! Application ID: {result['application_id']}")
                else:
                    st.error(result['error'])


def show_transactions_page():
    """Display user transaction history"""
    st.markdown("## 💳 Transaction History")
    
    trans_result = db.get_user_transactions(st.session_state.user_id, limit=100)
    if trans_result['success'] and trans_result['transactions']:
        df = pd.DataFrame(trans_result['transactions'])
        
        # Display stats
        col1, col2, col3 = st.columns(3)
        with col1:
            deposits = df[df['type'] == 'deposit']['amount'].sum()
            st.metric("Total Deposits", f"${deposits:,.2f}")
        with col2:
            withdrawals = df[df['type'] == 'withdrawal']['amount'].sum()
            st.metric("Total Withdrawals", f"${withdrawals:,.2f}")
        with col3:
            st.metric("Total Transactions", len(df))
        
        st.divider()
        
        # Display transactions table
        st.dataframe(df[['type', 'amount', 'description', 'timestamp']], use_container_width=True)
        
        # Visualization
        fig = px.bar(df.groupby('type')['amount'].sum(), title="Transaction Amount by Type")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No transactions yet")


def show_predictions_page():
    """Display prediction history"""
    st.markdown("## 🎯 Prediction History")
    st.info("View your loan default risk predictions and assessments")


def show_forecasting_page():
    """Forecasting page with time series prediction"""
    st.markdown("## 📊 Financial Forecasting")
    
    # Generate sample forecasting data
    dates = pd.date_range(start='2024-01-01', periods=90, freq='D')
    user_info = db.get_user(st.session_state.user_id)
    balance = user_info['user']['balance'] if user_info['success'] else 1000
    
    # Simulate balance trend
    trend = np.linspace(balance, balance * 1.2, 90)
    noise = np.random.normal(0, 50, 90)
    simulated_balance = trend + noise
    
    df_forecast = pd.DataFrame({
        'Date': dates,
        'Balance': simulated_balance
    })
    
    # Display current balance
    st.metric("Current Balance", f"${balance:,.2f}")
    st.divider()
    
    # Forecast chart
    fig = px.line(df_forecast, x='Date', y='Balance', 
                  title="Balance Forecast (90 Days)")
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Projected Balance (30 days)", f"${df_forecast['Balance'].iloc[30]:,.2f}")
    with col2:
        st.metric("Projected Balance (60 days)", f"${df_forecast['Balance'].iloc[60]:,.2f}")
    with col3:
        st.metric("Projected Balance (90 days)", f"${df_forecast['Balance'].iloc[89]:,.2f}")


def show_account_settings():
    """Account settings page"""
    st.markdown("## ⚙️ Account Settings")
    
    user_info = db.get_user(st.session_state.user_id)
    if user_info['success']:
        user = user_info['user']
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("### Account Information")
            st.text(f"Email: {user['email']}")
            st.text(f"Username: {user['username']}")
            st.text(f"Full Name: {user['full_name']}")
            st.text(f"Phone: {user['phone']}")
            st.text(f"Account Balance: ${user['balance']:,.2f}")
        
        with col2:
            st.markdown("### Account Statistics")
            st.text(f"Account Status: {user['account_status']}")
            st.text(f"Created: {user['created_at']}")
            st.text(f"Last Updated: {user['updated_at']}")
    
    st.divider()
    st.markdown("### Change Password")
    with st.form("change_password_form"):
        old_password = st.text_input("Old Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        
        if st.form_submit_button("Update Password"):
            st.info("Password change feature coming soon")


# Main App
def main():
    st.markdown("# 💰 Loan Default Prediction System")
    st.markdown("*Microfinance Risk Management & Digital Banking Platform*")
    
    if not st.session_state.logged_in:
        show_login_page()
    else:
        show_dashboard()


if __name__ == "__main__":
    main()
