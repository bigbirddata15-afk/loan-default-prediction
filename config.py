"""
Configuration Management for Loan Default Prediction App
"""

import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application Configuration"""
    
    @staticmethod
    def get_mongodb_uri():
        """Get MongoDB URI"""
        try:
            if 'MONGODB_URI' in st.secrets:
                return st.secrets['MONGODB_URI']
        except:
            pass
        return "mongodb+srv://euawari_db_user:6SnKvQvXXzrGeypA@cluster0.fkkzcvz.mongodb.net/microfinance_db?retryWrites=true&w=majority"
    
    @staticmethod
    def get_db_name():
        """Get database name"""
        try:
            if 'DB_NAME' in st.secrets:
                return st.secrets['DB_NAME']
        except:
            pass
        return "microfinance_db"
    
    @staticmethod
    def get_model_path():
        """Get model file path"""
        return 'best_loan_model.pkl'
