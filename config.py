import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

class Config:
    @staticmethod
    def get_mongodb_uri():
        try:
            if 'MONGODB_URI' in st.secrets:
                return st.secrets['MONGODB_URI']
        except:
            pass
        return os.getenv('MONGODB_URI')
    
    @staticmethod
    def get_db_name():
        try:
            if 'DB_NAME' in st.secrets:
                return st.secrets['DB_NAME']
        except:
            pass
        return os.getenv('DB_NAME', 'microfinance_db')
    
    @staticmethod
    def get_model_path():
        return 'best_loan_model.pkl'