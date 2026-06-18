"""
Model Loading and Prediction
"""

import pickle
import numpy as np
import streamlit as st
import logging
from config import Config

logger = logging.getLogger(__name__)


class ModelLoader:
    """Handle model loading and predictions"""
    
    @staticmethod
    @st.cache_resource
    def load_model():
        """Load trained model"""
        try:
            model_path = Config.get_model_path()
            with open(model_path, 'rb') as f:
                model_data = pickle.load(f)
            logger.info(f"✓ Model loaded: {model_data['model_name']}")
            return model_data
        except FileNotFoundError:
            logger.error(f"Model file not found: {Config.get_model_path()}")
            return None
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None
    
    @staticmethod
    def predict_default_risk(model_data, features):
        """
        Predict loan default risk
        
        features: [age, income, loan_amount, loan_term, credit_score, 
                  repayment_history, employment, collateral_value]
        """
        try:
            if not model_data:
                return {
                    'success': False,
                    'error': 'Model not loaded',
                    'default_probability': None
                }
            
            input_data = np.array([features])
            scaler = model_data['scaler']
            input_scaled = scaler.transform(input_data)
            
            model = model_data['model']
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            default_probability = prediction_proba[1]
            approval_probability = prediction_proba[0]
            
            # Risk level
            if default_probability > 0.6:
                risk_level = "🔴 HIGH RISK"
            elif default_probability > 0.35:
                risk_level = "🟡 MODERATE RISK"
            else:
                risk_level = "🟢 LOW RISK"
            
            return {
                'success': True,
                'default_probability': default_probability,
                'approval_probability': approval_probability,
                'risk_level': risk_level,
                'model_name': model_data['model_name']
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'default_probability': None
            }
    
    @staticmethod
    def get_risk_recommendations(default_probability, income, credit_score, collateral_value):
        """Get risk recommendations"""
        recommendations = []
        interest_rate_adjustment = 0
        required_collateral = None
        
        if default_probability > 0.6:
            recommendations.append("❌ Not recommended for approval")
            recommendations.append("📋 Request additional collateral or co-signer")
            interest_rate_adjustment = 5
            required_collateral = income * 2
        elif default_probability > 0.35:
            recommendations.append("⚠️ Conditional approval")
            recommendations.append("📋 Requires collateral > 75% of loan amount")
            recommendations.append("💳 Shorter loan term recommended")
            interest_rate_adjustment = 2.5
            required_collateral = income * 1.5
        else:
            recommendations.append("✅ Low risk - Approved")
            recommendations.append("💚 Standard terms apply")
            interest_rate_adjustment = 0
            required_collateral = income * 0.5
        
        if credit_score < 400:
            recommendations.append("⚠️ Credit score below threshold")
        
        if collateral_value > 0 and required_collateral and collateral_value < required_collateral:
            recommendations.append(f"📊 Collateral: Need ${required_collateral:,.2f}, Have ${collateral_value:,.2f}")
        
        return {
            'recommendations': recommendations,
            'interest_rate_adjustment': interest_rate_adjustment,
            'required_collateral': required_collateral
        }
