"""
Loan Default Prediction - Model Training
Trains and compares 3 ML models: Logistic Regression, Decision Tree, Random Forest
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, roc_auc_score, 
                             roc_curve, auc)
import pickle
import warnings
warnings.filterwarnings('ignore')

sns.set_style("darkgrid")
plt.rcParams['figure.figsize'] = (12, 6)


class LoanDefaultModelTrainer:
    """Train and compare multiple ML models for loan default prediction"""
    
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.results = {}
        
    def preprocess_data(self, target_column='default', test_size=0.2, random_state=42):
        """Preprocess and split the data"""
        print("=" * 80)
        print("DATA PREPROCESSING")
        print("=" * 80)
        
        # Handle missing values
        self.data = self.data.fillna(self.data.mean(numeric_only=True))
        
        # Separate features and target
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        # Encode categorical variables
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\nTraining set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        print(f"Default rate: {(y.sum() / len(y) * 100):.2f}%")
        
    def train_logistic_regression(self):
        """Train Logistic Regression"""
        print("\n" + "=" * 80)
        print("TRAINING: LOGISTIC REGRESSION")
        print("=" * 80)
        model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        self.models['Logistic Regression'] = model
        print("✓ Model trained successfully")
        
    def train_decision_tree(self):
        """Train Decision Tree"""
        print("\n" + "=" * 80)
        print("TRAINING: DECISION TREE")
        print("=" * 80)
        model = DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=42)
        model.fit(self.X_train, self.y_train)
        self.models['Decision Tree'] = model
        print("✓ Model trained successfully")
        
    def train_random_forest(self):
        """Train Random Forest"""
        print("\n" + "=" * 80)
        print("TRAINING: RANDOM FOREST")
        print("=" * 80)
        model = RandomForestClassifier(n_estimators=100, max_depth=15,
                                       min_samples_split=10, random_state=42, n_jobs=-1)
        model.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = model
        print("✓ Model trained successfully")
        
    def evaluate_model(self, model_name, model):
        """Evaluate a single model"""
        y_pred = model.predict(self.X_test)
        y_pred_proba = model.predict_proba(self.X_test)[:, 1]
        
        metrics = {
            'Accuracy': accuracy_score(self.y_test, y_pred),
            'Precision': precision_score(self.y_test, y_pred, zero_division=0),
            'Recall': recall_score(self.y_test, y_pred, zero_division=0),
            'F1-Score': f1_score(self.y_test, y_pred, zero_division=0),
            'ROC-AUC': roc_auc_score(self.y_test, y_pred_proba),
        }
        
        self.results[model_name] = {
            'metrics': metrics,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'confusion_matrix': confusion_matrix(self.y_test, y_pred)
        }
        return metrics
    
    def evaluate_all_models(self):
        """Evaluate all models and display comparison"""
        print("\n" + "=" * 80)
        print("MODEL EVALUATION & COMPARISON")
        print("=" * 80)
        
        for model_name, model in self.models.items():
            print(f"\n{model_name}:")
            print("-" * 40)
            metrics = self.evaluate_model(model_name, model)
            for metric_name, value in metrics.items():
                print(f"{metric_name}: {value:.4f}")
        
        comparison_df = pd.DataFrame({
            model_name: metrics['metrics'] 
            for model_name, metrics in self.results.items()
        }).T
        
        print("\n" + "=" * 80)
        print("MODELS COMPARISON TABLE")
        print("=" * 80)
        print(comparison_df.round(4))
        
        return comparison_df
    
    def get_best_model(self):
        """Identify best model based on F1-Score"""
        comparison_df = pd.DataFrame({
            model_name: metrics['metrics'] 
            for model_name, metrics in self.results.items()
        }).T
        
        best_model_name = comparison_df['F1-Score'].idxmax()
        best_model = self.models[best_model_name]
        
        print("\n" + "=" * 80)
        print("BEST MODEL SELECTED")
        print("=" * 80)
        print(f"Model: {best_model_name}")
        print(f"F1-Score: {comparison_df.loc[best_model_name, 'F1-Score']:.4f}")
        
        return best_model_name, best_model, comparison_df
    
    def save_model(self, model_name, model, filename='best_loan_model.pkl'):
        """Save model to pickle file"""
        model_data = {
            'model': model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'model_name': model_name,
            'feature_count': self.X_train.shape[1]
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"\n✓ Model saved successfully: {filename}")
    
    def plot_results(self):
        """Plot comparison results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        comparison_df = pd.DataFrame({
            model_name: metrics['metrics'] 
            for model_name, metrics in self.results.items()
        }).T
        
        # Metrics comparison
        comparison_df[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(kind='bar', ax=axes[0, 0])
        axes[0, 0].set_title('Model Performance Metrics Comparison', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].legend(loc='best')
        
        # ROC Curves
        for model_name in self.models.keys():
            y_pred_proba = self.results[model_name]['y_pred_proba']
            fpr, tpr, _ = roc_curve(self.y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            axes[0, 1].plot(fpr, tpr, label=f'{model_name} (AUC={roc_auc:.3f})')
        
        axes[0, 1].plot([0, 1], [0, 1], 'k--', label='Random')
        axes[0, 1].set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        
        # Confusion matrix of best model
        best_model_name = comparison_df['F1-Score'].idxmax()
        cm = self.results[best_model_name]['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
        axes[1, 0].set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
        
        # F1-Score comparison
        f1_scores = comparison_df['F1-Score'].sort_values(ascending=False)
        axes[1, 1].barh(f1_scores.index, f1_scores.values, color=['#2ecc71', '#3498db', '#e74c3c'])
        axes[1, 1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\n✓ Comparison plots saved: model_comparison.png")
        plt.show()


# Usage
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    sample_data = pd.DataFrame({
        'age': np.random.randint(20, 70, n_samples),
        'income': np.random.randint(5000, 100000, n_samples),
        'loan_amount': np.random.randint(1000, 50000, n_samples),
        'loan_term': np.random.randint(12, 60, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'repayment_history': np.random.randint(0, 5, n_samples),
        'employment_status': np.random.choice(['Employed', 'Self-employed', 'Unemployed'], n_samples),
        'collateral_value': np.random.randint(0, 100000, n_samples),
        'default': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    })
    
    sample_data.to_csv('sample_loan_data.csv', index=False)
    
    # Train models
    trainer = LoanDefaultModelTrainer('sample_loan_data.csv')
    trainer.preprocess_data(target_column='default')
    trainer.train_logistic_regression()
    trainer.train_decision_tree()
    trainer.train_random_forest()
    
    # Evaluate and save best model
    trainer.evaluate_all_models()
    best_model_name, best_model, _ = trainer.get_best_model()
    trainer.save_model(best_model_name, best_model, 'best_loan_model.pkl')
    trainer.plot_results()
    
    print("\n✓ Training completed! Model saved: best_loan_model.pkl")
