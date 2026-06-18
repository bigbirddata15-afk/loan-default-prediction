"""
MongoDB User and Transaction Management
"""

from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
import bcrypt
from datetime import datetime
from bson import ObjectId
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoDBManager:
    """Handle all MongoDB operations"""
    
    def __init__(self):
        """Initialize MongoDB connection"""
        try:
            self.uri = Config.get_mongodb_uri()
            self.db_name = Config.get_db_name()
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            self.db = self.client[self.db_name]
            
            # Collections
            self.users = self.db['users']
            self.transactions = self.db['transactions']
            self.loan_applications = self.db['loan_applications']
            self.predictions = self.db['predictions']
            
            # Create indexes
            self._create_indexes()
            logger.info("✓ MongoDB connected successfully")
        except Exception as e:
            logger.error(f"✗ MongoDB connection failed: {str(e)}")
            raise
    
    def _create_indexes(self):
        """Create database indexes"""
        try:
            self.users.create_index('email', unique=True, sparse=True)
            self.users.create_index('username', unique=True, sparse=True)
            self.transactions.create_index('user_id')
            self.loan_applications.create_index('user_id')
            self.predictions.create_index('user_id')
        except Exception as e:
            logger.warning(f"Index creation: {str(e)}")
    
    @staticmethod
    def hash_password(password):
        """Hash password"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    
    @staticmethod
    def verify_password(password, hashed):
        """Verify password"""
        try:
            return bcrypt.checkpw(password.encode('utf-8'), hashed)
        except:
            return False
    
    # ===== USER MANAGEMENT =====
    def register_user(self, email, username, password, full_name, phone):
        """Register new user"""
        try:
            if not all([email, username, password, full_name, phone]):
                return {'success': False, 'error': 'All fields required'}
            
            if len(password) < 6:
                return {'success': False, 'error': 'Password must be 6+ characters'}
            
            user = {
                'email': email.lower(),
                'username': username.lower(),
                'password': self.hash_password(password),
                'full_name': full_name,
                'phone': phone,
                'balance': 0.0,
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'account_status': 'active'
            }
            result = self.users.insert_one(user)
            logger.info(f"User registered: {email}")
            return {'success': True, 'user_id': str(result.inserted_id)}
        except DuplicateKeyError:
            return {'success': False, 'error': 'Email or username already exists'}
        except Exception as e:
            logger.error(f"Registration error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def login_user(self, email, password):
        """Login user"""
        try:
            user = self.users.find_one({'email': email.lower()})
            if user and self.verify_password(password, user['password']):
                logger.info(f"User logged in: {email}")
                return {
                    'success': True,
                    'user_id': str(user['_id']),
                    'user_data': {
                        'email': user['email'],
                        'username': user['username'],
                        'full_name': user['full_name'],
                        'balance': user['balance'],
                        'phone': user['phone']
                    }
                }
            return {'success': False, 'error': 'Invalid email or password'}
        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_user(self, user_id):
        """Get user details"""
        try:
            user = self.users.find_one({'_id': ObjectId(user_id)})
            if user:
                user.pop('password', None)
                user['_id'] = str(user['_id'])
                return {'success': True, 'user': user}
            return {'success': False, 'error': 'User not found'}
        except Exception as e:
            logger.error(f"Get user error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    # ===== BALANCE OPERATIONS =====
    def get_balance(self, user_id):
        """Get user balance"""
        try:
            user = self.users.find_one({'_id': ObjectId(user_id)})
            if user:
                return {'success': True, 'balance': user['balance']}
            return {'success': False, 'error': 'User not found', 'balance': 0}
        except Exception as e:
            return {'success': False, 'error': str(e), 'balance': 0}
    
    def deposit(self, user_id, amount, description='Deposit'):
        """Deposit money"""
        try:
            if amount <= 0:
                return {'success': False, 'error': 'Amount must be > 0'}
            
            result = self.users.update_one(
                {'_id': ObjectId(user_id)},
                {'$inc': {'balance': amount}, '$set': {'updated_at': datetime.now()}}
            )
            
            if result.modified_count > 0:
                self._log_transaction(user_id, 'deposit', amount, description)
                return {'success': True, 'message': f'✓ Deposited ${amount:.2f}'}
            return {'success': False, 'error': 'User not found'}
        except Exception as e:
            logger.error(f"Deposit error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def withdraw(self, user_id, amount, description='Withdrawal'):
        """Withdraw money"""
        try:
            if amount <= 0:
                return {'success': False, 'error': 'Amount must be > 0'}
            
            user = self.users.find_one({'_id': ObjectId(user_id)})
            
            if not user:
                return {'success': False, 'error': 'User not found'}
            if user['balance'] < amount:
                return {'success': False, 'error': f'Insufficient balance. Available: ${user["balance"]:.2f}'}
            
            self.users.update_one(
                {'_id': ObjectId(user_id)},
                {'$inc': {'balance': -amount}, '$set': {'updated_at': datetime.now()}}
            )
            
            self._log_transaction(user_id, 'withdrawal', amount, description)
            return {'success': True, 'message': f'✓ Withdrawn ${amount:.2f}'}
        except Exception as e:
            logger.error(f"Withdrawal error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def transfer(self, from_user_id, to_email, amount, description='Transfer'):
        """Transfer money between users"""
        try:
            if amount <= 0:
                return {'success': False, 'error': 'Amount must be > 0'}
            
            from_user = self.users.find_one({'_id': ObjectId(from_user_id)})
            to_user = self.users.find_one({'email': to_email.lower()})
            
            if not from_user:
                return {'success': False, 'error': 'Sender not found'}
            if not to_user:
                return {'success': False, 'error': 'Recipient not found'}
            if from_user['balance'] < amount:
                return {'success': False, 'error': 'Insufficient balance'}
            if from_user['_id'] == to_user['_id']:
                return {'success': False, 'error': 'Cannot transfer to yourself'}
            
            self.users.update_one({'_id': ObjectId(from_user_id)}, {'$inc': {'balance': -amount}})
            self.users.update_one({'_id': to_user['_id']}, {'$inc': {'balance': amount}})
            
            self._log_transaction(from_user_id, 'transfer_out', amount, f'Transfer to {to_email}')
            self._log_transaction(str(to_user['_id']), 'transfer_in', amount, f'Transfer from {from_user["email"]}')
            
            return {'success': True, 'message': f'✓ Transferred ${amount:.2f} to {to_email}'}
        except Exception as e:
            logger.error(f"Transfer error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def recharge(self, user_id, phone, amount, provider='MTN', description='Mobile Recharge'):
        """Mobile recharge"""
        try:
            if amount <= 0:
                return {'success': False, 'error': 'Amount must be > 0'}
            
            user = self.users.find_one({'_id': ObjectId(user_id)})
            
            if not user:
                return {'success': False, 'error': 'User not found'}
            if user['balance'] < amount:
                return {'success': False, 'error': 'Insufficient balance'}
            
            self.users.update_one(
                {'_id': ObjectId(user_id)},
                {'$inc': {'balance': -amount}, '$set': {'updated_at': datetime.now()}}
            )
            
            self._log_transaction(user_id, 'recharge', amount, f'Recharge: {phone} ({provider})')
            return {'success': True, 'message': f'✓ Recharged ${amount:.2f} on {phone}'}
        except Exception as e:
            logger.error(f"Recharge error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _log_transaction(self, user_id, transaction_type, amount, description):
        """Log transaction"""
        try:
            transaction = {
                'user_id': ObjectId(user_id),
                'type': transaction_type,
                'amount': amount,
                'description': description,
                'timestamp': datetime.now()
            }
            self.transactions.insert_one(transaction)
        except Exception as e:
            logger.error(f"Transaction log error: {str(e)}")
    
    # ===== LOAN APPLICATIONS =====
    def apply_for_loan(self, user_id, loan_amount, loan_term, purpose, risk_score):
        """Submit loan application"""
        try:
            application = {
                'user_id': ObjectId(user_id),
                'loan_amount': loan_amount,
                'loan_term': loan_term,
                'purpose': purpose,
                'risk_score': risk_score,
                'status': 'pending',
                'created_at': datetime.now(),
                'updated_at': datetime.now()
            }
            result = self.loan_applications.insert_one(application)
            return {'success': True, 'application_id': str(result.inserted_id)}
        except Exception as e:
            logger.error(f"Loan application error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_loan_applications(self, user_id, limit=10):
        """Get user loan applications"""
        try:
            apps = list(self.loan_applications.find(
                {'user_id': ObjectId(user_id)}
            ).sort('created_at', -1).limit(limit))
            
            for app in apps:
                app['_id'] = str(app['_id'])
                app['user_id'] = str(app['user_id'])
            
            return {'success': True, 'applications': apps}
        except Exception as e:
            logger.error(f"Get applications error: {str(e)}")
            return {'success': False, 'error': str(e), 'applications': []}
    
    # ===== PREDICTIONS =====
    def save_prediction(self, user_id, prediction_data, risk_score, model_name):
        """Save prediction"""
        try:
            prediction = {
                'user_id': ObjectId(user_id),
                'prediction_data': prediction_data,
                'risk_score': risk_score,
                'model_name': model_name,
                'is_default': risk_score > 0.5,
                'created_at': datetime.now()
            }
            result = self.predictions.insert_one(prediction)
            return {'success': True, 'prediction_id': str(result.inserted_id)}
        except Exception as e:
            logger.error(f"Save prediction error: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    # ===== TRANSACTIONS =====
    def get_user_transactions(self, user_id, limit=50):
        """Get user transactions"""
        try:
            transactions = list(self.transactions.find(
                {'user_id': ObjectId(user_id)}
            ).sort('timestamp', -1).limit(limit))
            
            for t in transactions:
                t['_id'] = str(t['_id'])
                t['user_id'] = str(t['user_id'])
                t['timestamp'] = str(t['timestamp'])
            
            return {'success': True, 'transactions': transactions}
        except Exception as e:
            logger.error(f"Get transactions error: {str(e)}")
            return {'success': False, 'error': str(e), 'transactions': []}
    
    def get_transaction_stats(self, user_id):
        """Get transaction stats"""
        try:
            trans = list(self.transactions.find({'user_id': ObjectId(user_id)}))
            
            stats = {
                'total_transactions': len(trans),
                'total_deposits': sum(t['amount'] for t in trans if t['type'] == 'deposit'),
                'total_withdrawals': sum(t['amount'] for t in trans if t['type'] == 'withdrawal'),
                'total_transfers_out': sum(t['amount'] for t in trans if t['type'] == 'transfer_out'),
                'total_transfers_in': sum(t['amount'] for t in trans if t['type'] == 'transfer_in'),
            }
            
            return {'success': True, 'stats': stats}
        except Exception as e:
            logger.error(f"Get stats error: {str(e)}")
            return {'success': False, 'error': str(e)}
