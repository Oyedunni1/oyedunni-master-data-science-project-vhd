"""
Advanced Model Training Pipeline for VHD Detection
Implements high-accuracy ensemble learning with feature integration
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
# import xgboost as xgb
# import lightgbm as lgb
from typing import Dict, List, Tuple, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

class EnsemblePredictor:
    """
    Ensemble predictor class that can be pickled
    """
    def __init__(self, models, weights, scaler, feature_selector):
        self.models = models
        self.weights = weights
        self.scaler = scaler
        self.feature_selector = feature_selector
    
    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        predictions = []
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X_selected)[:, 1]
            predictions.append(pred_proba)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        return (ensemble_pred > 0.5).astype(int)
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        X_selected = self.feature_selector.transform(X_scaled)
        
        predictions = []
        for name, model in self.models.items():
            pred_proba = model.predict_proba(X_selected)[:, 1]
            predictions.append(pred_proba)
        
        # Weighted average
        ensemble_pred = np.average(predictions, axis=0, weights=self.weights)
        return np.column_stack([1 - ensemble_pred, ensemble_pred])

class VHDModelTrainer:
    """
    Advanced model trainer for VHD detection with ensemble methods
    """
    
    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.models = {}
        self.best_model = None
        self.feature_importance = None
        
    def prepare_data(self, fractal_features: np.ndarray, deep_features: np.ndarray, 
                     labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare and integrate features for training
        """
        # Combine fractal and deep features
        if len(fractal_features.shape) == 1:
            fractal_features = fractal_features.reshape(-1, 1)
        
        # Flatten deep features if they're multi-dimensional
        if len(deep_features.shape) > 2:
            deep_features = deep_features.reshape(deep_features.shape[0], -1)
        
        # Integrate features
        integrated_features = np.concatenate([fractal_features, deep_features], axis=1)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            integrated_features, labels, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=labels
        )
        
        return X_train, X_test, y_train, y_test
    
    def feature_selection(self, X_train: np.ndarray, y_train: np.ndarray, 
                         method: str = 'mutual_info', k: int = 100) -> np.ndarray:
        """
        Advanced feature selection for optimal performance
        """
        if method == 'mutual_info':
            from sklearn.feature_selection import mutual_info_classif
            selector = SelectKBest(score_func=mutual_info_classif, k=k)
        elif method == 'f_score':
            selector = SelectKBest(score_func=f_classif, k=k)
        elif method == 'rfe':
            # Use Random Forest for RFE
            rf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
            selector = RFE(estimator=rf, n_features_to_select=k)
        else:
            return X_train
        
        X_selected = selector.fit_transform(X_train, y_train)
        self.feature_selector = selector
        
        return X_selected
    
    def train_statistical_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train ensemble of models for maximum accuracy with verbose progress
        """
        print("🔧 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        print(f"✅ Feature scaling completed: {X_train_scaled.shape}")
        
        print("🎯 Selecting best features...")
        X_train_selected = self.feature_selection(X_train_scaled, y_train, k=min(200, X_train_scaled.shape[1]))
        print(f"✅ Feature selection completed: {X_train_selected.shape[1]} features selected")
        
        models = {
            # 'xgboost': xgb.XGBClassifier(
            #     n_estimators=1000,
            #     max_depth=6,
            #     learning_rate=0.05,
            #     subsample=0.8,
            #     colsample_bytree=0.8,
            #     random_state=self.random_state,
            #     eval_metric='logloss',
            #     early_stopping_rounds=50
            # ),
            # 'lightgbm': lgb.LGBMClassifier(
            #     n_estimators=1000,
            #     max_depth=6,
            #     learning_rate=0.05,
            #     subsample=0.8,
            #     colsample_bytree=0.8,
            #     random_state=self.random_state,
            #     verbose=-1
            # ),
            'random_forest': RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                random_state=self.random_state
            ),
            'logistic_regression': LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state,
                n_jobs=-1
            )
        }
        
        # Train each model with verbose progress
        print("🤖 Training individual models...")
        trained_models = {}
        for i, (name, model) in enumerate(models.items(), 1):
            print(f"   [{i}/{len(models)}] Training {name}...")
            try:
                # if name in ['xgboost', 'lightgbm']:
                #     # Use early stopping for gradient boosting
                #     model.fit(X_train_selected, y_train, 
                #             eval_set=[(X_train_selected, y_train)],
                #             verbose=False)
                # else:
                model.fit(X_train_selected, y_train)
                
                trained_models[name] = model
                print(f"   ✅ {name} training completed successfully")
                
            except Exception as e:
                print(f"   ❌ Error training {name}: {e}")
                continue
        
        self.models = trained_models
        return trained_models
    
    def evaluate_models(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Comprehensive model evaluation with verbose progress
        """
        print("📊 Evaluating model performance...")
        X_test_scaled = self.scaler.transform(X_test)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        results = {}
        
        for i, (name, model) in enumerate(self.models.items(), 1):
            print(f"   [{i}/{len(self.models)}] Evaluating {name}...")
            try:
                # Predictions
                y_pred = model.predict(X_test_selected)
                y_pred_proba = model.predict_proba(X_test_selected)[:, 1]
                
                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                results[name] = {
                    'accuracy': accuracy,
                    'auc': auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                print(f"   ✅ {name} - Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
                
            except Exception as e:
                print(f"   ❌ Error evaluating {name}: {e}")
                continue
        
        return results
    
    def create_statistical_predictor(self, results: Dict[str, Dict[str, float]]) -> Any:
        """
        Create weighted ensemble predictor
        """
        # Select best models based on AUC
        best_models = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)[:3]
        
        print(f"Best models: {[name for name, _ in best_models]}")
        
        # Use the module-level EnsemblePredictor class
        
        # Calculate weights based on AUC scores
        weights = [results[name]['auc'] for name, _ in best_models]
        weights = np.array(weights) / np.sum(weights)
        
        ensemble_models = {name: self.models[name] for name, _ in best_models}
        
        self.best_model = EnsemblePredictor(ensemble_models, weights, self.scaler, self.feature_selector)
        
        return self.best_model
    
    def cross_validate_model(self, X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, float]:
        """
        Cross-validation for robust evaluation
        """
        X_scaled = self.scaler.fit_transform(X)
        X_selected = self.feature_selection(X_scaled, y)
        
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)
        
        cv_results = {}
        for name, model in self.models.items():
            try:
                scores = cross_val_score(model, X_selected, y, cv=cv, scoring='roc_auc')
                cv_results[name] = {
                    'mean_auc': np.mean(scores),
                    'std_auc': np.std(scores),
                    'scores': scores
                }
                print(f"{name} - CV AUC: {np.mean(scores):.4f} (+/- {np.std(scores)*2:.4f})")
            except Exception as e:
                print(f"Error in CV for {name}: {e}")
                continue
        
        return cv_results
    
    def optimize_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Hyperparameter optimization for best models
        """
        from sklearn.model_selection import GridSearchCV
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_train_selected = self.feature_selection(X_train_scaled, y_train)
        
        # Random Forest optimization (replacing XGBoost for now)
        rf_param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf_grid = GridSearchCV(
            RandomForestClassifier(random_state=self.random_state),
            rf_param_grid,
            cv=3,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        rf_grid.fit(X_train_selected, y_train)
        print(f"Best Random Forest params: {rf_grid.best_params_}")
        print(f"Best Random Forest score: {rf_grid.best_score_:.4f}")
        
        return rf_grid.best_estimator_
    
    def save_model(self, filepath: str):
        """
        Save trained model and preprocessing objects
        """
        model_data = {
            'best_model': self.best_model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'models': self.models
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load trained model and preprocessing objects
        """
        model_data = joblib.load(filepath)
        self.best_model = model_data['best_model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data['feature_selector']
        self.models = model_data['models']
        print(f"Model loaded from {filepath}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from best models
        """
        if self.best_model is None:
            return {}
        
        importance_dict = {}
        
        # Get importance from tree-based models
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_dict[name] = model.feature_importances_
        
        return importance_dict
