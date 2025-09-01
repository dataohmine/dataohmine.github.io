"""
Model Training Module
Handles LightGBM model training, hyperparameter tuning, and validation
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import optuna
import joblib
import logging
from typing import Dict, Tuple, List, Optional
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class ModelTrainer:
    """
    Handles model training, hyperparameter optimization, and validation
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_logging()
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.validation_results = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def prepare_data(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2) -> Tuple:
        """
        Prepare data for training
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.logger.info("Preparing data for training")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=self.config.get('random_seed', 42),
            stratify=y
        )
        
        self.logger.info(f"Training set: {X_train.shape[0]} samples")
        self.logger.info(f"Test set: {X_test.shape[0]} samples")
        self.logger.info(f"Feature count: {X_train.shape[1]}")
        
        return X_train, X_test, y_train, y_test
    
    def objective(self, trial, X_train, y_train, X_val, y_val):
        """
        Optuna objective function for hyperparameter optimization
        
        Args:
            trial: Optuna trial object
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            Validation AUC score
        """
        # Suggest hyperparameters
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 10, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 10.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 10.0),
            'verbosity': -1,
            'random_state': self.config.get('random_seed', 42)
        }
        
        # Train model
        train_data = lgb.Dataset(X_train, label=y_train)
        val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
        
        model = lgb.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=1000,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Predict and evaluate
        y_pred = model.predict(X_val, num_iteration=model.best_iteration)
        auc_score = roc_auc_score(y_val, y_pred)
        
        return auc_score
    
    def optimize_hyperparameters(self, X_train, y_train, n_trials: int = 100) -> Dict:
        """
        Optimize hyperparameters using Optuna
        
        Args:
            X_train, y_train: Training data
            n_trials: Number of optimization trials
            
        Returns:
            Dictionary of best hyperparameters
        """
        self.logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        # Split training data for validation
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train,
            test_size=0.2,
            random_state=self.config.get('random_seed', 42),
            stratify=y_train
        )
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(
            lambda trial: self.objective(trial, X_tr, y_tr, X_val, y_val),
            n_trials=n_trials
        )
        
        self.best_params = study.best_params
        self.logger.info(f"Best validation AUC: {study.best_value:.4f}")
        self.logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params
    
    def train_model(self, X_train, y_train, X_val=None, y_val=None, params=None) -> lgb.LGBMClassifier:
        """
        Train LightGBM model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional)
            params: Model parameters (optional)
            
        Returns:
            Trained LightGBM model
        """
        self.logger.info("Training LightGBM model")
        
        # Use optimized parameters if available
        if params is None:
            params = self.best_params or self.config.get('default_params', {})
        
        # Set default parameters
        default_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbosity': -1,
            'random_state': self.config.get('random_seed', 42)
        }
        
        # Merge with provided parameters
        final_params = {**default_params, **params}
        
        # Train model
        if X_val is not None and y_val is not None:
            self.model = lgb.LGBMClassifier(**final_params)
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='auc',
                early_stopping_rounds=50,
                verbose=False
            )
        else:
            self.model = lgb.LGBMClassifier(**final_params)
            self.model.fit(X_train, y_train)
        
        # Store feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.logger.info("Model training completed")
        return self.model
    
    def evaluate_model(self, X_test, y_test) -> Dict:
        """
        Evaluate model performance
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.logger.info("Evaluating model performance")
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Predictions
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Precision-recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        self.validation_results = {
            'auc_score': auc_score,
            'accuracy': class_report['accuracy'],
            'precision': class_report['1']['precision'],
            'recall': class_report['1']['recall'],
            'f1_score': class_report['1']['f1-score'],
            'confusion_matrix': conf_matrix.tolist(),
            'classification_report': class_report
        }
        
        self.logger.info(f"Model Performance:")
        self.logger.info(f"AUC Score: {auc_score:.4f}")
        self.logger.info(f"Accuracy: {class_report['accuracy']:.4f}")
        self.logger.info(f"Precision: {class_report['1']['precision']:.4f}")
        self.logger.info(f"Recall: {class_report['1']['recall']:.4f}")
        self.logger.info(f"F1 Score: {class_report['1']['f1-score']:.4f}")
        
        return self.validation_results
    
    def cross_validate(self, X, y, cv_folds: int = 5) -> Dict:
        """
        Perform cross-validation
        
        Args:
            X, y: Full dataset
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary of CV results
        """
        self.logger.info(f"Performing {cv_folds}-fold cross-validation")
        
        if self.model is None:
            # Use default model for CV
            params = self.best_params or self.config.get('default_params', {})
            model = lgb.LGBMClassifier(**params)
        else:
            model = self.model
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.config.get('random_seed', 42))
        
        # Perform cross-validation
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='roc_auc')
        
        cv_results = {
            'cv_scores': cv_scores.tolist(),
            'mean_cv_score': cv_scores.mean(),
            'std_cv_score': cv_scores.std(),
            'cv_folds': cv_folds
        }
        
        self.logger.info(f"CV AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return cv_results
    
    def plot_feature_importance(self, top_n: int = 20, save_path: Optional[str] = None):
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to plot
            save_path: Path to save plot (optional)
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance not available. Train model first.")
        
        plt.figure(figsize=(10, 8))
        top_features = self.feature_importance.head(top_n)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title(f'Top {top_n} Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def save_model(self, model_path: str, metadata_path: str = None):
        """
        Save trained model and metadata
        
        Args:
            model_path: Path to save model
            metadata_path: Path to save metadata (optional)
        """
        if self.model is None:
            raise ValueError("No model to save. Train model first.")
        
        # Save model
        joblib.dump(self.model, model_path)
        self.logger.info(f"Model saved to {model_path}")
        
        # Save metadata
        if metadata_path:
            metadata = {
                'best_params': self.best_params,
                'validation_results': self.validation_results,
                'feature_importance': self.feature_importance.to_dict() if self.feature_importance is not None else None,
                'config': self.config
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            self.logger.info(f"Metadata saved to {metadata_path}")
    
    def load_model(self, model_path: str, metadata_path: str = None):
        """
        Load trained model and metadata
        
        Args:
            model_path: Path to load model from
            metadata_path: Path to load metadata from (optional)
        """
        # Load model
        self.model = joblib.load(model_path)
        self.logger.info(f"Model loaded from {model_path}")
        
        # Load metadata
        if metadata_path and Path(metadata_path).exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.best_params = metadata.get('best_params')
            self.validation_results = metadata.get('validation_results', {})
            
            if metadata.get('feature_importance'):
                self.feature_importance = pd.DataFrame(metadata['feature_importance'])
            
            self.logger.info(f"Metadata loaded from {metadata_path}")


def main():
    """Example usage of ModelTrainer"""
    
    # Configuration
    config = {
        'random_seed': 42,
        'default_params': {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'random_state': 42
        }
    }
    
    # Generate sample data
    from sklearn.datasets import make_classification
    X, y = make_classification(
        n_samples=10000,
        n_features=50,
        n_informative=30,
        n_redundant=10,
        n_classes=2,
        random_state=42
    )
    
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    
    # Initialize trainer
    trainer = ModelTrainer(config)
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(df, 'target')
    
    # Optimize hyperparameters
    best_params = trainer.optimize_hyperparameters(X_train, y_train, n_trials=50)
    
    # Train model
    model = trainer.train_model(X_train, y_train, params=best_params)
    
    # Evaluate model
    results = trainer.evaluate_model(X_test, y_test)
    
    # Cross-validation
    cv_results = trainer.cross_validate(df.drop('target', axis=1), df['target'])
    
    # Save model
    trainer.save_model('models/lightgbm_model.pkl', 'models/model_metadata.json')
    
    print("Model training and evaluation completed successfully!")
    print(f"Test AUC: {results['auc_score']:.4f}")
    print(f"CV AUC: {cv_results['mean_cv_score']:.4f} (+/- {cv_results['std_cv_score'] * 2:.4f})")


if __name__ == "__main__":
    main()