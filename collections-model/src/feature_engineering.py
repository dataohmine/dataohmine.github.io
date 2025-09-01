"""
Feature Engineering Module
Creates and transforms features for mortgage collections modeling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import logging
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta

class FeatureEngineer:
    """
    Handles feature creation and transformation for collections modeling
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_logging()
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features
        
        Args:
            df: Input DataFrame with datetime columns
            
        Returns:
            DataFrame with additional temporal features
        """
        self.logger.info("Creating temporal features")
        
        df = df.copy()
        
        # Date columns to process
        date_columns = self.config.get('date_columns', ['last_payment_date', 'loan_origination_date'])
        
        for col in date_columns:
            if col in df.columns:
                # Convert to datetime if not already
                df[col] = pd.to_datetime(df[col])
                
                # Extract temporal components
                df[f'{col}_year'] = df[col].dt.year
                df[f'{col}_month'] = df[col].dt.month
                df[f'{col}_quarter'] = df[col].dt.quarter
                df[f'{col}_day_of_week'] = df[col].dt.dayofweek
                df[f'{col}_day_of_month'] = df[col].dt.day
                
                # Days since reference date
                reference_date = pd.to_datetime(self.config.get('reference_date', datetime.now()))
                df[f'days_since_{col}'] = (reference_date - df[col]).dt.days
                
                # Create seasonal features
                df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
                df[f'{col}_is_month_end'] = (df[col].dt.day > 25).astype(int)
                df[f'{col}_is_quarter_end'] = df[col].dt.month.isin([3, 6, 9, 12]).astype(int)
        
        self.logger.info(f"Created temporal features for {len(date_columns)} date columns")
        return df
    
    def create_payment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create payment history and behavior features
        
        Args:
            df: Input DataFrame with payment data
            
        Returns:
            DataFrame with payment-related features
        """
        self.logger.info("Creating payment features")
        
        df = df.copy()
        
        # Payment amount features
        if 'payment_amount' in df.columns:
            df['payment_amount_log'] = np.log1p(df['payment_amount'])
            df['payment_amount_sqrt'] = np.sqrt(df['payment_amount'])
        
        # Payment frequency features
        if 'payment_count_3m' in df.columns and 'payment_count_6m' in df.columns:
            df['payment_frequency_trend'] = df['payment_count_3m'] / (df['payment_count_6m'] + 1)
        
        # Delinquency features
        if 'days_delinquent' in df.columns:
            df['delinquency_severity'] = pd.cut(
                df['days_delinquent'], 
                bins=[0, 30, 60, 90, 120, float('inf')], 
                labels=['current', 'mild', 'moderate', 'severe', 'critical']
            )
            df['is_severely_delinquent'] = (df['days_delinquent'] > 90).astype(int)
        
        # Balance and payment ratios
        if 'current_balance' in df.columns and 'original_balance' in df.columns:
            df['balance_ratio'] = df['current_balance'] / (df['original_balance'] + 1)
            df['paid_down_ratio'] = 1 - df['balance_ratio']
        
        if 'monthly_payment' in df.columns and 'monthly_income' in df.columns:
            df['payment_to_income_ratio'] = df['monthly_payment'] / (df['monthly_income'] + 1)
        
        self.logger.info("Created payment behavior features")
        return df
    
    def create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create demographic and borrower characteristic features
        
        Args:
            df: Input DataFrame with borrower data
            
        Returns:
            DataFrame with demographic features
        """
        self.logger.info("Creating demographic features")
        
        df = df.copy()
        
        # Age-based features
        if 'birth_date' in df.columns:
            df['borrower_age'] = (pd.to_datetime('today') - pd.to_datetime(df['birth_date'])).dt.days / 365.25
            df['age_group'] = pd.cut(
                df['borrower_age'], 
                bins=[0, 25, 35, 45, 55, 65, 100], 
                labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
            )
        
        # Income features
        if 'annual_income' in df.columns:
            df['income_log'] = np.log1p(df['annual_income'])
            df['income_percentile'] = df['annual_income'].rank(pct=True)
            df['high_income'] = (df['annual_income'] > df['annual_income'].quantile(0.8)).astype(int)
        
        # Geographic features
        if 'zip_code' in df.columns:
            # Extract region information (simplified)
            df['zip_region'] = df['zip_code'].astype(str).str[:1]
            
        # Credit score features
        if 'credit_score' in df.columns:
            df['credit_score_normalized'] = (df['credit_score'] - df['credit_score'].mean()) / df['credit_score'].std()
            df['credit_tier'] = pd.cut(
                df['credit_score'], 
                bins=[0, 580, 670, 740, 800, 850], 
                labels=['poor', 'fair', 'good', 'excellent', 'exceptional']
            )
        
        self.logger.info("Created demographic features")
        return df
    
    def create_economic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create economic environment features
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with economic features
        """
        self.logger.info("Creating economic features")
        
        df = df.copy()
        
        # Mock economic indicators (in practice, these would come from external sources)
        # Unemployment rate by region/time
        df['unemployment_rate'] = np.random.normal(5.0, 1.5, len(df))
        df['unemployment_rate'] = np.clip(df['unemployment_rate'], 0, 15)
        
        # Housing price index
        df['housing_price_index'] = np.random.normal(100, 20, len(df))
        
        # Interest rate environment
        df['interest_rate_environment'] = np.random.choice(['low', 'medium', 'high'], len(df))
        
        # Economic stress indicators
        df['economic_stress_score'] = (
            (df['unemployment_rate'] > 7).astype(int) * 0.4 +
            (df['housing_price_index'] < 80).astype(int) * 0.3 +
            (df['interest_rate_environment'] == 'high').astype(int) * 0.3
        )
        
        self.logger.info("Created economic environment features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame, feature_pairs: List[Tuple[str, str]]) -> pd.DataFrame:
        """
        Create interaction features between specified feature pairs
        
        Args:
            df: Input DataFrame
            feature_pairs: List of feature name tuples to create interactions
            
        Returns:
            DataFrame with interaction features
        """
        self.logger.info(f"Creating {len(feature_pairs)} interaction features")
        
        df = df.copy()
        
        for feat1, feat2 in feature_pairs:
            if feat1 in df.columns and feat2 in df.columns:
                # Multiplicative interaction
                df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
                
                # Ratio interaction (avoid division by zero)
                df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-6)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_columns: List[str]) -> pd.DataFrame:
        """
        Encode categorical features
        
        Args:
            df: Input DataFrame
            categorical_columns: List of categorical column names
            
        Returns:
            DataFrame with encoded categorical features
        """
        self.logger.info(f"Encoding {len(categorical_columns)} categorical features")
        
        df = df.copy()
        
        for col in categorical_columns:
            if col in df.columns:
                # For high cardinality, use target encoding or frequency encoding
                unique_values = df[col].nunique()
                
                if unique_values > 10:
                    # Frequency encoding
                    freq_encoding = df[col].value_counts().to_dict()
                    df[f'{col}_frequency'] = df[col].map(freq_encoding)
                    
                    # Keep only top categories, others as 'Other'
                    top_categories = df[col].value_counts().head(10).index
                    df[f'{col}_grouped'] = df[col].apply(
                        lambda x: x if x in top_categories else 'Other'
                    )
                    col_to_encode = f'{col}_grouped'
                else:
                    col_to_encode = col
                
                # Label encoding
                if col_to_encode not in self.encoders:
                    self.encoders[col_to_encode] = LabelEncoder()
                    df[f'{col}_encoded'] = self.encoders[col_to_encode].fit_transform(df[col_to_encode].astype(str))
                else:
                    df[f'{col}_encoded'] = self.encoders[col_to_encode].transform(df[col_to_encode].astype(str))
        
        return df
    
    def scale_numerical_features(self, df: pd.DataFrame, numerical_columns: List[str]) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            df: Input DataFrame
            numerical_columns: List of numerical column names
            
        Returns:
            DataFrame with scaled numerical features
        """
        self.logger.info(f"Scaling {len(numerical_columns)} numerical features")
        
        df = df.copy()
        
        for col in numerical_columns:
            if col in df.columns:
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df[f'{col}_scaled'] = self.scalers[col].fit_transform(df[[col]])
                else:
                    df[f'{col}_scaled'] = self.scalers[col].transform(df[[col]])
        
        return df
    
    def select_features(self, df: pd.DataFrame, target_column: str, k: int = 100) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select top k features using univariate feature selection
        
        Args:
            df: Input DataFrame with features
            target_column: Name of target column
            k: Number of features to select
            
        Returns:
            Tuple of (DataFrame with selected features, list of selected feature names)
        """
        self.logger.info(f"Selecting top {k} features")
        
        # Separate features and target
        feature_columns = [col for col in df.columns if col != target_column and df[col].dtype in ['int64', 'float64']]
        X = df[feature_columns]
        y = df[target_column]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=min(k, len(feature_columns)))
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = [feature_columns[i] for i in selector.get_support(indices=True)]
        
        # Create DataFrame with selected features
        df_selected = pd.DataFrame(X_selected, columns=selected_features, index=df.index)
        df_selected[target_column] = y
        
        self.feature_names = selected_features
        self.logger.info(f"Selected {len(selected_features)} features")
        
        return df_selected, selected_features
    
    def engineer_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """
        Main feature engineering pipeline
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info("Starting feature engineering pipeline")
        
        # Create temporal features
        df = self.create_temporal_features(df)
        
        # Create payment features
        df = self.create_payment_features(df)
        
        # Create demographic features
        df = self.create_demographic_features(df)
        
        # Create economic features
        df = self.create_economic_features(df)
        
        # Create interaction features
        interaction_pairs = self.config.get('interaction_pairs', [
            ('payment_to_income_ratio', 'credit_score'),
            ('days_delinquent', 'payment_frequency_trend'),
            ('borrower_age', 'loan_amount')
        ])
        df = self.create_interaction_features(df, interaction_pairs)
        
        # Encode categorical features
        categorical_columns = self.config.get('categorical_columns', [
            'delinquency_severity', 'age_group', 'credit_tier', 'zip_region'
        ])
        df = self.encode_categorical_features(df, categorical_columns)
        
        # Scale numerical features
        numerical_columns = self.config.get('numerical_columns', [
            'borrower_age', 'annual_income', 'credit_score', 'loan_amount'
        ])
        df = self.scale_numerical_features(df, numerical_columns)
        
        self.logger.info("Feature engineering pipeline completed")
        return df


def main():
    """Example usage of FeatureEngineer"""
    
    # Configuration
    config = {
        'date_columns': ['last_payment_date', 'loan_origination_date'],
        'categorical_columns': ['state', 'loan_type', 'property_type'],
        'numerical_columns': ['loan_amount', 'annual_income', 'credit_score'],
        'interaction_pairs': [
            ('loan_amount', 'annual_income'),
            ('credit_score', 'loan_amount')
        ]
    }
    
    # Load sample data (replace with actual data loading)
    df = pd.DataFrame({
        'account_id': range(1000),
        'last_payment_date': pd.date_range('2023-01-01', periods=1000, freq='D'),
        'loan_amount': np.random.normal(200000, 50000, 1000),
        'annual_income': np.random.normal(75000, 25000, 1000),
        'credit_score': np.random.normal(700, 100, 1000),
        'target': np.random.binomial(1, 0.3, 1000)
    })
    
    # Initialize feature engineer
    engineer = FeatureEngineer(config)
    
    # Engineer features
    df_engineered = engineer.engineer_features(df, 'target')
    
    # Select top features
    df_selected, selected_features = engineer.select_features(df_engineered, 'target', k=50)
    
    print(f"Feature engineering complete. Selected {len(selected_features)} features from {df_engineered.shape[1]} total features")
    print(f"Final dataset shape: {df_selected.shape}")


if __name__ == "__main__":
    main()