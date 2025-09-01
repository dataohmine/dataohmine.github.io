"""
Insurance Policy Lapse Prediction - Feature Engineering Module

This module handles advanced feature creation for insurance policy lapse analysis.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional

class FeatureEngineer:
    """
    Handles feature engineering for insurance policy lapse prediction.
    
    This class provides methods for:
    - Cash value and payment analytics
    - Customer behavior indicators
    - Risk segmentation features
    - Temporal pattern analysis
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize FeatureEngineer with configuration.
        
        Args:
            config: Configuration dictionary with feature parameters
        """
        self.config = config or {}
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def create_cash_value_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create cash value related features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cash value features
        """
        self.logger.info("Creating cash value features")
        
        # Modified cash value (accounting for loans)
        df['mod_cash_value'] = df['mpt_tot_act_val'] - df.get('tot_loan_prncpl', 0)
        
        # Cash value normalized by face amount
        df['cash_value_ratio'] = df['mod_cash_value'] / df['base_face_amt']
        
        # Cash value adequacy (vs. premium requirements)
        df['cash_premium_diff'] = df['mod_cash_value'] - df['modal_prem_bld']
        
        # Cash value adequacy ratio
        df['cash_premium_ratio'] = np.where(
            df['modal_prem_bld'] > 0,
            df['mod_cash_value'] / df['modal_prem_bld'],
            np.inf
        )
        
        return df
    
    def create_payment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create payment behavior features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with payment features
        """
        self.logger.info("Creating payment behavior features")
        
        # Modified payment (including loan interest)
        df['mod_payment'] = df['modal_prem_bld'] + df.get('unpd_loan_int_due', 0)
        
        # Payment affordability indicators
        df['payment_burden'] = df['mod_payment'] / df.get('subj_aum_amt', df['mod_cash_value'])
        
        # Premium mode indicators (convert to risk scores)
        df['prem_mode_risk'] = df['prem_mode'].map({
            'A': 1,  # Annual - lowest risk
            'S': 2,  # Semi-annual
            'Q': 3,  # Quarterly  
            'M': 4   # Monthly - highest risk
        }).fillna(3)
        
        return df
    
    def create_policy_characteristics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create policy characteristic features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with policy features
        """
        self.logger.info("Creating policy characteristic features")
        
        # Policy age in years
        df['policy_age_years'] = df['policy_age'] / 12
        
        # Issue year cohorts
        df['issue_year_cohort'] = pd.cut(
            df['issue_year'],
            bins=[0, 1992, 2000, 2010, 2025],
            labels=['Pre-1992', '1992-2000', '2001-2010', '2011+']
        )
        
        # Face amount tiers
        df['face_amount_tier'] = pd.cut(
            df['base_face_amt'],
            bins=[0, 50000, 100000, 250000, 500000, np.inf],
            labels=['<50K', '50K-100K', '100K-250K', '250K-500K', '500K+']
        )
        
        return df
    
    def create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create demographic features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with demographic features
        """
        self.logger.info("Creating demographic features")
        
        # Age buckets
        df['age_bucket'] = pd.cut(
            df['holder_age'],
            bins=[0, 30, 40, 50, 60, 70, 80, 100],
            labels=['<30', '30-40', '40-50', '50-60', '60-70', '70-80', '80+']
        )
        
        # Gender encoding (if available)
        if 'gender' in df.columns:
            df['gender_code'] = df['gender'].map({'M': 1, 'F': 0}).fillna(0.5)
        
        # Smoking risk (if available)
        if 'smoking_habit' in df.columns:
            df['smoking_risk'] = df['smoking_habit'].map({'Y': 1, 'N': 0}).fillna(0)
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal and behavioral features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with temporal features
        """
        self.logger.info("Creating temporal features")
        
        # Sort for proper temporal calculations
        df = df.sort_values(['policy_id', 'month'])
        
        # Payment stability indicators
        df['payment_stability'] = df.groupby('policy_id')['modal_prem_bld'].transform(
            lambda x: x.std() / (x.mean() + 1e-6)
        )
        
        # Cash value trend
        df['cash_value_trend'] = df.groupby('policy_id')['mod_cash_value'].pct_change()
        
        # Status consistency
        if 'current_status' in df.columns:
            df['status_changes'] = df.groupby('policy_id')['current_status'].transform(
                lambda x: (x != x.shift()).sum()
            )
        
        return df
    
    def create_risk_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create composite risk indicators.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with risk indicators
        """
        self.logger.info("Creating risk indicators")
        
        # Financial stress indicator
        df['financial_stress'] = (
            (df['cash_premium_ratio'] < 1).astype(int) * 2 +
            (df['payment_burden'] > 0.1).astype(int) +
            (df['cash_value_ratio'] < 0.1).astype(int)
        )
        
        # Policy maturity risk
        df['maturity_risk'] = (
            (df['policy_age_years'] < 2).astype(int) * 2 +
            (df['policy_age_years'] > 20).astype(int)
        )
        
        # Premium mode risk composite
        df['premium_risk'] = (
            (df['prem_mode_risk'] > 3).astype(int) +
            (df['payment_stability'] > 0.5).astype(int)
        )
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between key variables.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with interaction features
        """
        self.logger.info("Creating interaction features")
        
        # Age-Cash Value interaction
        df['age_cash_interaction'] = df['holder_age'] * df['cash_value_ratio']
        
        # Policy Age-Payment interaction
        df['policy_payment_interaction'] = df['policy_age_years'] * df['payment_burden']
        
        # Face Amount-Age interaction
        df['face_age_interaction'] = np.log1p(df['base_face_amt']) * df['holder_age']
        
        return df
    
    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with all features
        """
        self.logger.info("Creating all engineered features")
        
        # Apply all feature engineering steps
        df = self.create_cash_value_features(df)
        df = self.create_payment_features(df)
        df = self.create_policy_characteristics(df)
        df = self.create_demographic_features(df)
        df = self.create_temporal_features(df)
        df = self.create_risk_indicators(df)
        df = self.create_interaction_features(df)
        
        # Handle infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        self.logger.info("Feature engineering complete")
        return df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Return feature groups for analysis and selection.
        
        Returns:
            Dictionary of feature groups
        """
        return {
            'cash_value_features': [
                'mod_cash_value', 'cash_value_ratio', 'cash_premium_diff', 
                'cash_premium_ratio'
            ],
            'payment_features': [
                'mod_payment', 'payment_burden', 'prem_mode_risk', 'payment_stability'
            ],
            'policy_features': [
                'policy_age_years', 'issue_year_cohort', 'face_amount_tier'
            ],
            'demographic_features': [
                'age_bucket', 'gender_code', 'smoking_risk'
            ],
            'temporal_features': [
                'cash_value_trend', 'status_changes'
            ],
            'risk_indicators': [
                'financial_stress', 'maturity_risk', 'premium_risk'
            ],
            'interaction_features': [
                'age_cash_interaction', 'policy_payment_interaction', 'face_age_interaction'
            ]
        }


def feature_selection_analysis(df: pd.DataFrame, target_col: str = '3mo_ahead_Lapse') -> pd.DataFrame:
    """
    Perform feature importance analysis for model selection.
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column
        
    Returns:
        DataFrame with feature importance scores
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif
    
    # Select numeric columns for analysis
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Remove columns with too many missing values
    complete_cols = [col for col in numeric_cols if df[col].notna().sum() > len(df) * 0.7]
    
    X = df[complete_cols].fillna(0)
    y = df[target_col]
    
    # Random Forest feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    rf_importance = rf.feature_importances_
    
    # Mutual information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': complete_cols,
        'rf_importance': rf_importance,
        'mutual_info': mi_scores
    })
    
    # Add correlation with target
    correlations = [df[col].corr(df[target_col]) for col in complete_cols]
    importance_df['correlation'] = np.abs(correlations)
    
    # Composite importance score
    importance_df['composite_score'] = (
        importance_df['rf_importance'] + 
        importance_df['mutual_info'] + 
        importance_df['correlation']
    ) / 3
    
    return importance_df.sort_values('composite_score', ascending=False)


if __name__ == "__main__":
    print("Feature engineering module ready for use.")
    
    # Example usage with mock data
    from data_processing import create_mock_data
    
    # Create sample data
    mock_df = create_mock_data("data/mock_policy_data.parquet")
    
    # Initialize feature engineer
    feature_engineer = FeatureEngineer()
    
    # Create features
    featured_df = feature_engineer.create_all_features(mock_df)
    
    print(f"Original features: {len(mock_df.columns)}")
    print(f"Enhanced features: {len(featured_df.columns)}")
    print(f"New features added: {len(featured_df.columns) - len(mock_df.columns)}")