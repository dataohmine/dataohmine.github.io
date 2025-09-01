"""
Insurance Policy Lapse Prediction - Data Processing Module

This module handles data consolidation, cleaning, and preprocessing for insurance policy lapse analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional

class DataProcessor:
    """
    Handles data processing for insurance policy lapse prediction.
    
    This class provides methods for:
    - Data consolidation from multiple sources (PFMC, ECIW, Clarify)
    - Data quality filtering and cleaning
    - Basic preprocessing and validation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize DataProcessor with configuration.
        
        Args:
            config: Configuration dictionary with processing parameters
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
    
    def load_pfmc_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and process PFMC (Policy and Financial Management Core) data.
        
        Args:
            file_path: Path to PFMC data file
            
        Returns:
            Processed PFMC DataFrame
        """
        self.logger.info(f"Loading PFMC data from {file_path}")
        
        # Load data (assuming parquet format for efficiency)
        df = pd.read_parquet(file_path)
        
        # Create date columns
        df['pfmc_cur_month'] = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2)
        
        # Add status lag features
        df = self._add_status_lags(df)
        
        self.logger.info(f"Loaded PFMC data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def load_eciw_data(self, file_path: str) -> pd.DataFrame:
        """
        Load and process ECIW (External Customer Information Warehouse) data.
        
        Args:
            file_path: Path to ECIW data file
            
        Returns:
            Processed ECIW DataFrame
        """
        self.logger.info(f"Loading ECIW data from {file_path}")
        
        df = pd.read_parquet(file_path)
        
        # Process date columns
        df['as_of_hist_dt'] = pd.to_datetime(df['as_of_hist_dt'])
        df['eciw_cur_month'] = df['as_of_hist_dt'].dt.strftime('%Y-%m')
        df['eciw_prev_month'] = (df['as_of_hist_dt'] - pd.DateOffset(months=1)).dt.to_period("M").astype(str)
        
        self.logger.info(f"Loaded ECIW data: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def _add_status_lags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add lagged status indicators for historical analysis."""
        
        # Sort by policy and month for proper lag calculation
        df = df.sort_values(['policy_id', 'year', 'month'])
        
        # Add lag features for status analysis
        status_cols = ['surrender_ind', 'reinstate_ind', 'current_status']
        
        for col in status_cols:
            if col in df.columns:
                for lag in [1, 2, 3]:
                    df[f'{col}_lag{lag}'] = df.groupby('policy_id')[col].shift(lag)
        
        return df
    
    def merge_datasets(self, pfmc_df: pd.DataFrame, eciw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge PFMC and ECIW datasets on policy ID and month.
        
        Args:
            pfmc_df: Processed PFMC DataFrame
            eciw_df: Processed ECIW DataFrame
            
        Returns:
            Merged DataFrame
        """
        self.logger.info("Merging PFMC and ECIW datasets")
        
        # Merge on policy ID and month
        merged_df = pfmc_df.merge(
            eciw_df, 
            how='left',
            left_on=['agmt_pkge_id', 'pfmc_cur_month'], 
            right_on=['agmt_pkge_id', 'eciw_cur_month'],
            suffixes=('', '_remove')
        )
        
        # Remove duplicate columns from merge
        cols_to_drop = [col for col in merged_df.columns if 'remove' in col]
        merged_df.drop(columns=cols_to_drop, inplace=True)
        
        self.logger.info(f"Merged dataset: {merged_df.shape[0]} rows, {merged_df.shape[1]} columns")
        return merged_df
    
    def apply_data_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data quality filters to remove invalid records.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """
        self.logger.info("Applying data quality filters")
        initial_rows = len(df)
        
        # Filter deceased policies (keep only alive policies)
        df = df[df['deceased_ind'] != 2]
        
        # Remove policies with zero base face amount
        df = df[df['base_face_amt'] != 0]
        
        # Calculate policy age and remove negative values
        df['policy_age'] = np.round(
            (pd.to_datetime(df['val_dt']) - pd.to_datetime(df['issue_dt'])) / np.timedelta64(1, 'M'), 
            0
        ).astype(int)
        df = df[df['policy_age'] >= 0]
        
        final_rows = len(df)
        self.logger.info(f"Filtered {initial_rows - final_rows} rows. Remaining: {final_rows}")
        
        return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for lapse prediction.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with target variables
        """
        self.logger.info("Creating target variables")
        
        # Rename status columns for clarity
        df['3mo_ahead_status'] = df['cur_status']
        df['current_status'] = df['prior_3mo_status']
        
        # Create binary lapse indicators
        df['3mo_ahead_Lapse'] = np.where(
            (df['3mo_ahead_status'] == 'lapse') | (df['3mo_ahead_status'] == 'surrender'), 
            1, 0
        )
        
        df['current_lapse_status'] = np.where(
            (df['current_status'] == 'lapse') | (df['current_status'] == 'surrender'), 
            1, 0
        )
        
        return df
    
    def remove_sensitive_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove sensitive or unnecessary columns for public dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with sensitive columns removed
        """
        # Define columns to remove (sensitive, null, or unnecessary)
        cols_to_drop = [
            # Sensitive identifiers
            'agmt_pkge_id', 'policy_id', 'ssn', 'customer_id',
            
            # Specific product codes that might be proprietary
            'lp_cntr_cd', 'lp_cntr_rsn_cd', 'lp_bill_cd', 'lp_bill_rsn_cd',
            
            # Mostly null or constant columns
            'ul', 'term', 'vul', 'series', 'debit_indicator',
            'reinstmt_dt', 'end_dt', 'year',
            
            # Geographic specifics
            'rptg_country', 'place_abbrev',
            
            # Detailed product classifications
            'closed_block_stat', 'wsc_ind', 'policy_part', 'cvrg_typ',
            'prod_type', 'ins_nature_cd',
            
            # Secondary insured information
            'issue_age_2nd', 'extra_ratings_2nd', 'imp_rating_2nd',
            'occ_rating_2nd', 'smoking_habit_2nd', 'pref_rating_2nd',
            'select_rating_2nd', 'gender_2nd', 'undw_class_2nd',
            'attained_age_2nd',
            
            # Detailed benefit and rider information
            'has_sps_ind', 'has_chld_ind', 'db_pattern_cd', 'liv_num',
            'frst_scnd_dth_ind', 'lim_prem_sig_ind', 'term_cvrg_sig_ind',
            
            # Complex product features
            'prem_pay_dur', 'prem_pay_age_num', 'cvrg_dur', 'cvrg_age_num',
            'conv_sig_ind', 'conv_dur', 'conv_age_num', 'ins_type_code',
            'trad_intsens_code', 'conv_pct', 'lvl_prem_prd_num',
            'enhnc_comp_ind', 'sngl_pay_ind', 'lvl_prem_prd_age',
            
            # Specific face amounts for different riders
            'aip_face_amt', 'pua_face_amt', 'puar_face_amt', 'oyt_face_amt',
            'designer_face_amt', 'rdr4_face_amt',
            
            # Company-specific codes
            'holding_co_cd', 'major_lob', 'prod_line_desc', 'prod_desc',
            'acord_prod_typ_cd', 'acord_prod_typ_desc', 'limra_lob_desc',
            
            # Detailed status fields
            'prior_3mo_status', 'cur_status'  # Keep aggregated versions only
        ]
        
        # Remove columns that exist in the dataframe
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        df_cleaned = df.drop(columns=existing_cols_to_drop)
        
        self.logger.info(f"Removed {len(existing_cols_to_drop)} sensitive/unnecessary columns")
        return df_cleaned
    
    def process_pipeline(self, pfmc_path: str, eciw_path: str, output_path: str) -> pd.DataFrame:
        """
        Execute complete data processing pipeline.
        
        Args:
            pfmc_path: Path to PFMC data
            eciw_path: Path to ECIW data  
            output_path: Path to save processed data
            
        Returns:
            Processed DataFrame
        """
        self.logger.info("Starting data processing pipeline")
        
        # Load datasets
        pfmc_df = self.load_pfmc_data(pfmc_path)
        eciw_df = self.load_eciw_data(eciw_path)
        
        # Merge datasets
        merged_df = self.merge_datasets(pfmc_df, eciw_df)
        
        # Apply filters and create targets
        filtered_df = self.apply_data_filters(merged_df)
        final_df = self.create_target_variables(filtered_df)
        
        # Remove sensitive information
        clean_df = self.remove_sensitive_columns(final_df)
        
        # Save processed data
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        clean_df.to_parquet(output_path, index=False)
        
        self.logger.info(f"Pipeline complete. Saved {len(clean_df)} rows to {output_path}")
        return clean_df


def create_mock_data(output_path: str, n_policies: int = 10000) -> pd.DataFrame:
    """
    Create mock insurance policy data for demonstration purposes.
    
    Args:
        output_path: Path to save mock data
        n_policies: Number of policies to generate
        
    Returns:
        Mock DataFrame
    """
    np.random.seed(42)
    
    # Generate policy IDs
    policy_ids = [f"POL_{i:06d}" for i in range(n_policies)]
    
    # Generate time series data (6 months per policy)
    months = ['2022-01', '2022-02', '2022-03', '2022-04', '2022-05', '2022-06']
    
    data = []
    for policy_id in policy_ids:
        for month in months:
            # Basic policy characteristics
            issue_year = np.random.choice(range(1990, 2020))
            holder_age = np.random.randint(25, 80)
            policy_age = int((2022 - issue_year) * 12 + np.random.randint(1, 12))
            
            # Financial characteristics
            base_face_amt = np.random.choice([50000, 100000, 250000, 500000, 1000000])
            cash_value = max(0, base_face_amt * 0.3 * np.random.uniform(0.1, 2.0))
            modal_premium = base_face_amt * 0.02 * np.random.uniform(0.5, 2.0)
            
            # Status and lapse indicators
            lapse_prob = 0.002 + (0.01 if cash_value < modal_premium else 0)
            is_lapsed = np.random.random() < lapse_prob
            
            data.append({
                'policy_id': policy_id,
                'month': month,
                'issue_year': issue_year,
                'holder_age': holder_age,
                'policy_age': policy_age,
                'base_face_amt': base_face_amt,
                'mpt_tot_act_val': cash_value,
                'modal_prem_bld': modal_premium,
                'tot_loan_prncpl': max(0, cash_value * np.random.uniform(0, 0.3)),
                'gender': np.random.choice(['M', 'F']),
                'smoking_habit': np.random.choice(['N', 'Y'], p=[0.8, 0.2]),
                'prem_mode': np.random.choice(['A', 'S', 'Q', 'M'], p=[0.4, 0.3, 0.2, 0.1]),
                'current_status': 'lapse' if is_lapsed else 'current',
                '3mo_ahead_Lapse': is_lapsed,
                'subj_aum_amt': cash_value * np.random.uniform(0.5, 3.0),
                'wc_tot_asset_amt': cash_value * np.random.uniform(1.0, 5.0)
            })
    
    df = pd.DataFrame(data)
    
    # Save mock data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    
    print(f"Created mock dataset with {len(df)} records saved to {output_path}")
    return df


if __name__ == "__main__":
    # Create mock data for demonstration
    create_mock_data("data/mock_policy_data.parquet")
    
    print("Data processing module ready for use.")