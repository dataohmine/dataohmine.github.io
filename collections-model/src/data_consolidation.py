"""
Data Consolidation Module
Handles data ingestion and initial consolidation from multiple sources
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict, Optional
import sqlite3
from datetime import datetime

class DataConsolidator:
    """
    Consolidates mortgage collections data from multiple sources
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_logging()
        self.consolidated_data = None
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def load_source_data(self, source_path: str, source_type: str) -> pd.DataFrame:
        """
        Load data from various source types
        
        Args:
            source_path: Path to data source
            source_type: Type of source (csv, excel, sql, api)
            
        Returns:
            DataFrame with loaded data
        """
        self.logger.info(f"Loading data from {source_path} (type: {source_type})")
        
        try:
            if source_type == 'csv':
                return pd.read_csv(source_path)
            elif source_type == 'excel':
                return pd.read_excel(source_path)
            elif source_type == 'sql':
                return self._load_from_sql(source_path)
            elif source_type == 'api':
                return self._load_from_api(source_path)
            else:
                raise ValueError(f"Unsupported source type: {source_type}")
                
        except Exception as e:
            self.logger.error(f"Error loading data from {source_path}: {str(e)}")
            raise
    
    def _load_from_sql(self, connection_string: str) -> pd.DataFrame:
        """Load data from SQL database"""
        query = self.config.get('sql_query', 'SELECT * FROM collections_data')
        return pd.read_sql(query, connection_string)
    
    def _load_from_api(self, api_endpoint: str) -> pd.DataFrame:
        """Load data from API endpoint"""
        # Implementation would depend on specific API
        # This is a placeholder for API data loading
        pass
    
    def standardize_schema(self, df: pd.DataFrame, source_name: str) -> pd.DataFrame:
        """
        Standardize column names and data types across sources
        
        Args:
            df: Input DataFrame
            source_name: Name of the data source
            
        Returns:
            DataFrame with standardized schema
        """
        self.logger.info(f"Standardizing schema for source: {source_name}")
        
        # Get column mapping for this source
        column_mapping = self.config.get('column_mappings', {}).get(source_name, {})
        
        # Rename columns
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        # Standardize data types
        type_mapping = self.config.get('data_types', {})
        for column, dtype in type_mapping.items():
            if column in df.columns:
                try:
                    if dtype == 'datetime':
                        df[column] = pd.to_datetime(df[column])
                    else:
                        df[column] = df[column].astype(dtype)
                except Exception as e:
                    self.logger.warning(f"Could not convert {column} to {dtype}: {str(e)}")
        
        # Add source identifier
        df['data_source'] = source_name
        df['load_timestamp'] = datetime.now()
        
        return df
    
    def validate_data_quality(self, df: pd.DataFrame) -> Dict:
        """
        Perform data quality checks
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info("Performing data quality validation")
        
        validation_results = {
            'total_records': len(df),
            'duplicate_records': df.duplicated().sum(),
            'missing_values': df.isnull().sum().to_dict(),
            'data_quality_score': 0.0
        }
        
        # Calculate data quality score
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        duplicate_ratio = validation_results['duplicate_records'] / len(df)
        
        validation_results['data_quality_score'] = 1.0 - (missing_ratio + duplicate_ratio)
        
        self.logger.info(f"Data quality score: {validation_results['data_quality_score']:.3f}")
        
        return validation_results
    
    def consolidate_sources(self, source_configs: List[Dict]) -> pd.DataFrame:
        """
        Consolidate data from multiple sources
        
        Args:
            source_configs: List of source configuration dictionaries
            
        Returns:
            Consolidated DataFrame
        """
        self.logger.info(f"Consolidating {len(source_configs)} data sources")
        
        consolidated_dfs = []
        
        for source_config in source_configs:
            # Load data
            df = self.load_source_data(
                source_config['path'], 
                source_config['type']
            )
            
            # Standardize schema
            df = self.standardize_schema(df, source_config['name'])
            
            # Validate quality
            quality_results = self.validate_data_quality(df)
            self.logger.info(f"Source {source_config['name']}: {quality_results['total_records']} records")
            
            consolidated_dfs.append(df)
        
        # Combine all sources
        self.consolidated_data = pd.concat(consolidated_dfs, ignore_index=True)
        
        # Remove duplicates based on key columns
        key_columns = self.config.get('deduplication_keys', ['account_id', 'loan_number'])
        available_keys = [col for col in key_columns if col in self.consolidated_data.columns]
        
        if available_keys:
            before_dedup = len(self.consolidated_data)
            self.consolidated_data = self.consolidated_data.drop_duplicates(subset=available_keys)
            after_dedup = len(self.consolidated_data)
            self.logger.info(f"Removed {before_dedup - after_dedup} duplicate records")
        
        self.logger.info(f"Final consolidated dataset: {len(self.consolidated_data)} records")
        
        return self.consolidated_data
    
    def save_consolidated_data(self, output_path: str, format: str = 'parquet'):
        """
        Save consolidated data to file
        
        Args:
            output_path: Path to save the data
            format: Output format (parquet, csv, excel)
        """
        if self.consolidated_data is None:
            raise ValueError("No consolidated data to save. Run consolidate_sources first.")
        
        self.logger.info(f"Saving consolidated data to {output_path}")
        
        if format == 'parquet':
            self.consolidated_data.to_parquet(output_path, index=False)
        elif format == 'csv':
            self.consolidated_data.to_csv(output_path, index=False)
        elif format == 'excel':
            self.consolidated_data.to_excel(output_path, index=False)
        else:
            raise ValueError(f"Unsupported output format: {format}")
        
        self.logger.info("Data saved successfully")


def main():
    """Example usage of DataConsolidator"""
    
    # Configuration
    config = {
        'column_mappings': {
            'system_a': {
                'acc_id': 'account_id',
                'loan_num': 'loan_number',
                'amt_due': 'amount_due'
            },
            'system_b': {
                'account': 'account_id',
                'loan': 'loan_number',
                'balance': 'amount_due'
            }
        },
        'data_types': {
            'account_id': 'str',
            'loan_number': 'str',
            'amount_due': 'float',
            'last_payment_date': 'datetime'
        },
        'deduplication_keys': ['account_id', 'loan_number']
    }
    
    # Source configurations
    sources = [
        {
            'name': 'system_a',
            'path': 'data/system_a_data.csv',
            'type': 'csv'
        },
        {
            'name': 'system_b',
            'path': 'data/system_b_data.csv',
            'type': 'csv'
        }
    ]
    
    # Initialize consolidator
    consolidator = DataConsolidator(config)
    
    # Consolidate data
    consolidated_df = consolidator.consolidate_sources(sources)
    
    # Save results
    consolidator.save_consolidated_data('data/consolidated_data.parquet')
    
    print(f"Consolidation complete. Final dataset: {len(consolidated_df)} records")


if __name__ == "__main__":
    main()