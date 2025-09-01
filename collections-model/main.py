


"""
Main execution script for Collections Model
Orchestrates the complete ML pipeline from data consolidation to A/B testing
"""

import argparse
import json
import logging
from pathlib import Path
import pandas as pd
from datetime import datetime

from src.data_consolidation import DataConsolidator
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.ab_testing import ABTester

def setup_logging(log_level: str = 'INFO'):
    """Setup centralized logging"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('collections_model.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def run_data_consolidation(config: dict, logger: logging.Logger) -> pd.DataFrame:
    """Run data consolidation pipeline"""
    logger.info("Starting data consolidation")
    
    consolidator = DataConsolidator(config['data_consolidation'])
    
    # Load and consolidate data sources
    consolidated_df = consolidator.consolidate_sources(config['data_sources'])
    
    # Save consolidated data
    output_path = config['paths']['consolidated_data']
    consolidator.save_consolidated_data(output_path)
    
    logger.info(f"Data consolidation completed. Saved to {output_path}")
    return consolidated_df

def run_feature_engineering(df: pd.DataFrame, config: dict, logger: logging.Logger) -> pd.DataFrame:
    """Run feature engineering pipeline"""
    logger.info("Starting feature engineering")
    
    engineer = FeatureEngineer(config['feature_engineering'])
    
    # Engineer features
    df_engineered = engineer.engineer_features(df, config['target_column'])
    
    # Select top features
    df_selected, selected_features = engineer.select_features(
        df_engineered, 
        config['target_column'],
        k=config['feature_engineering'].get('max_features', 100)
    )
    
    # Save feature data
    output_path = config['paths']['engineered_features']
    df_selected.to_parquet(output_path, index=False)
    
    # Save feature list
    feature_list_path = config['paths']['feature_list']
    with open(feature_list_path, 'w') as f:
        json.dump(selected_features, f, indent=2)
    
    logger.info(f"Feature engineering completed. Selected {len(selected_features)} features")
    return df_selected

def run_model_training(df: pd.DataFrame, config: dict, logger: logging.Logger) -> ModelTrainer:
    """Run model training pipeline"""
    logger.info("Starting model training")
    
    trainer = ModelTrainer(config['model_training'])
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        df, 
        config['target_column'],
        test_size=config['model_training'].get('test_size', 0.2)
    )
    
    # Optimize hyperparameters if requested
    if config['model_training'].get('optimize_hyperparameters', True):
        logger.info("Optimizing hyperparameters")
        best_params = trainer.optimize_hyperparameters(
            X_train, y_train,
            n_trials=config['model_training'].get('optimization_trials', 100)
        )
    else:
        best_params = config['model_training'].get('default_params', {})
    
    # Train model
    model = trainer.train_model(X_train, y_train, params=best_params)
    
    # Evaluate model
    results = trainer.evaluate_model(X_test, y_test)
    
    # Cross-validation
    cv_results = trainer.cross_validate(
        df.drop(config['target_column'], axis=1), 
        df[config['target_column']]
    )
    
    # Save model and results
    model_path = config['paths']['model']
    metadata_path = config['paths']['model_metadata']
    trainer.save_model(model_path, metadata_path)
    
    logger.info(f"Model training completed. Test AUC: {results['auc_score']:.4f}")
    return trainer

def run_ab_testing(df: pd.DataFrame, config: dict, logger: logging.Logger) -> dict:
    """Run A/B testing simulation"""
    logger.info("Starting A/B testing simulation")
    
    tester = ABTester(config['ab_testing'])
    
    # Design experiment
    experiment_design = tester.design_experiment(
        total_sample_size=len(df),
        expected_effect_size=config['ab_testing'].get('expected_effect_size', 0.2)
    )
    
    # Randomize assignment
    df_assigned = tester.randomize_assignment(
        df,
        stratify_columns=config['ab_testing'].get('stratify_columns', [])
    )
    
    # Check balance
    balance_results = tester.analyze_balance(
        df_assigned,
        'test_group',
        config['ab_testing'].get('balance_columns', [])
    )
    
    # Simulate treatment effect for demonstration
    # In practice, this would be actual observed outcomes
    df_assigned['simulated_outcome'] = df_assigned.apply(
        lambda row: simulate_treatment_effect(row, config['ab_testing']),
        axis=1
    )
    
    # Analyze results
    results = tester.analyze_results(
        df_assigned,
        'test_group',
        'simulated_outcome',
        outcome_type=config['ab_testing'].get('outcome_type', 'binary')
    )
    
    # Check for SRM
    srm_results = tester.calculate_sample_ratio_mismatch(df_assigned, 'test_group')
    
    # Generate report
    report = tester.generate_report(results, balance_results, srm_results)
    
    # Save results
    results_path = config['paths']['ab_test_results']
    with open(results_path, 'w') as f:
        json.dump({
            'experiment_design': experiment_design,
            'results': results,
            'balance_results': balance_results,
            'srm_results': srm_results
        }, f, indent=2, default=str)
    
    # Save report
    report_path = config['paths']['ab_test_report']
    with open(report_path, 'w') as f:
        f.write(report)
    
    logger.info("A/B testing completed")
    return results

def simulate_treatment_effect(row: pd.Series, ab_config: dict) -> int:
    """Simulate treatment effect for A/B testing demonstration"""
    import numpy as np
    
    base_probability = ab_config.get('base_success_rate', 0.3)
    treatment_lift = ab_config.get('treatment_lift', 0.15)
    
    if row['test_group'] == 'treatment':
        probability = base_probability * (1 + treatment_lift)
    else:
        probability = base_probability
    
    return np.random.binomial(1, probability)

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Collections Model Pipeline')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to configuration file')
    parser.add_argument('--steps', nargs='+', 
                       choices=['consolidation', 'features', 'training', 'testing', 'all'],
                       default=['all'], help='Pipeline steps to run')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    logger.info("Starting Collections Model Pipeline")
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Create output directories
    for path_key, path_value in config['paths'].items():
        Path(path_value).parent.mkdir(parents=True, exist_ok=True)
    
    # Initialize data
    df = None
    
    # Run pipeline steps
    if 'all' in args.steps or 'consolidation' in args.steps:
        df = run_data_consolidation(config, logger)
    
    if 'all' in args.steps or 'features' in args.steps:
        if df is None:
            df = pd.read_parquet(config['paths']['consolidated_data'])
        df = run_feature_engineering(df, config, logger)
    
    if 'all' in args.steps or 'training' in args.steps:
        if df is None:
            df = pd.read_parquet(config['paths']['engineered_features'])
        trainer = run_model_training(df, config, logger)
    
    if 'all' in args.steps or 'testing' in args.steps:
        if df is None:
            df = pd.read_parquet(config['paths']['engineered_features'])
        ab_results = run_ab_testing(df, config, logger)
    
    logger.info("Pipeline execution completed successfully")

if __name__ == "__main__":
    main()