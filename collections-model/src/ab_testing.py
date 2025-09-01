"""
A/B Testing Module
Statistical testing framework for evaluating collection strategies
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

class ABTester:
    """
    Handles A/B testing for collection strategies and model performance
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = self._setup_logging()
        self.test_results = {}
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(__name__)
    
    def design_experiment(self, 
                         total_sample_size: int,
                         expected_effect_size: float,
                         alpha: float = 0.05,
                         power: float = 0.8) -> Dict:
        """
        Design A/B test experiment parameters
        
        Args:
            total_sample_size: Total number of subjects available
            expected_effect_size: Expected difference between groups
            alpha: Significance level
            power: Statistical power
            
        Returns:
            Dictionary with experiment design parameters
        """
        self.logger.info("Designing A/B test experiment")
        
        # Calculate required sample size per group
        effect_size = expected_effect_size
        required_n = stats.norm.ppf(1 - alpha/2) + stats.norm.ppf(power)
        required_n = (required_n / effect_size) ** 2
        required_n = int(np.ceil(required_n))
        
        # Check if we have enough samples
        required_total = required_n * 2
        
        experiment_design = {
            'total_available_samples': total_sample_size,
            'required_samples_per_group': required_n,
            'required_total_samples': required_total,
            'can_achieve_power': total_sample_size >= required_total,
            'actual_power': self._calculate_power(total_sample_size // 2, effect_size, alpha),
            'alpha': alpha,
            'expected_effect_size': expected_effect_size,
            'recommended_split': 0.5,
            'minimum_test_duration_days': self.config.get('min_test_duration', 14)
        }
        
        if not experiment_design['can_achieve_power']:
            self.logger.warning(f"Insufficient sample size. Need {required_total}, have {total_sample_size}")
        
        self.logger.info(f"Experiment design completed. Required per group: {required_n}")
        
        return experiment_design
    
    def _calculate_power(self, n_per_group: int, effect_size: float, alpha: float) -> float:
        """Calculate statistical power given sample size"""
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = effect_size * np.sqrt(n_per_group) - z_alpha
        power = stats.norm.cdf(z_beta)
        return power
    
    def randomize_assignment(self, 
                           subjects: pd.DataFrame,
                           group_column: str = 'test_group',
                           stratify_columns: List[str] = None,
                           treatment_ratio: float = 0.5) -> pd.DataFrame:
        """
        Randomly assign subjects to treatment and control groups
        
        Args:
            subjects: DataFrame with subject information
            group_column: Name of column to store group assignments
            stratify_columns: Columns to stratify randomization on
            treatment_ratio: Proportion assigned to treatment group
            
        Returns:
            DataFrame with group assignments
        """
        self.logger.info(f"Randomizing {len(subjects)} subjects to groups")
        
        df = subjects.copy()
        
        if stratify_columns:
            # Stratified randomization
            def assign_within_stratum(group_df):
                n = len(group_df)
                n_treatment = int(n * treatment_ratio)
                assignments = ['treatment'] * n_treatment + ['control'] * (n - n_treatment)
                np.random.shuffle(assignments)
                return assignments
            
            # Group by stratification variables and assign
            df[group_column] = df.groupby(stratify_columns).apply(
                lambda x: pd.Series(assign_within_stratum(x), index=x.index)
            ).droplevel(list(range(len(stratify_columns))))
            
        else:
            # Simple randomization
            n = len(df)
            n_treatment = int(n * treatment_ratio)
            assignments = ['treatment'] * n_treatment + ['control'] * (n - n_treatment)
            np.random.shuffle(assignments)
            df[group_column] = assignments
        
        # Add assignment metadata
        df['assignment_date'] = datetime.now()
        df['experiment_id'] = self.config.get('experiment_id', 'AB_TEST_001')
        
        treatment_count = (df[group_column] == 'treatment').sum()
        control_count = (df[group_column] == 'control').sum()
        
        self.logger.info(f"Assignment complete: {treatment_count} treatment, {control_count} control")
        
        return df
    
    def analyze_balance(self, df: pd.DataFrame, 
                       group_column: str,
                       balance_columns: List[str]) -> Dict:
        """
        Check balance between treatment and control groups
        
        Args:
            df: DataFrame with group assignments
            group_column: Column with group assignments
            balance_columns: Columns to check balance on
            
        Returns:
            Dictionary with balance test results
        """
        self.logger.info("Analyzing group balance")
        
        balance_results = {}
        
        for col in balance_columns:
            if col not in df.columns:
                continue
                
            treatment_data = df[df[group_column] == 'treatment'][col].dropna()
            control_data = df[df[group_column] == 'control'][col].dropna()
            
            if df[col].dtype in ['int64', 'float64']:
                # Numerical variable - use t-test
                statistic, p_value = ttest_ind(treatment_data, control_data)
                test_type = 't-test'
                
                balance_results[col] = {
                    'test_type': test_type,
                    'statistic': statistic,
                    'p_value': p_value,
                    'treatment_mean': treatment_data.mean(),
                    'control_mean': control_data.mean(),
                    'treatment_std': treatment_data.std(),
                    'control_std': control_data.std(),
                    'balanced': p_value > 0.05
                }
            else:
                # Categorical variable - use chi-square test
                contingency_table = pd.crosstab(df[col], df[group_column])
                statistic, p_value, _, _ = chi2_contingency(contingency_table)
                test_type = 'chi-square'
                
                balance_results[col] = {
                    'test_type': test_type,
                    'statistic': statistic,
                    'p_value': p_value,
                    'contingency_table': contingency_table.to_dict(),
                    'balanced': p_value > 0.05
                }
        
        # Overall balance assessment
        balanced_vars = sum([result['balanced'] for result in balance_results.values()])
        total_vars = len(balance_results)
        
        overall_balance = {
            'balanced_variables': balanced_vars,
            'total_variables': total_vars,
            'balance_ratio': balanced_vars / total_vars if total_vars > 0 else 0,
            'well_balanced': (balanced_vars / total_vars) > 0.8 if total_vars > 0 else False
        }
        
        self.logger.info(f"Balance check: {balanced_vars}/{total_vars} variables balanced")
        
        return {'variable_balance': balance_results, 'overall_balance': overall_balance}
    
    def analyze_results(self, df: pd.DataFrame,
                       group_column: str,
                       outcome_column: str,
                       outcome_type: str = 'continuous') -> Dict:
        """
        Analyze A/B test results
        
        Args:
            df: DataFrame with results
            group_column: Column with group assignments
            outcome_column: Column with outcome variable
            outcome_type: Type of outcome ('continuous', 'binary', 'count')
            
        Returns:
            Dictionary with test results
        """
        self.logger.info(f"Analyzing A/B test results for {outcome_column}")
        
        treatment_data = df[df[group_column] == 'treatment'][outcome_column].dropna()
        control_data = df[df[group_column] == 'control'][outcome_column].dropna()
        
        results = {
            'outcome_column': outcome_column,
            'outcome_type': outcome_type,
            'treatment_n': len(treatment_data),
            'control_n': len(control_data),
            'treatment_mean': treatment_data.mean(),
            'control_mean': control_data.mean()
        }
        
        if outcome_type == 'continuous':
            # T-test for continuous outcomes
            statistic, p_value = ttest_ind(treatment_data, control_data)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(treatment_data) - 1) * treatment_data.var() + 
                                 (len(control_data) - 1) * control_data.var()) / 
                                (len(treatment_data) + len(control_data) - 2))
            effect_size = (treatment_data.mean() - control_data.mean()) / pooled_std
            
            # Confidence interval for difference in means
            se_diff = pooled_std * np.sqrt(1/len(treatment_data) + 1/len(control_data))
            t_critical = stats.t.ppf(0.975, len(treatment_data) + len(control_data) - 2)
            mean_diff = treatment_data.mean() - control_data.mean()
            ci_lower = mean_diff - t_critical * se_diff
            ci_upper = mean_diff + t_critical * se_diff
            
            results.update({
                'test_type': 't-test',
                'statistic': statistic,
                'p_value': p_value,
                'effect_size_cohens_d': effect_size,
                'mean_difference': mean_diff,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'treatment_std': treatment_data.std(),
                'control_std': control_data.std()
            })
            
        elif outcome_type == 'binary':
            # Chi-square test for binary outcomes
            treatment_success = treatment_data.sum()
            control_success = control_data.sum()
            
            contingency_table = np.array([
                [treatment_success, len(treatment_data) - treatment_success],
                [control_success, len(control_data) - control_success]
            ])
            
            statistic, p_value, _, _ = chi2_contingency(contingency_table)
            
            # Effect size (relative risk and odds ratio)
            treatment_rate = treatment_success / len(treatment_data)
            control_rate = control_success / len(control_data)
            
            relative_risk = treatment_rate / control_rate if control_rate > 0 else np.inf
            
            odds_treatment = treatment_rate / (1 - treatment_rate) if treatment_rate < 1 else np.inf
            odds_control = control_rate / (1 - control_rate) if control_rate < 1 else np.inf
            odds_ratio = odds_treatment / odds_control if odds_control > 0 else np.inf
            
            results.update({
                'test_type': 'chi-square',
                'statistic': statistic,
                'p_value': p_value,
                'treatment_rate': treatment_rate,
                'control_rate': control_rate,
                'rate_difference': treatment_rate - control_rate,
                'relative_risk': relative_risk,
                'odds_ratio': odds_ratio,
                'treatment_successes': treatment_success,
                'control_successes': control_success
            })
            
        elif outcome_type == 'count':
            # Mann-Whitney U test for count data
            statistic, p_value = mannwhitneyu(treatment_data, control_data, alternative='two-sided')
            
            results.update({
                'test_type': 'mann-whitney-u',
                'statistic': statistic,
                'p_value': p_value,
                'treatment_median': treatment_data.median(),
                'control_median': control_data.median()
            })
        
        # Statistical significance
        alpha = self.config.get('alpha', 0.05)
        results['significant'] = p_value < alpha
        results['alpha'] = alpha
        
        # Practical significance
        if outcome_type == 'continuous':
            min_effect = self.config.get('minimum_effect_size', 0.2)
            results['practically_significant'] = abs(effect_size) > min_effect
        elif outcome_type == 'binary':
            min_lift = self.config.get('minimum_lift', 0.05)
            results['practically_significant'] = abs(treatment_rate - control_rate) > min_lift
        
        self.logger.info(f"Test results: p-value = {p_value:.4f}, significant = {results['significant']}")
        
        return results
    
    def calculate_sample_ratio_mismatch(self, df: pd.DataFrame, group_column: str) -> Dict:
        """
        Check for Sample Ratio Mismatch (SRM)
        
        Args:
            df: DataFrame with group assignments
            group_column: Column with group assignments
            
        Returns:
            Dictionary with SRM test results
        """
        self.logger.info("Checking for Sample Ratio Mismatch")
        
        group_counts = df[group_column].value_counts()
        expected_ratio = self.config.get('expected_ratio', 0.5)
        
        total_samples = len(df)
        expected_treatment = int(total_samples * expected_ratio)
        expected_control = total_samples - expected_treatment
        
        observed_treatment = group_counts.get('treatment', 0)
        observed_control = group_counts.get('control', 0)
        
        # Chi-square goodness of fit test
        observed = [observed_treatment, observed_control]
        expected = [expected_treatment, expected_control]
        
        statistic, p_value = stats.chisquare(observed, expected)
        
        srm_results = {
            'observed_treatment': observed_treatment,
            'observed_control': observed_control,
            'expected_treatment': expected_treatment,
            'expected_control': expected_control,
            'observed_ratio': observed_treatment / total_samples,
            'expected_ratio': expected_ratio,
            'chi_square_statistic': statistic,
            'p_value': p_value,
            'srm_detected': p_value < 0.01,  # Conservative threshold for SRM
            'severity': 'high' if p_value < 0.001 else 'medium' if p_value < 0.01 else 'low'
        }
        
        if srm_results['srm_detected']:
            self.logger.warning(f"Sample Ratio Mismatch detected! p-value: {p_value:.6f}")
        else:
            self.logger.info("No Sample Ratio Mismatch detected")
        
        return srm_results
    
    def generate_report(self, 
                       experiment_results: Dict,
                       balance_results: Dict = None,
                       srm_results: Dict = None) -> str:
        """
        Generate comprehensive A/B test report
        
        Args:
            experiment_results: Results from analyze_results
            balance_results: Results from analyze_balance
            srm_results: Results from calculate_sample_ratio_mismatch
            
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("A/B TEST RESULTS REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append("")
        
        # Sample sizes
        report_lines.append("SAMPLE SIZES:")
        report_lines.append(f"Treatment Group: {experiment_results['treatment_n']:,}")
        report_lines.append(f"Control Group: {experiment_results['control_n']:,}")
        report_lines.append(f"Total: {experiment_results['treatment_n'] + experiment_results['control_n']:,}")
        report_lines.append("")
        
        # Primary results
        report_lines.append("PRIMARY RESULTS:")
        report_lines.append(f"Outcome: {experiment_results['outcome_column']}")
        report_lines.append(f"Test Type: {experiment_results['test_type']}")
        report_lines.append(f"P-value: {experiment_results['p_value']:.6f}")
        report_lines.append(f"Statistically Significant: {'YES' if experiment_results['significant'] else 'NO'}")
        
        if 'practically_significant' in experiment_results:
            report_lines.append(f"Practically Significant: {'YES' if experiment_results['practically_significant'] else 'NO'}")
        
        report_lines.append("")
        
        # Effect size
        if experiment_results['outcome_type'] == 'continuous':
            report_lines.append("EFFECT SIZE:")
            report_lines.append(f"Treatment Mean: {experiment_results['treatment_mean']:.4f}")
            report_lines.append(f"Control Mean: {experiment_results['control_mean']:.4f}")
            report_lines.append(f"Mean Difference: {experiment_results['mean_difference']:.4f}")
            report_lines.append(f"Cohen's d: {experiment_results['effect_size_cohens_d']:.4f}")
            report_lines.append(f"95% CI: [{experiment_results['ci_lower']:.4f}, {experiment_results['ci_upper']:.4f}]")
            
        elif experiment_results['outcome_type'] == 'binary':
            report_lines.append("CONVERSION RATES:")
            report_lines.append(f"Treatment Rate: {experiment_results['treatment_rate']:.4f} ({experiment_results['treatment_rate']*100:.2f}%)")
            report_lines.append(f"Control Rate: {experiment_results['control_rate']:.4f} ({experiment_results['control_rate']*100:.2f}%)")
            report_lines.append(f"Absolute Lift: {experiment_results['rate_difference']:.4f} ({experiment_results['rate_difference']*100:.2f} pp)")
            report_lines.append(f"Relative Lift: {(experiment_results['relative_risk']-1)*100:.2f}%")
            report_lines.append(f"Odds Ratio: {experiment_results['odds_ratio']:.4f}")
        
        report_lines.append("")
        
        # Sample Ratio Mismatch
        if srm_results:
            report_lines.append("SAMPLE RATIO MISMATCH CHECK:")
            report_lines.append(f"Expected Ratio: {srm_results['expected_ratio']:.1%}")
            report_lines.append(f"Observed Ratio: {srm_results['observed_ratio']:.1%}")
            report_lines.append(f"SRM Detected: {'YES' if srm_results['srm_detected'] else 'NO'}")
            if srm_results['srm_detected']:
                report_lines.append(f"Severity: {srm_results['severity'].upper()}")
            report_lines.append("")
        
        # Balance check
        if balance_results:
            report_lines.append("RANDOMIZATION BALANCE:")
            overall = balance_results['overall_balance']
            report_lines.append(f"Balanced Variables: {overall['balanced_variables']}/{overall['total_variables']}")
            report_lines.append(f"Balance Quality: {'GOOD' if overall['well_balanced'] else 'POOR'}")
            report_lines.append("")
        
        # Recommendations
        report_lines.append("RECOMMENDATIONS:")
        if experiment_results['significant']:
            if experiment_results.get('practically_significant', True):
                report_lines.append("✓ IMPLEMENT: Results are both statistically and practically significant")
            else:
                report_lines.append("⚠ CAUTION: Statistically significant but effect size may be too small")
        else:
            report_lines.append("✗ DO NOT IMPLEMENT: Results are not statistically significant")
        
        if srm_results and srm_results['srm_detected']:
            report_lines.append("⚠ INVESTIGATE: Sample ratio mismatch detected - check randomization")
        
        if balance_results and not balance_results['overall_balance']['well_balanced']:
            report_lines.append("⚠ CAUTION: Poor randomization balance detected")
        
        report_lines.append("")
        report_lines.append("=" * 60)
        
        return "\n".join(report_lines)


def main():
    """Example usage of ABTester"""
    
    # Configuration
    config = {
        'experiment_id': 'COLLECTIONS_AB_001',
        'alpha': 0.05,
        'minimum_effect_size': 0.2,
        'minimum_lift': 0.05,
        'expected_ratio': 0.5
    }
    
    # Generate sample data
    np.random.seed(42)
    n_subjects = 10000
    
    # Create subject data
    subjects = pd.DataFrame({
        'subject_id': range(n_subjects),
        'credit_score': np.random.normal(700, 100, n_subjects),
        'loan_amount': np.random.normal(200000, 50000, n_subjects),
        'age': np.random.randint(25, 70, n_subjects),
        'state': np.random.choice(['CA', 'TX', 'NY', 'FL'], n_subjects)
    })
    
    # Initialize AB tester
    tester = ABTester(config)
    
    # Design experiment
    experiment_design = tester.design_experiment(
        total_sample_size=n_subjects,
        expected_effect_size=0.3
    )
    print("Experiment Design:", experiment_design)
    
    # Randomize assignment
    subjects_assigned = tester.randomize_assignment(
        subjects,
        stratify_columns=['state']
    )
    
    # Check balance
    balance_results = tester.analyze_balance(
        subjects_assigned,
        'test_group',
        ['credit_score', 'loan_amount', 'age', 'state']
    )
    
    # Simulate outcomes (treatment has 15% higher success rate)
    subjects_assigned['collection_success'] = np.where(
        subjects_assigned['test_group'] == 'treatment',
        np.random.binomial(1, 0.35, len(subjects_assigned)),  # 35% success rate
        np.random.binomial(1, 0.30, len(subjects_assigned))   # 30% success rate
    )
    
    # Analyze results
    results = tester.analyze_results(
        subjects_assigned,
        'test_group',
        'collection_success',
        outcome_type='binary'
    )
    
    # Check for SRM
    srm_results = tester.calculate_sample_ratio_mismatch(subjects_assigned, 'test_group')
    
    # Generate report
    report = tester.generate_report(results, balance_results, srm_results)
    print(report)


if __name__ == "__main__":
    main()