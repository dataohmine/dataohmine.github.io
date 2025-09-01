# Insurance Policy Lapse Prediction System

A comprehensive machine learning framework for predicting insurance policy lapses and surrenders using advanced statistical modeling and risk analytics.

## Overview

This project implements a sophisticated analytical framework to predict insurance policy lapse events 3 months in advance. The system processes policyholder data, performs extensive feature engineering, and applies logistic regression models to identify policies at risk of lapse or surrender.

## Business Problem

Insurance companies face significant challenges with policy lapse and surrender rates, which directly impact:
- **Revenue Stability**: Lost premium income from lapsed policies
- **Customer Retention**: Understanding factors driving policy discontinuation  
- **Risk Management**: Predicting cash flow impacts from surrender events
- **Resource Allocation**: Targeting retention efforts effectively

This system provides early warning signals to enable proactive customer retention strategies.

## Key Features

### Predictive Modeling
- **3-Month Prediction Window**: Forecasts lapse events with sufficient lead time for intervention
- **Logistic Regression Models**: Statistical modeling with interpretable coefficients  
- **Decile Analysis**: Risk segmentation for targeted retention campaigns
- **Model Validation**: Comprehensive performance testing and validation

### Advanced Feature Engineering  
- **Cash Value Analytics**: Modified cash value calculations accounting for loan principals
- **Payment Behavior Analysis**: Actual vs. target payment comparisons over rolling windows
- **Customer Status Tracking**: Historical patterns of current/lapse/reinstatement events
- **Demographic Segmentation**: Age, gender, and regional risk factors
- **Policy Characteristics**: Age, face amount, premium mode, and product features

### Comprehensive Data Pipeline
- **Multi-Source Integration**: PFMC, ECIW, and Clarify data processing
- **Data Quality Controls**: Filtering deceased policies, zero face amounts, and data anomalies
- **Feature Normalization**: Cash values normalized by face amounts for comparability
- **Temporal Aggregation**: Rolling window calculations and lag feature creation

## Technical Architecture

### Data Processing Pipeline
```
Raw Data Sources → Data Consolidation → Feature Engineering → Model Training → Risk Scoring
```

### Core Components
- **Data Aggregation**: Multi-source data integration and harmonization
- **Feature Engineering**: Advanced calculated fields and derived metrics  
- **Exploratory Analysis**: Comprehensive EDA with visualization frameworks
- **Model Development**: Statistical modeling with business interpretability
- **Risk Analytics**: Decile analysis and customer segmentation

## Project Structure

```
lapse-prediction/
├── notebooks/
│   ├── 01_data_processing.ipynb       # Data consolidation and cleaning
│   ├── 02_exploratory_analysis.ipynb  # EDA and feature analysis  
│   ├── 03_feature_engineering.ipynb   # Advanced feature creation
│   ├── 04_model_development.ipynb     # Logistic regression modeling
│   └── 05_risk_analysis.ipynb         # Decile analysis and segmentation
├── src/
│   ├── data_processing.py             # Data pipeline utilities
│   ├── feature_engineering.py         # Feature creation functions
│   ├── model_training.py              # Model training and validation
│   └── risk_analytics.py              # Risk scoring and analysis
├── config/
│   └── model_config.py                # Model parameters and settings
├── requirements.txt                   # Python dependencies
└── README.md                          # Project documentation
```

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/dataohmine/lapse-prediction.git
   cd lapse-prediction
   ```

2. **Create virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Data Processing
```bash
# Run complete data pipeline
python src/data_processing.py --input data/raw --output data/processed

# Feature engineering
python src/feature_engineering.py --data data/processed/consolidated_data.parquet
```

### Model Training
```bash
# Train lapse prediction model
python src/model_training.py --config config/model_config.py --output models/

# Generate risk scores
python src/risk_analytics.py --model models/lapse_model.pkl --data data/scored/
```

### Jupyter Analysis
```bash
# Launch Jupyter for interactive analysis
jupyter notebook notebooks/
```

## Key Features and Methodology

### Target Variable Definition
- **Lapse Events**: Policies transitioning to lapse or surrender status
- **3-Month Prediction**: Forecasting status changes 3 months ahead
- **Binary Classification**: 1 = Lapse/Surrender, 0 = Current/Active

### Feature Categories

#### Financial Metrics
- **Modified Cash Value**: `cash_value - loan_principal`
- **Payment Ratios**: `(cash_value - premium_due) / face_amount`
- **Payment Differences**: `actual_premium - target_premium`
- **Household AUM**: Total assets under management vs. policy cash value

#### Behavioral Indicators  
- **Payment History**: Rolling 3-month payment compliance
- **Status Transitions**: Historical lapse/reinstatement patterns
- **Premium Mode Changes**: Frequency of payment mode modifications
- **Policy Utilization**: Loan usage and cash value accessibility

#### Demographic Segmentation
- **Age Buckets**: Policyholder age grouped into risk segments
- **Issue Year Cohorts**: Policy vintage for cohort analysis
- **Geographic Factors**: Regional risk variations
- **Product Categories**: Universal life, whole life, and term variations

### Model Performance

#### Statistical Metrics
- **Transition Matrix Analysis**: Current → Future status probabilities
- **Decile Performance**: Top decile capture rates and lift metrics
- **Feature Importance**: Coefficient analysis and business interpretation
- **Validation Framework**: Out-of-time testing and stability analysis

#### Business Impact
- **Risk Segmentation**: 10-decile risk scoring for targeted interventions
- **Early Warning System**: 3-month lead time for retention efforts
- **Resource Optimization**: Focus retention spending on highest-risk policies
- **Revenue Protection**: Predictive identification of revenue at risk

## Model Insights

### Key Risk Factors
1. **Low Cash Value Relative to Premiums**: Policies with insufficient cash value to cover premiums
2. **Payment Delinquency Patterns**: Historical late payments and reinstatements  
3. **Policy Age and Vintage**: Newer policies and specific issue year cohorts show higher risk
4. **Premium Mode Instability**: Frequent changes in payment frequency indicate stress
5. **Household Financial Profile**: Lower total AUM relative to policy cash value

### Segmentation Results
- **High-Risk Policies**: Cash value below premium requirements, payment delinquencies
- **Stable Policies**: Consistent payment history, adequate cash value reserves
- **Monitored Policies**: Mixed signals requiring ongoing surveillance

## Technical Specifications

### Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing and array operations
- **scikit-learn**: Machine learning algorithms and validation
- **matplotlib/seaborn**: Statistical visualization and plotting
- **boto3**: AWS data access and cloud integration

### Performance Considerations
- **Memory Optimization**: Efficient data processing for large policy portfolios
- **Feature Selection**: Automated feature importance ranking and selection
- **Model Interpretability**: Business-friendly coefficient interpretation
- **Scalability**: Pipeline design for production deployment

## Business Applications

### Retention Management
- **Targeted Campaigns**: Focus efforts on highest-risk policy segments
- **Customer Outreach**: Proactive contact for policies showing early warning signals
- **Product Modifications**: Offer flexible payment options for at-risk policies
- **Intervention Timing**: 3-month lead time enables meaningful intervention strategies

### Financial Planning
- **Cash Flow Forecasting**: Predict surrender outflows for liquidity management
- **Revenue Projections**: Estimate premium income at risk from lapses
- **Capital Planning**: Reserve calculations incorporating lapse predictions
- **Profitability Analysis**: Customer lifetime value adjusted for lapse probability

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with comprehensive testing
4. Update documentation as needed
5. Submit a pull request

For questions and feature requests, please use the GitHub issue tracker.