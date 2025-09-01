# Data Directory

This directory contains the data files for the Collections Model project.

## Structure

```
data/
├── raw/                    # Raw input data files
│   ├── system_a_data.csv  # Sample: Account data from System A
│   └── system_b_data.csv  # Sample: Account data from System B
├── processed/              # Processed and cleaned data
│   ├── consolidated_data.parquet
│   └── engineered_features.parquet
└── sample/                 # Sample data for testing
    └── sample_data.csv
```

## Data Sources

The system expects data with the following structure:

### Required Columns
- `account_id` (string): Unique account identifier
- `loan_number` (string): Loan reference number
- `amount_due` (float): Outstanding amount
- `last_payment_date` (datetime): Date of last payment
- `credit_score` (float): Credit score (300-850)
- `annual_income` (float): Annual income in USD
- `collection_success` (boolean): Target variable for modeling

### Optional Columns
- `loan_amount` (float): Original loan amount
- `debt_to_income` (float): Debt-to-income ratio
- `employment_status` (string): Employment status
- `state` (string): State/region
- `property_type` (string): Property type for real estate loans

## Usage

1. Place your raw data files in the `raw/` directory
2. Update the `config.json` file with your data source configurations
3. Run the pipeline: `python main.py --steps consolidation features training`

## Sample Data

To get started quickly, you can generate sample data:

```python
from src.data_consolidation import DataConsolidator
# Sample data generation code here
```

**Note**: Never commit actual customer data to version control. Use the provided .gitignore to exclude sensitive files.