"""
Collections Model Package
Advanced mortgage collections predictive analytics system
"""

__version__ = "1.0.0"
__author__ = "Data & AI Engineer"
__email__ = "contact@dataohmine.com"

from .data_consolidation import DataConsolidator
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer
from .ab_testing import ABTester

__all__ = [
    "DataConsolidator",
    "FeatureEngineer", 
    "ModelTrainer",
    "ABTester"
]