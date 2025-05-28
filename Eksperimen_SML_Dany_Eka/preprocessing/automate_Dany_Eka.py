"""
Automated Data Preprocessing Script for Credit Default Dataset
Author: Ida-Sri-Afiqah
Date: May 21, 2025
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional, List, Any


class CreditDataPreprocessor:
    """
    A class to automate preprocessing for credit default dataset.
    This preprocessor handles missing values, outliers, and prepares the data for model training.
    """
    
    def __init__(self, train_path: str, test_path: str):
        """
        Initialize the preprocessor with paths to the training and testing datasets.
        
        Args:
            train_path (str): Path to the training dataset CSV file
            test_path (str): Path to the testing dataset CSV file
        """
        self.train_path = train_path
        self.test_path = test_path
        self.train_data = None
        self.test_data = None
        self.preprocessed_train = None
        self.preprocessed_test = None
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load training and testing datasets from the specified paths.
        
        Returns:
            Tuple containing training and testing dataframes
        """
        try:
            self.train_data = pd.read_csv(self.train_path)
            self.test_data = pd.read_csv(self.test_path)
            print(f"Training data loaded: {self.train_data.shape}")
            print(f"Testing data loaded: {self.test_data.shape}")
            return self.train_data, self.test_data
        except Exception as e:
            print(f"Error loading data: {e}")
            raise
    
    def explore_data(self, save_plots: bool = False, output_dir: str = 'data/visualization') -> Dict[str, Any]:
        """
        Perform basic exploratory data analysis on the datasets.
        
        Args:
            save_plots (bool): Whether to save plots to disk
            output_dir (str): Directory to save plots if save_plots is True
            
        Returns:
            Dictionary containing EDA results
        """
        if self.train_data is None or self.test_data is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        eda_results = {}
        
        # Create output directory if it doesn't exist
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Basic information
        print("Analyzing training data basic information...")
        eda_results['train_info'] = self.train_data.info()
        eda_results['train_describe'] = self.train_data.describe()
        
        print("Analyzing testing data basic information...")
        eda_results['test_info'] = self.test_data.info()
        eda_results['test_describe'] = self.test_data.describe()
        
        # Missing values
        print("Checking for missing values...")
        eda_results['train_missing'] = self.train_data.isnull().sum()
        eda_results['test_missing'] = self.test_data.isnull().sum()
        
        # Distribution of target variable (only for training data)
        if 'SeriousDlqin2yrs' in self.train_data.columns:
            print("Analyzing target variable distribution...")
            plt.figure(figsize=(6, 4))
            sns.countplot(x='SeriousDlqin2yrs', data=self.train_data)
            plt.title('Distribution of Target Variable')
            plt.xlabel('SeriousDlqin2yrs')
            plt.ylabel('Count')
            
            if save_plots:
                plt.savefig(os.path.join(output_dir, 'target_distribution.png'))
                plt.close()
            else:
                plt.show()
        
        # Correlation matrix
        print("Calculating feature correlations...")
        numerical_cols = self.train_data.select_dtypes(include=np.number).columns.tolist()
        
        if 'SeriousDlqin2yrs' in numerical_cols:
            target_col = 'SeriousDlqin2yrs'
            numerical_cols.remove(target_col)
            correlation_matrix = self.train_data[numerical_cols + [target_col]].corr()
        else:
            correlation_matrix = self.train_data[numerical_cols].corr()
        
        eda_results['correlation_matrix'] = correlation_matrix
        
        # Save correlation matrix as CSV
        if save_plots:
            correlation_matrix.to_csv(os.path.join(output_dir, 'correlation_matrix.csv'))
            
            # Also create a heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
            plt.close()
        
        return eda_results
    
    def handle_missing_values(self) -> None:
        """Handle missing values in the datasets using median imputation."""
    
        
        # For MonthlyIncome
        if 'MonthlyIncome' in self.train_data.columns:
            train_median = self.train_data['MonthlyIncome'].median()
            missing_count = self.train_data['MonthlyIncome'].isnull().sum()
            self.train_data['MonthlyIncome'] = self.train_data['MonthlyIncome'].fillna(train_median)
            print(f"Filled {missing_count} missing MonthlyIncome values in training data with median: {train_median}")
            
            # Use the same median from training for test data
            missing_count = self.test_data['MonthlyIncome'].isnull().sum()
            self.test_data['MonthlyIncome'] = self.test_data['MonthlyIncome'].fillna(train_median)
            print(f"Filled {missing_count} missing MonthlyIncome values in testing data with training median: {train_median}")
        
        # For NumberOfDependents
        if 'NumberOfDependents' in self.train_data.columns:
            train_median = self.train_data['NumberOfDependents'].median()
            missing_count = self.train_data['NumberOfDependents'].isnull().sum()
            self.train_data['NumberOfDependents'] = self.train_data['NumberOfDependents'].fillna(train_median)
            print(f"Filled {missing_count} missing NumberOfDependents values in training data with median: {train_median}")
            
            # Use the same median from training for test data
            missing_count = self.test_data['NumberOfDependents'].isnull().sum()
            self.test_data['NumberOfDependents'] = self.test_data['NumberOfDependents'].fillna(train_median)
            print(f"Filled {missing_count} missing NumberOfDependents values in testing data with training median: {train_median}")
    
    def handle_outliers(self) -> None:
        """Handle outliers in the datasets."""
        # Remove records with age = 0 from training data
        if 'age' in self.train_data.columns:
            before_count = len(self.train_data)
            self.train_data = self.train_data[self.train_data['age'] > 0]
            after_count = len(self.train_data)
            print(f"Removed {before_count - after_count} records with age = 0 from training data")
        
        # Note: We are deliberately not capping extreme values in columns like
        # 'NumberOfTime30-59DaysPastDueNotWorst', 'NumberOfTimes90DaysLate', etc.
        # This decision can be revisited based on model performance.
    
    def remove_duplicates(self) -> None:
        """Remove duplicate records from the datasets."""
        # Training data
        before_count = len(self.train_data)
        self.train_data = self.train_data.drop_duplicates()
        after_count = len(self.train_data)
        print(f"Removed {before_count - after_count} duplicate records from training data")
        
        # Testing data
        before_count = len(self.test_data)
        self.test_data = self.test_data.drop_duplicates()
        after_count = len(self.test_data)
        print(f"Removed {before_count - after_count} duplicate records from testing data")
    
    def create_summary_report(self, output_path: str = 'data/processed/preprocessing_summary.txt') -> None:
        """
        Create a summary report of the preprocessing steps and results.
        
        Args:
            output_path (str): Path to save the summary report
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            f.write("# Preprocessing Summary Report\n\n")
            f.write(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Original Dataset Information\n\n")
            f.write(f"Training data shape: {self.train_data.shape}\n")
            f.write(f"Testing data shape: {self.test_data.shape}\n\n")
            
            f.write("## Preprocessing Steps Applied\n\n")
            f.write("1. Removed unnecessary index columns\n")
            f.write("2. Handled missing values using median imputation\n")
            f.write("3. Removed outliers (records with age = 0)\n")
            f.write("4. Removed duplicate records\n\n")
            
            f.write("## Missing Values Summary\n\n")
            f.write("### Before Preprocessing\n")
            f.write("Training data missing values:\n")
            f.write(str(self.train_data.isnull().sum()) + "\n\n")
            f.write("Testing data missing values:\n")
            f.write(str(self.test_data.isnull().sum()) + "\n\n")
            
            f.write("### After Preprocessing\n")
            f.write("Training data missing values:\n")
            f.write(str(self.preprocessed_train.isnull().sum()) + "\n\n")
            f.write("Testing data missing values:\n")
            f.write(str(self.preprocessed_test.isnull().sum()) + "\n\n")
            
            f.write("## Dataset Statistics After Preprocessing\n\n")
            f.write("### Training Data\n")
            f.write(str(self.preprocessed_train.describe()) + "\n\n")
            f.write("### Testing Data\n")
            f.write(str(self.preprocessed_test.describe()) + "\n\n")
        
        print(f"Summary report saved to: {output_path}")
    
    def save_preprocessed_data(self, train_output_path: str, test_output_path: str) -> None:
        """
        Save preprocessed datasets to specified paths.
        
        Args:
            train_output_path (str): Path to save preprocessed training data
            test_output_path (str): Path to save preprocessed testing data
        """
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(train_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(test_output_path), exist_ok=True)
        
        self.train_data.to_csv(train_output_path, index=False)
        self.test_data.to_csv(test_output_path, index=False)
        print(f"Preprocessed training data saved to: {train_output_path}")
        print(f"Preprocessed testing data saved to: {test_output_path}")
    
    def preprocess(self, save_data: bool = True,
                  train_output_path: str = 'data/processed/cleaned_training.csv',
                  test_output_path: str = 'data/processed/cleaned_testing.csv',
                  create_report: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Execute the complete preprocessing pipeline.
        
        Args:
            save_data (bool): Whether to save the preprocessed data
            train_output_path (str): Path to save preprocessed training data
            test_output_path (str): Path to save preprocessed testing data
            create_report (bool): Whether to create a summary report
            
        Returns:
            Tuple containing preprocessed training and testing dataframes
        """
        print("Starting preprocessing pipeline...")
        
        # Check if data is loaded
        if self.train_data is None or self.test_data is None:
            self.load_data()
        
        # Step 1: Handle missing values
        print("\nStep 1: Handling missing values...")
        self.handle_missing_values()
        
        # Step 2: Handle outliers
        print("\nStep 2: Handling outliers...")
        self.handle_outliers()
        
        # Step 3: Remove duplicates
        print("\nStep 3: Removing duplicates...")
        self.remove_duplicates()
        
        # Save preprocessed data if requested
        if save_data:
            print("\nSaving preprocessed data...")
            self.save_preprocessed_data(train_output_path, test_output_path)
        
        self.preprocessed_train = self.train_data.copy()
        self.preprocessed_test = self.test_data.copy()
        
        # Create summary report if requested
        if create_report:
            print("\nCreating preprocessing summary report...")
            self.create_summary_report()
        
        print("\nPreprocessing completed successfully!")
        return self.preprocessed_train, self.preprocessed_test


def preprocess_credit_data(train_path: str, test_path: str, 
                         save_data: bool = True,
                         train_output_path: str = 'data/processed/cleaned_training.csv',
                         test_output_path: str = 'data/processed/cleaned_testing.csv') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to preprocess credit default dataset.
    
    Args:
        train_path (str): Path to the training dataset CSV file
        test_path (str): Path to the testing dataset CSV file
        save_data (bool): Whether to save the preprocessed data
        train_output_path (str): Path to save preprocessed training data
        test_output_path (str): Path to save preprocessed testing data
        
    Returns:
        Tuple containing preprocessed training and testing dataframes
    """
    preprocessor = CreditDataPreprocessor(train_path, test_path)
    return preprocessor.preprocess(save_data, train_output_path, test_output_path)


if __name__ == "__main__":
    # Define paths - adjust as needed based on your repository structure
    train_path = "data/raw/train.csv"
    test_path = "data/raw/test.csv"
    
    # Check if files exist, if not, try alternative paths
    if not os.path.exists(train_path):
        if os.path.exists("train.csv"):
            train_path = "train.csv"
        else:
            print(f"Training file not found at {train_path}. Searching for alternatives...")
            potential_paths = [
                "./train.csv",
                "../data/raw/train.csv",
                "../train.csv",
                "../../data/raw/train.csv"
            ]
            for path in potential_paths:
                if os.path.exists(path):
                    train_path = path
                    print(f"Found training file at {train_path}")
                    break
            else:
                raise FileNotFoundError("Training file not found!")
    
    if not os.path.exists(test_path):
        if os.path.exists("test.csv"):
            test_path = "test.csv"
        else:
            print(f"Test file not found at {test_path}. Searching for alternatives...")
            potential_paths = [
                "./test.csv",
                "../data/raw/test.csv",
                "../test.csv",
                "../../data/raw/test.csv"
            ]
            for path in potential_paths:
                if os.path.exists(path):
                    test_path = path
                    print(f"Found test file at {test_path}")
                    break
            else:
                raise FileNotFoundError("Test file not found!")
    
    print(f"Using training data from: {train_path}")
    print(f"Using test data from: {test_path}")
    
    # Create preprocessor instance
    preprocessor = CreditDataPreprocessor(train_path, test_path)
    
    # Load data
    preprocessor.load_data()
    
    # Optional: Explore the data and save visualizations
    eda_results = preprocessor.explore_data(save_plots=True)
    
    # Run the entire preprocessing pipeline
    preprocessed_train, preprocessed_test = preprocessor.preprocess()
    
    print(f"Final preprocessed training data shape: {preprocessed_train.shape}")
    print(f"Final preprocessed testing data shape: {preprocessed_test.shape}")