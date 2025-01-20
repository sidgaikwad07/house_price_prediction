# Data Loader file for House Price Prediction
# Author: Siddhant Gaikwad
# Date: 20 January 2025
# Description: This script is loading the datafiles

import pandas as pd

# defining the path of files
TRAIN_FILE_PATH = r'/Users/sid/Downloads/house_price_prediction/train.csv'
TEST_FILE_PATH = r'/Users/sid/Downloads/house_price_prediction/test.csv'

def load_data(file_path):
    """
    Load the data from the CSV file

    Parameters
    ----------
    file_path : String
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame : Loaded dataset as a Pandas DataFrame.

    """
    try:
        data = pd.read_csv(file_path)
        print(f"Data successfully loaded from {file_path}. Shape: {data.shape}")
        return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}.")
        raise
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        raise
        
def save_data(data, file_path):
    """
    Save a dataset to a CSV file

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to save.
    file_path : String
        Path to save the CSV file.

    Returns
    -------
    None.

    """
    try:
        data.to_csv(file_path, index=False)
        print(f"Data successfully saved to {file_path}.")
    except Exception as e:
        print(f"An error occurred while saving data: {e}")
        raise
    
# Example usage
if __name__ == "__main__":
    # Load train and test datasets
    train_data = load_data(TRAIN_FILE_PATH)
    test_data = load_data(TEST_FILE_PATH)
