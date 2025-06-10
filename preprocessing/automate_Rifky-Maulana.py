import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

def load_data(file_path):
    """
    Load raw penguin dataset
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found!")
        return None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_penguins_data(df):
    """
    Comprehensive preprocessing for penguins dataset
    """
    # 1. Copy data
    df_processed = df.copy()
    
    # 2. Handle missing values
    print(f"Missing values before: {df_processed.isnull().sum().sum()}")
    
    # Drop rows with missing values (only ~10 rows)
    df_processed = df_processed.dropna()
    print(f"Missing values after: {df_processed.isnull().sum().sum()}")
    
    # 3. Encode categorical variables
    # Target variable
    label_encoder = LabelEncoder()
    df_processed['species_encoded'] = label_encoder.fit_transform(df_processed['species'])
    
    # Other categorical variables
    df_processed['island_encoded'] = LabelEncoder().fit_transform(df_processed['island'])
    df_processed['sex_encoded'] = LabelEncoder().fit_transform(df_processed['sex'])
    
    # 4. Feature scaling for numerical features
    scaler = StandardScaler()
    numerical_features = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
    df_processed[numerical_features] = scaler.fit_transform(df_processed[numerical_features])
    
    # 5. Select final features
    feature_columns = numerical_features + ['island_encoded', 'sex_encoded']
    X = df_processed[feature_columns]
    y = df_processed['species_encoded']
    
    print(f"Final dataset shape: {df_processed.shape}")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
    
    return X, y, df_processed

def save_processed_data(df_processed, output_path):
    """
    Save processed data to CSV
    """
    try:
        df_processed.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

def main():
    """
    Main function to run the preprocessing pipeline
    """
    # Define file paths
    raw_data_path = '../penguins_raw.csv'  # Adjust path as needed
    processed_data_path = 'penguins_processed.csv'
    
    # Check if raw data exists, if not try alternative paths
    if not os.path.exists(raw_data_path):
        # Try different possible paths
        alternative_paths = [
            'penguins.csv',
            '../penguins.csv',
            'penguins_raw.csv'
        ]
        
        for path in alternative_paths:
            if os.path.exists(path):
                raw_data_path = path
                break
        else:
            print("Raw data file not found. Please ensure the penguins dataset is available.")
            return
    
    print("=== Starting Penguins Data Preprocessing Pipeline ===")
    
    # Step 1: Load data
    print("\n1. Loading raw data...")
    df_raw = load_data(raw_data_path)
    
    if df_raw is None:
        return
    
    # Step 2: Preprocess data
    print("\n2. Preprocessing data...")
    X, y, df_processed = preprocess_penguins_data(df_raw)
    
    # Step 3: Save processed data
    print("\n3. Saving processed data...")
    save_processed_data(df_processed, processed_data_path)
    
    print("\n=== Preprocessing Pipeline Completed Successfully ===")
    print(f"Processed data available at: {processed_data_path}")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

if __name__ == "__main__":
    main()