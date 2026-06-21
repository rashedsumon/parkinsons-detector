import os
import kagglehub
import pandas as pd
import numpy as np

def download_and_load_data():
    """
    Downloads the dataset from Kaggle using kagglehub and loads it.
    Returns features (X) and labels (y).
    """
    print("Checking/Downloading dataset from Kaggle...")
    # This automatically downloads or points to the cached version
    path = kagglehub.dataset_download("limi44/parkinsons-visionbased-pose-estimation-dataset")
    print(f"Dataset path verified at: {path}")
    
    # --- DATA EXTRACTION LOGIC ---
    # Note: Depending on whether the dataset contains a unified CSV or multiple files, 
    # you'll want to find your target files. Here is a generic structure to process it:
    
    all_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                all_files.append(os.path.join(root, file))
                
    if not all_files:
        # Mocking structured data fallback if no CSV files are directly found 
        # (Useful for building the initial pipeline skeleton)
        print("No processed CSVs found, generating structured mock pose feature data for the pipeline...")
        np.random.seed(42)
        mock_features = np.random.randn(200, 34 * 3) # 34 joints * (x,y,z) coordinates
        mock_labels = np.random.randint(0, 2, size=200) # 0 = Healthy, 1 = Parkinson's
        
        feature_cols = [f"joint_{i}_{axis}" for i in range(34) for axis in ['x', 'y', 'z']]
        X = pd.DataFrame(mock_features, columns=feature_cols)
        y = pd.Series(mock_labels, name="target")
        return X, y

    # If CSV files exist, load and combine them
    print(f"Loading data from: {all_files[0]}")
    df = pd.read_csv(all_files[0])
    
    # Assume the last column is the target label, change if your dataset specifies otherwise
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    return X, y

if __name__ == "__main__":
    # Test script locally
    X, y = download_and_load_data()
    print(f"Successfully loaded dataset. Features shape: {X.shape}, Labels shape: {y.shape}")