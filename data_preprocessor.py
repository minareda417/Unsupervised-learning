import pandas as pd
import numpy as np

def preprocess_data(df: pd.DataFrame):
    df.drop(['id', 'Unnamed: 32', 'diagnosis'], axis=1, inplace=True)
    for column in df.select_dtypes(np.number).columns:
        mean = df[column].mean()
        std = df[column].std()
        df[column] = (df[column] - mean) / std
    return df

dataset_path = input("Enter the path to the dataset CSV file: ")
df = pd.read_csv(dataset_path)
df = preprocess_data(df)
print("Recommended to save to ./data_processed.csv:")
processed_path = input("Enter the path to save the processed dataset CSV file: ")
df.to_csv(processed_path, index=False)