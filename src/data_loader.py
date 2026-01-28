import pandas as pd

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.fillna("data/Medicine_Details.csv", inplace=True)
    return df
