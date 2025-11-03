import numpy as np
import pandas as pd
import os

pay_col_numbers = [0, 2, 3, 4, 5, 6]

pay_cols = [f'PAY_{i}' for i in pay_col_numbers]

def prepare_data(input_path="data/raw/UCI_Credit_Card.csv", output_path="data/processed/credit_prepared.csv"):
    df = pd.read_csv(input_path)
    df = prepare_data_df(df)
    df.to_csv(output_path, index=False)
    return df

def prepare_data_df(data_frame):
    data_frame = data_frame.drop_duplicates()
    data_frame.loc[data_frame['EDUCATION'] == 0, 'EDUCATION'] = 6
    data_frame.loc[data_frame['MARRIAGE'] == 0, 'MARRIAGE'] = 3
    data_frame['AGE'] = data_frame['AGE'].astype(int)
    data_frame.loc[:, pay_cols] = data_frame[pay_cols].clip(lower=-1)
    # data_frame = repair_inconsistent_pay_rows(data_frame)
    numeric_cols = data_frame.select_dtypes(include=[np.number]).columns
    data_frame[numeric_cols] = data_frame[numeric_cols].apply(pd.to_numeric, errors='coerce')
    return data_frame

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    prepare_data("data/raw/UCI_Credit_Card.csv", "data/processed/prepared.csv")