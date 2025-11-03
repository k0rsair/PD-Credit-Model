import sys
import pandas as pd
import numpy as np
import os


pay_amt_col_numbers = [1, 2, 3, 4, 5, 6]
pay_col_numbers = [0, 2, 3, 4, 5, 6]

pay_cols = [f'PAY_{i}' for i in pay_col_numbers]
pay_amt_cols = [f'PAY_AMT{i}' for i in pay_amt_col_numbers]
bill_cols = [f'BILL_AMT{i}' for i in pay_amt_col_numbers]


def feature_engineering(df_file):
    # пробую добавить своего рода вес, который ростет за своевременные платежи
    df_file['PAY_WEIGHT'] = (df_file[pay_cols] * -1).sum(axis=1)
    # Биннинг возраста
    df_file['AGE_BINNED'] = pd.cut(df_file['AGE'], bins=[20, 30, 40, 50, 60, 80], labels=False)

    # === Отношение выплат к счетам ===

    df_file["BILL_TOTAL"] = df_file[bill_cols].sum(axis=1)
    df_file["PAY_TOTAL"] = df_file[pay_amt_cols].sum(axis=1)
    df_file["PAY_RATIO"] = np.where(df_file["BILL_TOTAL"] > 0, df_file["PAY_TOTAL"] / df_file["BILL_TOTAL"], 0)
    df_file = df_file.drop(columns=['AGE', 'ID'])
    return df_file

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python feature_engineering.py <input_csv> <output_csv>")
        sys.exit(1)

    input_path, output_path = sys.argv[1], sys.argv[2]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.read_csv(input_path)
    df = feature_engineering(df)
    df.to_csv(output_path, index=False)
    print(f"✅ Feature engineering complete: {output_path}")