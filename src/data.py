import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "Loan_Default.csv"


def get_data(path=DATA_PATH):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    str_cloumns = list(df.dtypes[df.dtypes == 'str'].index)
    for col in str_cloumns:
        df[col] = df[col].str.strip().str.lower().str.replace(' ', '_')
    return df


def split_data(df):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
    return df_train, df_val, df_test, df_full_train

def main():
    df = get_data()
    
    
if __name__ == '__main__':
    main()