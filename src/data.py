import pandas as pd


def get_data(path='./data/raw/Loan_Default.csv'):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    str_cloumns = list(df.dtypes[df.dtypes == 'str'].index)
    for col in str_cloumns:
        df[col] = df[col].str.strip().str.lower().str.replace(' ', '_')
    return df


def main():
    df = get_data()
    
    
if __name__ == '__main__':
    main()