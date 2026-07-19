import pandas as pd


def clean_data(df):
    
    #dropping nans of columns with very low missing value
    LOW_MISSING_SAFE_TO_DROP = ['submission_of_application', 'age', 'loan_purpose', 'neg_ammortization', 'term', 'approv_in_adv', 'loan_limit']
    df = df.dropna(subset=LOW_MISSING_SAFE_TO_DROP)
    
    #dropping columns with direct correlaion with output
    LEAKAGE_COLUMNS_DROP = ['upfront_charges', 'interest_rate_spread', 'rate_of_interest']
    df = df.drop(columns=LEAKAGE_COLUMNS_DROP)
    
    #filling income nans with median
    df['income'] = df['income'].fillna(df['income'].median())
    
    #flag column for missing dtir1 values
    df['dtir1_was_missing'] = df['dtir1'].isnull().astype('int')
    
    #filling debt to income ratio nans with median
    df['dtir1'] = df['dtir1'].fillna(df['dtir1'].median())
    
    #fflag column for missing property value
    df['property_value_was_missing'] = df['property_value'].isnull().astype('int')
    
    #filling with median ltv and property value
    df['property_value'] = df['property_value'].fillna(df['property_value'].median())
    df['ltv'] = df['ltv'].fillna(df['ltv'].median())

    df = df.drop(columns=['id', 'year'])
    
    return df
    #dtir1 debt to income ratio still nans 23k
    #ltv loan to property value ratio 15k
    #property value 15k
    #loanlimit 3k
    #approv_in_adv 908

def prepare_data(df):
    pass
