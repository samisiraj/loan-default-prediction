import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder



def clean_data(df, fit=None, medians=None):
                
    #dropping nans of columns with very low missing value
    LOW_MISSING_SAFE_TO_DROP = ['submission_of_application', 'age', 'loan_purpose', 'neg_ammortization', 'term', 'approv_in_adv', 'loan_limit']
    df = df.dropna(subset=LOW_MISSING_SAFE_TO_DROP)
    
    #dropping columns with direct correlaion with output
    LEAKAGE_COLUMNS_DROP = ['upfront_charges', 'interest_rate_spread', 'rate_of_interest']
    df = df.drop(columns=LEAKAGE_COLUMNS_DROP)
    
    if fit:
        medians = []
        income_median = df['income'].median()
        dtir1_median = df['dtir1'].median()
        pv_median = df['property_value'].median()
        ltv_median =  df['ltv'].median()
        medians = {'income': income_median, 
                   'dtir1': dtir1_median, 
                   'property_value': pv_median, 
                   'ltv': ltv_median}
    
    #filling income nans with median
    df['income'] = df['income'].fillna(medians['income'])
    
    #flag column for missing dtir1 values
    df['dtir1_was_missing'] = df['dtir1'].isnull().astype('int')
    
    #filling debt to income ratio nans with median
    df['dtir1'] = df['dtir1'].fillna(medians['dtir1'])
    
    #flag column for missing property value
    df['property_value_was_missing'] = df['property_value'].isnull().astype('int')
    
    #filling with median ltv and property value
    df['property_value'] = df['property_value'].fillna(medians['property_value'])
    df['ltv'] = df['ltv'].fillna(medians['ltv'])

    df = df.drop(columns=['id', 'year'])
    
    return df, medians
    #dtir1 debt to income ratio still nans 23k
    #ltv loan to property value ratio 15k
    #property value 15k
    #loanlimit 3k
    #approv_in_adv 908


def prepare_data(df, encoder=None, fit=False):
    
    cat_columns = ['loan_limit', 'gender', 'approv_in_adv', 'loan_type', 'loan_purpose',
       'credit_worthiness', 'open_credit', 'business_or_commercial',
       'neg_ammortization', 'interest_only', 'lump_sum_payment',
       'construction_type', 'occupancy_type', 'secured_by', 'total_units',
       'credit_type', 'co-applicant_credit_type', 'age',
       'submission_of_application', 'region', 'security_type',
       'dtir1_was_missing', 'property_value_was_missing']
    
    num_columns = ['loan_amount', 'term', 'property_value', 'income', 'credit_score',
       'ltv', 'dtir1']
    
    if fit:
        encoder = OneHotEncoder(drop='first',
                                handle_unknown='ignore',
                                sparse_output=False)
        encoder.fit(df[cat_columns])
        
    cat_encoded = encoder.transform(df[cat_columns])
    num_values = df[num_columns].values
    feature_names = num_columns + list(encoder.get_feature_names_out(cat_columns))  # both plain lists here, safe

    y = df['status'].values
    X = np.hstack([num_values, cat_encoded])
    return X, y, encoder, feature_names