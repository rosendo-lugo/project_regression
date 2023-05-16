import pandas as pd
import numpy as np
from env import get_db_url
import os

# Stats
from scipy import stats
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector


# ----------------------------------------------------------------------------------
def check_file_exists(fn, query, url):
    """
    check if file exists in my local directory, if not, pull from sql db
    return dataframe
    """
    if os.path.isfile(fn):
        print('csv file found and loaded')
        return pd.read_csv(fn, index_col=0)
    else: 
        print('creating df and exporting csv')
        df = pd.read_sql(query, url)
        df.to_csv(fn)
        return df

    
# ----------------------------------------------------------------------------------
def get_zillow_data():
    # How to import a database from MySQL
    url = get_db_url('zillow')
    query = '''
    select p.parcelid, p.bedroomcnt, p.bathroomcnt, p.calculatedfinishedsquarefeet, p.taxvaluedollarcnt, p.yearbuilt, p.fips, p2.transactiondate
    from properties_2017 p
        join predictions_2017 p2 using (parcelid)
    where p.propertylandusetypeid = 261 and 279
            '''
    filename = 'zillow.csv'
    df = check_file_exists(filename, query, url)
    
    # filter to just 2017 transactions
    df = df[df['transactiondate'].str.startswith("2017", na=False)]
    
    # split transaction date to year, month, and day
    df_split = df['transactiondate'].str.split(pat='-', expand=True).add_prefix('transaction_')
    df = pd.concat([df.iloc[:, :40], df_split, df.iloc[:, 40:]], axis=1)
    
    # Drop duplicate rows in column: 'parcelid', keeping max transaction date
    df = df.drop_duplicates(subset=['parcelid'])
    
    # rename columns
    df.columns
    df = df.rename(columns={'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms',
                            'calculatedfinishedsquarefeet':'area','taxvaluedollarcnt':'property_value',
                            'fips':'county','transaction_0':'transaction_year',
                            'transaction_1':'transaction_month','transaction_2':'transaction_day'})
    
    # Look at properties less than 6,000 sqft
    df = df[df.area < 6000]
    
    # Property value reduce to 95% total
    df = df[df.property_value < df.property_value.quantile(0.95)]
    
    # replace missing values with "0"
    df = df.fillna({'bedrooms':0,'bathrooms':0,'area':0,'property_value':0,'county':0})
    
    # drop any nulls in the dataset
    df = df.dropna()
    
    # drop all duplicates
    df = df.drop_duplicates(subset=['parcelid'])
    
    # change the dtype from float to int  
    df[['bedrooms','area','property_value','yearbuilt','transaction_month','transaction_day']] = df[['bedrooms','area','property_value','yearbuilt','transaction_month','transaction_day']].astype(int)
    
    # rename the county codes inside county
    df['county'] = df['county'].map({6037: 'LA', 6059: 'Orange', 6111: 'Ventura'})
    
    # get dummies and concat to the dataframe
    dummy_tips = pd.get_dummies(df[['county']], dummy_na=False, drop_first=[True, True])
    df = pd.concat([df, dummy_tips], axis=1)
    
    # dropping these columns for right now until I find a use for them
    df = df.drop(columns =['parcelid','transactiondate','transaction_year','transaction_month','transaction_day'])
    
    # Define the desired column order
    new_column_order = ['bedrooms','bathrooms','area','yearbuilt','county','county_Orange','county_Ventura','property_value',]

    # Reindex the DataFrame with the new column order
    df = df.reindex(columns=new_column_order)

    # write the results to a CSV file
    df.to_csv('df_prep.csv', index=False)

    # read the CSV file into a Pandas dataframe
    prep_df = pd.read_csv('df_prep.csv')
    
    return df, prep_df

# ----------------------------------------------------------------------------------
def get_split(df):
    '''
    train=tr
    validate=val
    test=ts
    test size = .2 and .25
    random state = 123
    '''  
    # split your dataset
    train_validate, ts = train_test_split(df, test_size=.2, random_state=123)
    tr, val = train_test_split(train_validate, test_size=.25, random_state=123)
    
    return tr, val, ts




