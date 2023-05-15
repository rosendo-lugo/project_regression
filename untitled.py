import pandas as pd

def get_zillow_data():
    # How to import a database from MySQL
    url = get_db_url('zillow')
    query = '''
    select p.bedroomcnt, p.bathroomcnt, p.calculatedfinishedsquarefeet, p.taxvaluedollarcnt, p.yearbuilt, p.fips, p2.transactiondate
    from properties_2017 p
        join predictions_2017 p2 using (parcelid)
    where p.propertylandusetypeid = 261
    order by p2.transactiondate
            '''
    filename = 'zillow.csv'
    df = check_file_exists(filename, query, url)
    
    # rename columns
    df.columns = ['bedrooms', 'bathrooms', 'area', 'property_value', 'yearbuilt', 'county', 'transaction_date']
    
    # Look at properties less than 25,000 sqft
    df = df[df.area < 25000]
    
    # Property value reduce to 95% total
    df = df[df.property_value < df.property_value.quantile(0.95)]
    
    # drop any nulls in the dataset
    df = df.dropna()
    
    # change the dtype from float to int  
    df[['area', 'yearbuilt', 'transaction_date']] = df[['area', 'yearbuilt', 'transaction_date']].astype(int)
    
    # rename the county codes inside county
    df['county'] = df['county'].map({6037: 'LA', 6059: 'Orange', 6111: 'Ventura'})
    
    # write the results to a CSV file
    df.to_csv('df_prep.csv', index=False)

    # read the CSV file into a Pandas dataframe
    prep_df = pd.read_csv('df_prep.csv')
    
    return df, prep_df


def get_zillow_data():
    # How to import a database from MySQL
    url = get_db_url('zillow')
    query = '''
    select p.bedroomcnt, p.bathroomcnt, p.calculatedfinishedsquarefeet, p.taxvaluedollarcnt, p.yearbuilt, p.fips, p2.transactiondate
    from properties_2017 p
        join predictions_2017 p2 using (parcelid)
    where p.propertylandusetypeid = 261
    order by p2.transactiondate
            '''
    filename = 'zillow.csv'
    df = check_file_exists(filename, query, url)
    
    # rename columns
    df.columns
    df = df.rename(columns={'bedroomcnt':'bedrooms', 'bathroomcnt':'bathrooms', 'calculatedfinishedsquarefeet':'area',
       'taxvaluedollarcnt':'property_value', 'fips':'county', 'transactiondate':'transaction_date'})
    
    # Look at properties lessthan 25,000 sqft
    df.area = df.area[df.area < 25_000].copy()
    
    # Property value reduce to 95% total
    df.property_value = df.property_value[df.property_value < df.property_value.quantile(.95)].copy()
    
        
    # drop any nulls in the dataset
    df = df.dropna()
    
    # change the dtype from float to int  
    df.area = df.area.astype(int).copy()
    df.county = df.county.astype(int).copy()
    df.yearbuilt = df.yearbuilt.astype(int).copy()
    df.transaction_date = df.transaction_date.astype(int).copy()
    
    # renamed the county codes inside countyß
    df.county = df.county.map({6037:'LA', 6059:'Orange', 6111:'Ventura'})
    
    # write the results to a CSV file
    df.to_csv('df_prep.csv', index=False)

    # read the CSV file into a Pandas dataframe
    prep_df = pd.read_csv('df_prep.csv')
    
    return df, prep_df