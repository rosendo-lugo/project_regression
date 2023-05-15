import pandas as pd 
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

# ----------------------------------------------------------------------------------
def get_split(df):
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.25, random_state=123)
    
    return train, validate, test

# ----------------------------------------------------------------------------------

def scale_data(train,validate,test,to_scale):
    """
    to_scale = ['column1','column2','column3','column4','column5']
    """
    
    #make copies for scaling
    train_scaled = train.copy()
    validate_scaled = validate.copy()
    test_scaled = test.copy()

    #scale them!
    #make the thing
    scaler = MinMaxScaler()

    #fit the thing
    scaler.fit(train[to_scale])

    #use the thing
    train_scaled[to_scale] = scaler.transform(train[to_scale])
    validate_scaled[to_scale] = scaler.transform(validate[to_scale])
    test_scaled[to_scale] = scaler.transform(test[to_scale])
    
    return train_scaled, validate_scaled, test_scaled

# ----------------------------------------------------------------------------------
def metrics_reg(y, yhat):
    """
    send in y_true, y_pred & returns RMSE, R2
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2
# ----------------------------------------------------------------------------------
def get_minmax_train_scaler(X_train, X_validate):
    
    columns=['bedrooms', 'bathrooms','area','yearbuilt','taxamount','tax_rate']
    X_tr_mm_scaler = X_train.copy()
    X_v_mm_scaler = X_validate.copy()
    
    # Create a MinMaxScaler object
    mm_scaler = MinMaxScaler()

    # Fit the scaler to the training data and transform it
    X_tr_mm_scaler[columns] = mm_scaler.fit_transform(X_train[columns])
    X_tr_mm_scaler = pd.DataFrame(X_tr_mm_scaler)
    
    #using our scaler on validate
    X_v_mm_scaler[columns] = mm_scaler.transform(X_validate[columns])
    X_v_mm_scaler = pd.DataFrame(X_v_mm_scaler)
    
    # inverse transform
    X_tr_mm_inv = mm_scaler.inverse_transform(X_tr_mm_scaler[columns])
    X_tr_mm_inv = pd.DataFrame(X_tr_mm_inv, columns=columns)

    # add the county column to X_tr_mm_inv
    X_tr_mm_inv = pd.concat([X_tr_mm_inv, X_train[['county']].reset_index(drop=True)], axis=1)
    
    return X_tr_mm_scaler, X_v_mm_scaler, X_tr_mm_inv


# ----------------------------------------------------------------------------------
def get_std_train_scaler(X_train, X_validate):
    
    columns=['bedrooms', 'bathrooms','area','yearbuilt','taxamount','tax_rate']
    X_tr_std_scaler = X_train.copy()
    X_v_std_scaler = X_validate.copy()
    
    # Create a MinMaxScaler object
    std_scaler = StandardScaler()

    # Fit the scaler to the training data and transform it
    X_tr_std_scaler[columns] = std_scaler.fit_transform(X_train[columns])
    
    #using our scaler on validate
    X_v_std_scaler[columns] = std_scaler.transform(X_validate[columns])
    
    return X_tr_std_scaler, X_v_std_scaler


# ----------------------------------------------------------------------------------
def get_robust_train_scaler(X_train, X_validate):
     
    columns=['bedrooms', 'bathrooms','area','yearbuilt','taxamount','tax_rate']
    X_tr_rbs_scaler = X_train.copy()
    X_v_rbs_scaler = X_validate.copy()
    
    # Create a MinMaxScaler object
    rbs_scaler = RobustScaler()

    # Fit the scaler to the training data and transform it
    X_tr_rbs_scaler[columns] = rbs_scaler.fit_transform(X_train[columns])
    
    #using our scaler on validate
    X_v_rbs_scaler[columns] = rbs_scaler.transform(X_validate[columns])
    
    return X_tr_rbs_scaler, X_v_rbs_scaler


# ----------------------------------------------------------------------------------
def get_quant_normal(X_train):
    # Columns
    columns=['bedrooms', 'bathrooms','area','yearbuilt','taxamount','tax_rate']
    
    # Quantile transform with output distribution "normal"
    quant_norm = QuantileTransformer(output_distribution='normal')
    X_tr_quant_norm = pd.DataFrame(quant_norm.fit_transform(X_train[columns]),columns=columns)
    
    # Quantile trainsform by it self
    quant = QuantileTransformer()
    X_tr_quant = pd.DataFrame(quant.fit_transform(X_train[columns]),columns=columns)
    
    return quant_norm, X_tr_quant_norm, quant, X_tr_quant

# ----------------------------------------------------------------------------------
def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    object_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return object_cols

# ----------------------------------------------------------------------------------
def get_numeric_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # get a list of the column names that are objects (from the mask)
    num_cols = df.select_dtypes(exclude=['object', 'category']).columns.tolist()
    
    return num_cols


# ----------------------------------------------------------------------------------
def select_kbest(X,y,k):
    # X = X_train
    # y = y_train
    # k = the number of features to select (we are only sending two as of right now)
    
    # MAKE the thing
    kbest = SelectKBest(f_regression, k=k)

    # FIT the thing
    kbest.fit(X, y)
    
    # Create a DATAFRAME
    kbest_results = pd.DataFrame(
                dict(pvalues=kbest.pvalues_, feature_scores=kbest.scores_),
                index = X.columns)
    
    # we can apply this mask to the columns in our original dataframe
    top_k = X.columns[kbest.get_support()]
    
    return top_k
# ----------------------------------------------------------------------------------
def rfe(X,v,y,k):
    '''
    # X = X_train_scaled
    # v = X_validate_scaled
    # y = y_train
    # k = the number of features to select
    '''
    
    # make a model object to use in RFE process.
    # The model is here to give us metrics on feature importance and model score
    # allowing us to recursively reduce the number of features to reach our desired space
    model = LinearRegression()
    
    # MAKE the thing
    rfe = RFE(model, n_features_to_select=k)

    # FIT the thing
    rfe.fit(X, y)
    
    X_train_rfe = pd.DataFrame(rfe.transform(X),index=X.index,
                                          columns = X.columns[rfe.support_])
    
    X_val_rfe = pd.DataFrame(rfe.transform(v),index=v.index,
                                      columns = v.columns[rfe.support_])
    
    top_k_rfe = X.columns[rfe.get_support()]
    
    return top_k_rfe, X_train_rfe, X_val_rfe