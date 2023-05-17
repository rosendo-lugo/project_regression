import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# My imports
import wrangle as w
import explore as e
import modeling as m

# Graph imports
import matplotlib.pyplot as plt
import seaborn as sns

# Stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.linear_model import LassoLars
from sklearn.linear_model import TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, RFE, f_regression, SequentialFeatureSelector


# ----------------------------------------------------------------------------------
def get_Xs_ys_to_scale_baseline(tr, val, ts, target):
    '''
    tr = train
    val = validate
    ts = test
    target = target value
    '''
    # after getting the dummies drop the county column
    # tr, val, ts = tr.drop(columns =['county'])
    
    # Separate the features (X) and target variable (y) for the training set
    X_tr, y_tr = tr.drop(columns=[target,'county']), tr[target]
    
    # Separate the features (X) and target variable (y) for the validation set
    X_val, y_val = val.drop(columns=[target,'county']), val[target]
    
    # Separate the features (X) and target variable (y) for the test set
    X_ts, y_ts = ts.drop(columns=[target,'county']), ts[target]
    
    # Get the list of columns to be scaled
    to_scale = X_tr.columns.tolist()
    
    # Calculate the baseline (mean) of the target variable in the training set
    baseline = y_tr.mean()
    
    # Return the separated features and target variables, columns to scale, and baseline
    return X_tr, X_val, X_ts, y_tr, y_val, y_ts, to_scale, baseline
# ----------------------------------------------------------------------------------

def scale_data(X,Xv,Xts,to_scale):
    '''
    X = X_train
    Xv = X_validate
    Xts = X_test
    to_scale, is found in the get_Xs_ys_to_scale_baseline
    '''
    
    #make copies for scaling
    X_tr_sc = X.copy()
    X_val_sc = Xv.copy()
    X_ts_sc = Xts.copy()

    #scale them!
    #make the thing
    scaler = MinMaxScaler()

    #fit the thing
    scaler.fit(X[to_scale])

    #use the thing
    X_tr_sc[to_scale] = scaler.transform(X[to_scale])
    X_val_sc[to_scale] = scaler.transform(Xv[to_scale])
    X_ts_sc[to_scale] = scaler.transform(Xts[to_scale])
    
    return X_tr_sc, X_val_sc, X_ts_sc

# ----------------------------------------------------------------------------------
def metrics_reg(y, yhat):
    """
    y = y_train
    send in y_train, y_pred & returns RMSE, R2
    """
    rmse = mean_squared_error(y, yhat, squared=False)
    r2 = r2_score(y, yhat)
    return rmse, r2
# ----------------------------------------------------------------------------------
def get_minmax_train_scaler(X, Xv):
    '''
    X = X_train
    Xv = X_validate
    '''    
    
    # only using 4 columns
    columns=['bedrooms','bathrooms','area','yearbuilt']
    X_tr_mm_scaler = X.copy()
    X_v_mm_scaler = Xv.copy()
    
    # Create a MinMaxScaler object
    mm_scaler = MinMaxScaler()

    # Fit the scaler to the training data and transform it
    X_tr_mm_scaler[columns] = mm_scaler.fit_transform(X[columns])
    X_tr_mm_scaler = pd.DataFrame(X_tr_mm_scaler)
    
    #using our scaler on validate
    X_v_mm_scaler[columns] = mm_scaler.transform(Xv[columns])
    X_v_mm_scaler = pd.DataFrame(X_v_mm_scaler)
    
    # inverse transform
    X_tr_mm_inv = mm_scaler.inverse_transform(X_tr_mm_scaler[columns])
    X_tr_mm_inv = pd.DataFrame(X_tr_mm_inv, columns=columns)

    # add the county column to X_tr_mm_inv
    X_tr_mm_inv = pd.concat([X_tr_mm_inv, X[['county']].reset_index(drop=True)], axis=1)
    
    return X_tr_mm_scaler, X_v_mm_scaler, X_tr_mm_inv


# ----------------------------------------------------------------------------------
def get_std_train_scaler(X, Xv):
    '''
    X = X_train
    Xv = X_validate
    '''    
    
    columns=['bedrooms', 'bathrooms','area','yearbuilt']
    X_tr_std_scaler = X.copy()
    X_v_std_scaler = Xv.copy()
    
    # Create a MinMaxScaler object
    std_scaler = StandardScaler()

    # Fit the scaler to the training data and transform it
    X_tr_std_scaler[columns] = std_scaler.fit_transform(X[columns])
    
    #using our scaler on validate
    X_v_std_scaler[columns] = std_scaler.transform(Xv[columns])
    
    return X_tr_std_scaler, X_v_std_scaler


# ----------------------------------------------------------------------------------
def get_robust_train_scaler(X, Xv):
    '''
    X = X_train
    Xv = X_validate
    '''

    columns = ['bedrooms', 'bathrooms', 'area', 'yearbuilt']
    X_tr_rbs_scaler = X.copy()
    X_v_rbs_scaler = Xv.copy()

    # Create a MinMaxScaler object
    rbs_scaler = RobustScaler()

    # Fit the scaler to the training data and transform it
    X_tr_rbs_scaler[columns] = rbs_scaler.fit_transform(X[columns])

    # Using the scaler on the validate data
    X_v_rbs_scaler[columns] = rbs_scaler.transform(Xv[columns])

    return X_tr_rbs_scaler, X_v_rbs_scaler


# ----------------------------------------------------------------------------------
def get_quant_normal(X):
    '''
    X = X_train
    '''
    # Columns
    columns=['bedrooms', 'bathrooms','area','yearbuilt']
    
    # Quantile transform with output distribution "normal"
    quant_norm = QuantileTransformer(output_distribution='normal')
    X_tr_quant_norm = pd.DataFrame(quant_norm.fit_transform(X[columns]),columns=columns)
    
    # Quantile trainsform by it self
    quant = QuantileTransformer()
    X_tr_quant = pd.DataFrame(quant.fit_transform(X[columns]),columns=columns)
    
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
    '''
    # X = X_train
    # y = y_train
    # k = the number of features to select (we are only sending two as of right now)
    '''
    
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
    
    X_tr_rfe = pd.DataFrame(rfe.transform(X),index=X.index,
                                          columns = X.columns[rfe.support_])
    
    X_val_rfe = pd.DataFrame(rfe.transform(v),index=v.index,
                                      columns = v.columns[rfe.support_])
    
    top_k_rfe = X.columns[rfe.get_support()]
    
    return top_k_rfe, X_tr_rfe, X_val_rfe

# ----------------------------------------------------------------------------------
def get_models_dataframe(baseline, tr, y_tr, y_val, y_ts, X_tr_sc, X_val_sc, X_ts_sc):
    baseline_array = np.repeat(baseline, len(tr))
    rmse, r2 = m.metrics_reg(y_tr, baseline_array)
    metrics_df = pd.DataFrame(data=[{'model': 'baseline', 'rmse val': rmse, 'r2 val': r2}])

    # OLS + RFE
    top_k_rfe, X_tr_rfe, X_val_rfe = m.rfe(X_tr_sc, X_val_sc, y_tr, 3)
    lr_rfe = LinearRegression()
    lr_rfe.fit(X_tr_rfe, y_tr)
    pred_lr_rfe = lr_rfe.predict(X_val_rfe)
    rmse, r2 = m.metrics_reg(y_val, pred_lr_rfe)
    metrics_df.loc[1] = ['ols+RFE', rmse, r2]

    # OLS
    lr = LinearRegression()
    lr.fit(X_tr_sc, y_tr)
    pred_lr = lr.predict(X_val_sc)
    rmse, r2 = m.metrics_reg(y_val, pred_lr)
    metrics_df.loc[2] = ['ols', rmse, r2]

    # LARS
    lars = LassoLars(alpha=1)
    lars.fit(X_tr_sc, y_tr)
    pred_lars = lars.predict(X_val_sc)
    rmse, r2 = m.metrics_reg(y_val, pred_lars)
    metrics_df.loc[3] = ['lars', rmse, r2]
    
    # Polynomial
    degrees = range(2, 9)
    for degree in degrees:
        pf = PolynomialFeatures(degree=degree)
        X_tr_degree = pf.fit_transform(X_tr_sc)
        X_val_degree = pf.transform(X_val_sc)
        X_ts_degree = pf.transform(X_ts_sc)

        pr = LinearRegression()
        pr.fit(X_tr_degree, y_tr)
        pred_pr = pr.predict(X_val_degree)
        rmse, r2 = m.metrics_reg(y_val, pred_pr)
        metrics_df.loc[degree + 2] = [f'poly_{degree}D', rmse, r2]

    # GLM
    glm = TweedieRegressor(power=1, alpha=0)
    glm.fit(X_tr_sc, y_tr)
    pred_glm = glm.predict(X_val_sc)
    rmse, r2 = m.metrics_reg(y_val, pred_glm)
    metrics_df.loc[11] = ['glm', rmse, r2]

    return metrics_df, pred_lr_rfe, pred_lr, pred_lars, pred_pr, pred_glm

# ----------------------------------------------------------------------------------
def get_best_model(baseline, tr, y_tr, y_val, y_ts, X_tr_sc, X_val_sc, X_ts_sc):
    baseline_array = np.repeat(baseline, len(tr))
    rmse, r2 = m.metrics_reg(y_tr, baseline_array)
    best_model = {'best model': None, 'rmse': np.inf, 'r2': -np.inf}

    # OLS + RFE
    top_k_rfe, X_tr_rfe, X_val_rfe = m.rfe(X_tr_sc, X_val_sc, y_tr, 3)
    lr_rfe = LinearRegression()
    lr_rfe.fit(X_tr_rfe, y_tr)
    pred_lr_rfe = lr_rfe.predict(X_val_rfe)
    rmse, r2 = m.metrics_reg(y_val, pred_lr_rfe)
    if rmse < best_model['rmse'] and r2 > best_model['r2']:
        best_model = {'model': 'ols+RFE', 'rmse': rmse, 'r2': r2}

    # OLS
    lr = LinearRegression()
    lr.fit(X_tr_sc, y_tr)
    pred_lr = lr.predict(X_val_sc)
    rmse, r2 = m.metrics_reg(y_val, pred_lr)
    if rmse < best_model['rmse'] and r2 > best_model['r2']:
        best_model = {'best model': 'ols', 'rmse': rmse, 'r2': r2}

    # LARS
    lars = LassoLars(alpha=1)
    lars.fit(X_tr_sc, y_tr)
    pred_lars = lars.predict(X_val_sc)
    rmse, r2 = m.metrics_reg(y_val, pred_lars)
    if rmse < best_model['rmse'] and r2 > best_model['r2']:
        best_model = {'best model': 'lars', 'rmse': rmse, 'r2': r2}
    
    # Polynomial
    degrees = range(2, 9)
    for degree in degrees:
        pf = PolynomialFeatures(degree=degree)
        X_tr_degree = pf.fit_transform(X_tr_sc)
        X_val_degree = pf.transform(X_val_sc)
        X_ts_degree = pf.transform(X_ts_sc)

        pr = LinearRegression()
        pr.fit(X_tr_degree, y_tr)
        pred_pr = pr.predict(X_val_degree)
        rmse, r2 = m.metrics_reg(y_val, pred_pr)
        if rmse < best_model['rmse'] and r2 > best_model['r2']:
            best_model = {'best model': f'poly_{degree}D', 'rmse': rmse, 'r2': r2}

    # GLM
    glm = TweedieRegressor(power=1, alpha=0)
    glm.fit(X_tr_sc, y_tr)
    pred_glm = glm.predict(X_val_sc)
    rmse, r2 = m.metrics_reg(y_val, pred_glm)
    if rmse < best_model['rmse'] and r2 > best_model['r2']:
        best_model = {'best model': 'glm', 'rmse': rmse, 'r2': r2}
    
    best_model = pd.DataFrame([best_model])
    return best_model

# ----------------------------------------------------------------------------------
def test_best_model(best_model, X_ts_sc, y_ts, X_tr_sc, y_tr):
    best_model_values = best_model.iloc[0]  # Extract the first row of the DataFrame
    model_name = best_model_values['best model']  # Extract the model name

    if model_name == 'ols+RFE':
        top_k_rfe, X_tr_rfe, X_ts_rfe = m.rfe(X_tr_sc, X_ts_sc, y_tr, 3)
        lr_rfe = LinearRegression()
        lr_rfe.fit(X_tr_rfe, y_tr)
        y_pred = lr_rfe.predict(X_ts_rfe)
    elif model_name == 'ols':
        lr = LinearRegression()
        lr.fit(X_tr_sc, y_tr)
        y_pred = lr.predict(X_ts_sc)
    elif model_name == 'lars':
        lars = LassoLars(alpha=1)
        lars.fit(X_tr_sc, y_tr)
        y_pred = lars.predict(X_ts_sc)
    elif model_name.startswith('poly_'):
        degree = int(model_name.split('_')[1].strip('D'))
        pf = PolynomialFeatures(degree=degree)
        X_tr_poly = pf.fit_transform(X_tr_sc)
        X_ts_poly = pf.transform(X_ts_sc)
        pr = LinearRegression()
        pr.fit(X_tr_poly, y_tr)
        y_pred = pr.predict(X_ts_poly)
    elif model_name == 'glm':
        glm = TweedieRegressor(power=1, alpha=0)
        glm.fit(X_tr_sc, y_tr)
        y_pred = glm.predict(X_ts_sc)
    else:
        raise ValueError('Invalid model name.')

    rmse, r2 = m.metrics_reg(y_ts, y_pred)
    result = {'model': 'test', 'rmse': rmse, 'r2': r2}
    result = pd.DataFrame([result])
    return result


















































# def get_models_dataframe(baseline,tr,y_tr,y_val,y_ts,X_tr_sc,X_val_sc,X_ts_sc):
#     #make an array to send into my mean_square_error function
#     baseline_array = np.repeat(baseline, len(tr))
#     rmse, r2 = m.metrics_reg(y_tr, baseline_array)
#     metrics_df = pd.DataFrame(data=[
#         {
#             'model':'baseline',
#             'rmse':rmse,
#             'r2':r2
#         }

#     ])

#     # OLS + RFE
#     # USE THIS ONE to find the best features for RFE
#     top_k_rfe, X_tr_rfe, X_val_rfe = m.rfe(X_tr_sc,X_val_sc, y_tr,3)
#     #intial ML model
#     lr = LinearRegression()

#     #fit the thing
#     lr.fit(X_tr_rfe, y_tr)

#     #use the thing (make predictions)
#     pred_lr_rfe = lr.predict(X_tr_rfe)
#     pred_val_lr_rfe = lr.predict(X_val_rfe)
#     #train
#     m.metrics_reg(y_tr, pred_lr_rfe)
#     #validate
#     rmse, r2 = m.metrics_reg(y_val, pred_val_lr_rfe)
#     #add to my metrics df
#     metrics_df.loc[1] = ['ols+RFE', rmse, r2]

#     # OLS
#     #make it
#     lr = LinearRegression()

#     #fit it on our RFE features
#     lr.fit(X_tr_sc, y_tr)

#     #use it (make predictions)
#     pred_lr = lr.predict(X_tr_sc)

#     #use it on validate
#     pred_val_lr = lr.predict(X_val_sc)
#     #train 
#     m.metrics_reg(y_tr, pred_lr)
#     #validate
#     rmse, r2 = m.metrics_reg(y_val, pred_val_lr)
#     #add to my metrics df
#     metrics_df.loc[2] = ['ols', rmse, r2]


#     # LARS
#     #make it
#     lars = LassoLars(alpha=1)

#     #fit it
#     lars.fit(X_tr_sc, y_tr)

#     #use it
#     pred_lars = lars.predict(X_tr_sc)
#     pred_val_lars = lars.predict(X_val_sc)

#     #train
#     m.metrics_reg(y_tr, pred_lars)

#     #validate
#     rmse, r2 = m.metrics_reg(y_val, pred_val_lars)

#     #add to my metrics df
#     metrics_df.loc[3] = ['lars', rmse, r2]

    
#     # POLYNOMIAL
#     # make the polynomial features to get a new set of features
#     pf = PolynomialFeatures(degree=4)

#     # fit and transform X_train_scaled
#     X_tr_degree2 = pf.fit_transform(X_tr_sc)

#     # transform X_validate_scaled & X_test_scaled
#     X_val_degree2 = pf.transform(X_val_sc)
#     X_ts_degree2 = pf.transform(X_ts_sc)

#     #make it
#     pr = LinearRegression()

#     #fit it
#     pr.fit(X_tr_degree2, y_tr)

#     #use it
#     pred_pr = pr.predict(X_tr_degree2)
#     pred_val_pr = pr.predict(X_val_degree2)

#     #train
#     m.metrics_reg(y_tr, pred_pr)

#     #validate
#     rmse, r2 = m.metrics_reg(y_val, pred_val_pr)

#     #add to my metrics df
#     metrics_df.loc[4] = ['poly_2D', rmse, r2]


#     # GLM
#     #make it
#     glm = TweedieRegressor(power=1, alpha=0)

#     #fit it
#     glm.fit(X_tr_sc, y_tr)

#     #use it
#     pred_glm = glm.predict(X_tr_sc)
#     pred_val_glm = glm.predict(X_val_sc)

#     #train
#     m.metrics_reg(y_tr, pred_glm)

#     #validate
#     rmse, r2 = m.metrics_reg(y_val, pred_val_glm)

#     #add to my metrics df
#     metrics_df.loc[5] = ['glm',rmse,r2]


#     # BEST TEST
#     #use it
#     pred_test = pr.predict(X_ts_degree2)

#     #train
#     rmse, r2 = m.metrics_reg(y_ts, pred_test)

#     #add to my metrics df
#     metrics_df.loc[6] = ['test',rmse,r2]
#     return metrics_df


# # ----------------------------------------------------------------------------------
# def get_models_dataframe_copy(baseline,tr,y_tr,y_val,y_ts,X_tr_sc,X_val_sc,X_ts_sc):
#     #make an array to send into my mean_square_error function
#     baseline_array = np.repeat(baseline, len(tr))
#     rmse, r2 = m.metrics_reg(y_tr, baseline_array)
#     metrics_df = pd.DataFrame(data=[
#         {
#             'model':'baseline',
#             'rmse':rmse,
#             'r2':r2
#         }

#     ])

#     # OLS + RFE
#     # USE THIS ONE to find the best features for RFE
#     top_k_rfe, X_tr_rfe, X_val_rfe = m.rfe(X_tr_sc,X_val_sc, y_tr,3)
#     #intial ML model
#     lr = LinearRegression()

#     #fit the thing
#     lr.fit(X_tr_rfe, y_tr)

#     #use the thing (make predictions)
#     pred_lr_rfe = lr.predict(X_tr_rfe)
#     pred_val_lr_rfe = lr.predict(X_val_rfe)
#     #train
#     m.metrics_reg(y_tr, pred_lr_rfe)
#     #validate
#     rmse, r2 = m.metrics_reg(y_val, pred_val_lr_rfe)
#     #add to my metrics df
#     metrics_df.loc[1] = ['ols+RFE', rmse, r2]

#     # OLS
#     #make it
#     lr = LinearRegression()

#     #fit it on our RFE features
#     lr.fit(X_tr_sc, y_tr)

#     #use it (make predictions)
#     pred_lr = lr.predict(X_tr_sc)

#     #use it on validate
#     pred_val_lr = lr.predict(X_val_sc)
#     #train 
#     m.metrics_reg(y_tr, pred_lr)
#     #validate
#     rmse, r2 = m.metrics_reg(y_val, pred_val_lr)
#     #add to my metrics df
#     metrics_df.loc[2] = ['ols', rmse, r2]


#     # LARS
#     #make it
#     lars = LassoLars(alpha=1)

#     #fit it
#     lars.fit(X_tr_sc, y_tr)

#     #use it
#     pred_lars = lars.predict(X_tr_sc)
#     pred_val_lars = lars.predict(X_val_sc)

#     #train
#     m.metrics_reg(y_tr, pred_lars)

#     #validate
#     rmse, r2 = m.metrics_reg(y_val, pred_val_lars)

#     #add to my metrics df
#     metrics_df.loc[3] = ['lars', rmse, r2]
    
#     # GLM
#     #make it
#     glm = TweedieRegressor(power=1, alpha=0)

#     #fit it
#     glm.fit(X_tr_sc, y_tr)

#     #use it
#     pred_glm = glm.predict(X_tr_sc)
#     pred_val_glm = glm.predict(X_val_sc)

#     #train
#     m.metrics_reg(y_tr, pred_glm)

#     #validate
#     rmse, r2 = m.metrics_reg(y_val, pred_val_glm)

#     #add to my metrics df
#     metrics_df.loc[4] = ['glm',rmse,r2]

#     degrees = range(2, 9)  # Range of degrees: 2 to 8

#     for degree in degrees:
#         # POLYNOMIAL
#         # Make the polynomial features to get a new set of features
#         pf = PolynomialFeatures(degree=degree)

#         # Fit and transform X_train_scaled
#         X_tr_degree = pf.fit_transform(X_tr_sc)

#         # Transform X_validate_scaled & X_test_scaled
#         X_val_degree = pf.transform(X_val_sc)
#         X_ts_degree = pf.transform(X_ts_sc)

#         # Make it
#         pr = LinearRegression()

#         # Fit it
#         pr.fit(X_tr_degree, y_tr)

#         # Use it
#         pred_pr = pr.predict(X_tr_degree)
#         pred_val_pr = pr.predict(X_val_degree)

#         # Train
#         m.metrics_reg(y_tr, pred_pr)

#         # Validate
#         rmse, r2 = m.metrics_reg(y_val, pred_val_pr)

#         # Add to the metrics_df
#         metrics_df.loc[degree + 2] = [f'poly_{degree}D', rmse, r2]  # Adjusted index


#     # BEST TEST
#     #use it
#     pred_test = pr.predict(X_ts_degree)

#     #train
#     rmse, r2 = m.metrics_reg(y_ts, pred_test)

#     #add to my metrics df
#     metrics_df.loc[11] = ['test',rmse,r2]
#     return metrics_df
# # ----------------------------------------------------------------------------------

# def get_models_dataframe_copycopy(baseline, tr, y_tr, y_val, y_ts, X_tr_sc, X_val_sc, X_ts_sc):
#     # Initialize metrics_df
#     metrics_df = pd.DataFrame(columns=['model', 'rmse', 'r2'])

#     # Baseline model
#     baseline_array = np.repeat(baseline, len(tr))
#     rmse, r2 = m.metrics_reg(y_tr, baseline_array)
#     metrics_df.loc[0] = ['baseline', rmse, r2]

#     # Model list with their respective objects
#     models = [
#         ('ols+RFE', LinearRegression(), m.rfe(X_tr_sc, X_val_sc, y_tr, 3)),
#         ('ols', LinearRegression(), (X_tr_sc, X_val_sc)),
#         ('lars', LassoLars(alpha=1), (X_tr_sc, X_val_sc)),
#         ('glm', TweedieRegressor(power=1, alpha=0), (X_tr_sc, X_val_sc))
#     ]

#     for degree in range(2, 9):
#         models.append((f'poly_{degree}D', LinearRegression(), (PolynomialFeatures(degree=degree).fit_transform(X_tr_sc), PolynomialFeatures(degree=degree).transform(X_val_sc))))

#     # Fit and evaluate models
#     for i, (model_name, model, (X_tr_model, X_val_model)) in enumerate(models):
#         model.fit(X_tr_model, y_tr)
#         pred_tr = model.predict(X_tr_model)
#         pred_val = model.predict(X_val_model)
#         rmse_tr, r2_tr = m.metrics_reg(y_tr, pred_tr)
#         rmse_val, r2_val = m.metrics_reg(y_val, pred_val)
#         metrics_df.loc[i + 1] = [model_name, rmse_val, r2_val]

#     # Find the best model
#     best_model_index = metrics_df['r2'].idxmax()

#     # Test the best model
#     best_model_name = metrics_df.loc[best_model_index, 'model']
#     best_model = models[best_model_index - 1][1]
#     X_ts_best_model = models[best_model_index - 1][2][1]
#     pred_ts_best = best_model.predict(X_ts_best_model)
#     rmse_ts_best, r2_ts_best = m.metrics_reg(y_ts, pred_ts_best)
#     metrics_df.loc[len(models) + 1] = ['test', rmse_ts_best, r2_ts_best]

#     return metrics_df


# # ----------------------------------------------------------------------------------
# def get_models_dataframe_copycopycopy(baseline, tr, y_tr, y_val, y_ts, X_tr_sc, X_val_sc, X_ts_sc):
#     # Initialize metrics_df
#     metrics_df = pd.DataFrame(columns=['model', 'rmse', 'r2'])

#     # Baseline model
#     baseline_array = np.repeat(baseline, len(tr))
#     rmse, r2 = m.metrics_reg(y_tr, baseline_array)
#     metrics_df.loc[0] = ['baseline', rmse, r2]

#     # Model list with their respective objects
#     models = [
#         ('ols+RFE', LinearRegression(), m.rfe(X_tr_sc, X_val_sc, y_tr, 3)),
#         ('ols', LinearRegression(), (X_tr_sc, X_val_sc)),
#         ('lars', LassoLars(alpha=1), (X_tr_sc, X_val_sc)),
#         ('glm', TweedieRegressor(power=1, alpha=0), (X_tr_sc, X_val_sc))
#     ]

#     for degree in range(2, 9):
#         pf = PolynomialFeatures(degree=degree)
#         X_tr_degree = pf.fit_transform(X_tr_sc)
#         X_val_degree = pf.transform(X_val_sc)
#         X_ts_degree = pf.transform(X_ts_sc)
#         models.append((f'poly_{degree}D', LinearRegression(), (X_tr_degree, X_val_degree, X_ts_degree)))

#     # Fit and evaluate models
#     for i, (model_name, model, (X_tr_model, X_val_model, X_ts_model)) in enumerate(models):
#         model.fit(X_tr_model, y_tr)
#         pred_tr = model.predict(X_tr_model)
#         pred_val = model.predict(X_val_model)
#         rmse_tr, r2_tr = m.metrics_reg(y_tr, pred_tr)
#         rmse_val, r2_val = m.metrics_reg(y_val, pred_val)
#         metrics_df.loc[i + 1] = [model_name, rmse_val, r2_val]

#     # Find the best model
#     best_model_index = metrics_df['r2'].idxmax()

#     # Test the best model
#     best_model_name = metrics_df.loc[best_model_index, 'model']
#     best_model = models[best_model_index - 1][1]
#     X_ts_best_model = models[best_model_index - 1][2][2]
#     pred_ts_best = best_model.predict(X_ts_best_model)
#     rmse_ts_best, r2_ts_best = m.metrics_reg(y_ts, pred_ts_best)
#     metrics_df.loc[len(models) + 1] = ['test', rmse_ts_best, r2_ts_best]

#     return metrics_df

# # ----------------------------------------------------------------------------------

# def get_models_dataframe2(y_tr, y_val, y_ts, X_tr_sc, X_val_sc, X_ts_sc):
#     metrics_df = pd.DataFrame(columns=['model', 'rmse', 'r2'])

#     models = {
#         'ols+RFE': (LinearRegression(), m.rfe(X_tr_sc, X_val_sc, y_tr, 3)),
#         'ols': (LinearRegression(), (X_tr_sc, X_val_sc)),
#         'lars': (LassoLars(alpha=1), (X_tr_sc, X_val_sc)),
#         'poly_2D': (PolynomialFeatures(degree=2), (X_tr_sc, X_val_sc, X_ts_sc)),
#         'glm': (TweedieRegressor(power=1, alpha=0), (X_tr_sc, X_val_sc))
#     }

#     for i, (model_name, (model, feature_set)) in enumerate(models.items()):
#         if model_name == 'poly_2D':
#             pf = feature_set[0]
#             X_tr_model = pf.fit_transform(feature_set[1])
#             X_val_model = pf.transform(feature_set[2])
#         else:
#             X_tr_model, X_val_model = feature_set

#         model.fit(X_tr_model, y_tr)
#         pred_tr = model.predict(X_tr_model)
#         pred_val = model.predict(X_val_model)
#         rmse_tr, r2_tr = m.metrics_reg(y_tr, pred_tr)
#         rmse_val, r2_val = m.metrics_reg(y_val, pred_val)
#         metrics_df.loc[i] = [model_name, rmse_val, r2_val]

#     best_model_index = metrics_df['r2'].idxmax()
#     best_model_name = metrics_df.loc[best_model_index, 'model']
#     best_model = models[best_model_name][0]
#     X_ts_model = models[best_model_name][1][2]
#     if best_model_name == 'poly_2D':
#         X_ts_model = pf.transform(X_ts_model)
#     pred_ts_best = best_model.predict(X_ts_model)
#     rmse_ts_best, r2_ts_best = m.metrics_reg(y_ts, pred_ts_best)
#     metrics_df.loc[len(models)] = ['test', rmse_ts_best, r2_ts_best]

#     return metrics_df
