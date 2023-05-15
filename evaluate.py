from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import seaborn as sns
import matplotlib.pyplot as plt



# ----------------------------------------------------------------------------------
def plot_residuals(y, yhat):
    '''
    y=train
    yhat=yhat
    '''
    # residual = actual - predicted
    residuals = y - yhat
    
    # plot a scatterplot graph
    sns.scatterplot(x=y, y=residuals)
    plt.xlabel('Property Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.axhline(y=0, color='r', linestyle='-')
    
    return plt.show()
# ----------------------------------------------------------------------------------
def regression_errors(y, yhat): 
    '''
    y=target variable(column)
    yhat=yhat
    '''
    ## Sum of Squared Errors
    SSE = MSE * len(train)

    # Total sum of squares
    # TSS = ESS + SSE
    TSS = ESS + SSE

    # Explained sum of squares
    # Sum the squares of the (prediction - mean of tax value)
    ESS = ((yhat - y.mean())**2).sum()

    # Mean squared error
    MSE = mean_squared_error(y, yhat)

    # Root mean squared error
    # Raising to the power of 1/2 (0.5) is the same as taking the square root
    RMSE = MSE ** .5
    
    return SSE, ESS, TSS, MSE, RMSE

# ----------------------------------------------------------------------------------
def baseline_mean_errors(y):
    '''
    y=train
    '''
    # Use the MSE to find the SSE
    SSE_baseline = MSE_baseline * len(train)

    # Compute the mean squared error for the baseline
    MSE_baseline = mean_squared_error(train.tax_value, train.baseline)

    #calculate RMSE
    RMSE_baseline = MSE_baseline ** .5
    
    return SSE_baseline, MSE_baseline, RMSE_baseline

# ----------------------------------------------------------------------------------
def better_than_baseline(y, yhat):
    '''
    y=train
    yhat=yhat
    '''
    # Calculate baseline residuals and SSE
    baseline_residual = y - y.mean()
    SSE_baseline = (baseline_residual ** 2).sum()

    # Calculate SSE for the given model
    SSE = ((y - yhat) ** 2).sum()
    
    # Check if the model has lower SSE than baseline
    if SSE < SSE_baseline:
        print("The model performs better than the baseline model.")
    else:
        print("The model does not perform better than the baseline model.")

# ----------------------------------------------------------------------------------
def better_than_baseline(y, yhat):
    '''
    y=train
    yhat=yhat
    '''
    # Calculate the r2 using the r2_score
    r2_score(train.property_value, yhat)

    # Check if the model has R square higher than .5
    if R_2 > .5:
        print("The model performs better than as it's closer to 1.")
    else:
        print("The model does not perform better as it's too far away from 1.")