# project_regression
> My Regression project

## Project Description
> To predict property tax assessed values and make recommendation for better models.

## Project Goal
> To predict the property tax assessed values ('taxvaluedollarcnt') of Single Family Properties that had a transaction during 2017> 
> Recommend on a ways to make a better model> 
> Find what states and counties are the property located in.



## THE Big Pipeline Plan
Acquire
>- get the data into pandas
>- look at it
    >>- describe, info, head, shape
>- understand what your data means
    >>- know what each column is
  >>- know what your target variable is
    
Wrangle
>- clean the data
   >>- handle nulls
   >>- handle outliers
   >>- correct datatypes
>- univariate analysis (looking at only one variable)
>- encode variables -- Preprocessing
>- split into train, validate/, test
>- scale data (after train/validate/test split) -- Preprocessing
>- document how your changing the data

Explore
>- use only train data!
   >>- use unscaled data
>- establish relationships using multivariate analysis
   >>- hypothesize
   >>- visualize
   >>- statistize
   >>- summarize
>- feature engineering
   >>- when using RFE, use scaled data

Model
>- use scaled/encoded data
>- split into X_variables and y_variables
  >>- X_train, y_train, X_validate, y_validate, X_test, y_test
>- build models
  >>- make the thing
  >>- fit the thing (on train)
  >>- use the thing
>- evaluate models on train and validate
>- pick the best model and evaluate it on test

Models used
>- Ordinary Least Squares + RFE
>- LASSO + LARS
>- Polynomial Regression
>- Generalized Linear Model
    

## Data Dictionary

| Features | Definition |
| --- | --- |
| bedrooms | Number of bedrooms |
| bathrooms | Number of bathrooms |
| area | Area of the house in square footage |
| yearbuilt | The year the property was built |
| county | The county were the properies are located |
| county_Orange | Orange county - where the property is located |
| county_Ventura | Ventura county - where the property is located |
| property_value | property tax assessed value in dollars |



## Instructions on how someone else can reproduce this project and findings (What would someone need to be able to recreate your project on their own?)
> 1. Clone this entire repository.
> 2. Acquire the Zillow dataset from MySQL. To have access to this data in MySQL request user and password from Codeup instructors.
> 3. Run project_regression.ipynb to extract zillow.csv file.

## Recommendations
> Based on the findings, removing as many outliers as possible is recommended, as testing several models and using algorithms to show the best models.

## Next Steps
> The next steps include comparing the train RMSE vs the validate RMSE and further exploring other features. .