import pandas as pd
import numpy as np

# Viz imports
import matplotlib.pyplot as plt
import seaborn as sns

# stats
from scipy import stats

# ----------------------------------------------------------------------------------
def plot_variable_pairs(tr_smpl):
    '''
    tr_smpl = train sample of only 10,000
    target = target variable
    '''
    # We can drop the redundant information in the upper right half of the chart if we like.
    g = sns.pairplot(data=tr_smpl, corner=True, kind='reg')
    g.map_lower(sns.regplot, line_kws={'color': 'orange'})
    plt.show()

    
# ----------------------------Question one------------------------------------------
def get_qone_chart(tr_smpl):
    '''
    tr_smpl (train sample of only 10,000)
    '''
    # Create a boxplot for the current column
    sns.boxplot(data=tr_smpl, x='county', y='property_value', order=['LA','Orange','Ventura'])
    # Set the title for the plot
    plt.title(f"County vs. Property Value")
    # Display the plot
    plt.show()
    
    # Bar chart
    avg_value_by_county = tr_smpl.groupby('county')['property_value'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=avg_value_by_county, x='county', y='property_value')
    plt.title('Average Assessed Property Values by County')
    plt.xlabel('County')
    plt.ylabel('Average Property Value')
    plt.xticks(rotation=0)
    plt.show()
# ----------------------------------------------------------------------------------
def get_anova_n_kruskal_test(tr_smpl):
    '''
    tr_smpl (train sample of only 10,000)
    '''
    county = tr_smpl['county']
    property_value = tr_smpl['property_value']
    
    # significance level
    alpha = .05

#     # Perform one-way ANOVA
#     fvalue, pvalue = stats.f_oneway(*[property_value[county == c] for c in county.unique()])

#     # Print the results
#     print("ANOVA - p-value:", pvalue)

#     if pvalue < alpha:
#         print('We reject the null hypothesis\n')
#     else:
#         print('We fail to reject the null hypothesis\n')


    # Perform Kruskal-Wallis test
    hvalue, pv = stats.kruskal(*[property_value[county == c] for c in county.unique()])

    # Print the results
    print("Kruskal-Wallis - p-value:", pv)

    if pv < alpha:
        print('We reject the null hypothesis')
    else:
        print('We fail to reject the null hypothesis')

# ----------------------------Question two------------------------------------------
def get_qtwo_chart(tr_smpl):
    '''
    tr_smpl (train sample of only 10,000)
    '''
    # lmplot chart
    sns.lmplot(data=tr_smpl, x='area', y='property_value', markers='.', line_kws={'color':'red'})
    plt.show()
    
    
    # Define the bin edges
    bin_edges = np.arange(0, tr_smpl['area'].max() + 1000, 1000)
    # Create the bins based on the bin edges
    bin_labels = pd.cut(tr_smpl['area'], bins=bin_edges)
    # Group the data by the bins and calculate the mean property value
    grouped_data = tr_smpl.groupby(bin_labels)['property_value'].mean()

    # Bar chart
    plt.figure(figsize=(10, 6))
    sns.barplot(x=grouped_data.index, y=grouped_data.values)
    plt.title('Assessed Property Values by Area')
    plt.xlabel('Area (sqft)')
    plt.ylabel('Property Value')
    plt.xticks(rotation=0)
    plt.show()


# ----------------------------------------------------------------------------------
def get_qtwo_stats(tr_smpl):
    # Significance level
    alpha = 0.05
    
    # Perform Pearson correlation test
    r_value, p_value = stats.pearsonr(tr_smpl['area'], tr_smpl['property_value'])
    
    # Print the results
    print("Pearson correlation - p-value:", p_value)

    if p_value < alpha:
        print('We reject the null hypothesis\n')
    else:
        print('We fail to reject the null hypothesis\n')

    print('Correlation between area and property value')
    print(f'  r = {r_value:.4f}')
    
# def perform_pearsonr_test(tr_smpl):
#     # Significance level
#     alpha = 0.05
    
#     # Perform Pearson correlation test
#     r_value, p_value = stats.pearsonr(tr_smpl['area'], tr_smpl['property_value'])
    
#     # Print the results
#     print("Pearson correlation - p-value: {:.30f}".format(p_value))

#     if p_value < alpha:
#         print('We reject the null hypothesis\n')
#     else:
#         print('We fail to reject the null hypothesis\n')

#     print('Correlation between area and property value')
#     print(f'  r = {r_value:.4f}')
# ----------------------------Question three----------------------------------------
def get_qthree_chart(tr_smpl):
    # Create a scatter plot
    plt.figure(figsize=(9, 6))

    # Scatter plot with x-axis as the number of bedrooms, y-axis as the number of bathrooms,
    # and color of markers representing the property value
    plt.scatter(tr_smpl['bedrooms'], tr_smpl['bathrooms'], c=tr_smpl['property_value'], cmap='cool', s=80)

    # Add a color bar to show the mapping of colors to property values
    plt.colorbar(label='Property Value')

    # Set the title and axis labels
    plt.title('Relationship between Bedrooms, Bathrooms, and Property Value')
    plt.xlabel('Number of Bedrooms')
    plt.ylabel('Number of Bathrooms')

    # Display the scatter plot
    plt.show()


# # Scatter plot
# plt.figure(figsize=(10, 6))
# plt.scatter(tr_smpl['bedrooms'], tr_smpl['property_value'], label='Bedrooms', color='blue')
# plt.scatter(tr_smpl['bathrooms'], tr_smpl['property_value'], label='Bathrooms', color='red')
# plt.title('Relationship between Bedrooms, Bathrooms, and Property Value')
# plt.xlabel('Number of Bedrooms / Bathrooms')
# plt.ylabel('Property Value')
# plt.legend()
# plt.show()

# ----------------------------------------------------------------------------------
def get_qthree_stats(tr_smpl):
    # Create the correlation matrix for the bedrooms, bathrooms and property values.
    exam_corr = tr_smpl.drop(columns=['area','yearbuilt','county_Orange','county_Ventura']).corr()
    

    # Perform one-way ANOVA
    fvalue, pvalue = stats.f_oneway(*[tr_smpl.property_value[tr_smpl.bedrooms == c] for c in tr_smpl.bedrooms.unique()],
                                    *[tr_smpl.property_value[tr_smpl.bathrooms == c] for c in tr_smpl.bathrooms.unique()])
    alpha = .05
    # Print the results
    print("\nANOVA - p-value:", pvalue)

    if pvalue < alpha:
        print('We reject the null hypothesis\n')
    else:
        print('We fail to reject the null hypothesis\n')
        
    return exam_corr

# ----------------------------Question four-----------------------------------------
def get_qfour_chart(tr_smpl):
    # Linear plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=tr_smpl, x='yearbuilt', y='property_value')
    plt.title('Assessed Property Values by Year Built')
    plt.xlabel('Year Built')
    plt.ylabel('Property Value')
    plt.xticks(rotation=0)
    plt.show()

# ----------------------------------------------------------------------------------
def get_qfour_stats(tr_smpl):
    # t-test 
    t, p = stats.ttest_ind(tr_smpl.yearbuilt, tr_smpl.property_value)
    t, p

    # significan level
    alpha = .05

    # Print the results
    print("\nt-test - p-value:", p)

    if p < alpha:
        print('We reject the null hypothesis\n')
    else:
        print('We fail to reject the null hypothesis\n')
    
    
    
# ----------------------------------------------------------------------------------
# def plot_categorical_and_continuous_vars(tr_smpl, cat_var, cont_var, num_bins=50):
#     for cat_col in cat_var:
#         for cont_col in cont_var:
#             # Subset the dataframe based on the loop variables
#             temp_df = tr_smpl[[cat_col, cont_col]]

#             # Define the bin edges for the continuous variable
#             bin_edges = np.linspace(temp_df[cont_col].min(), temp_df[cont_col].max(), num_bins + 1)

#             # Boxplot
#             plt.figure(figsize=(8,6))
#             sns.boxplot(x=cat_col, y=cont_col, data=temp_df)
#             plt.title(f"{cat_col} vs. {cont_col} (Boxplot)")
#             plt.show()

#             # Histogram
#             plt.figure(figsize=(8,6))
#             for category in temp_df[cat_col].unique():
#                 sns.histplot(temp_df[temp_df[cat_col] == category][cont_col], bins=bin_edges, kde=True, label=category)
#             plt.title(f"{cat_col} vs. {cont_col} (Histogram)")
#             plt.legend()
#             plt.show()
def plot_categorical_and_continuous_vars(tr_smpl, cat_var, cont_var, num_bins=50):
    for cat_col in cat_var:
        for cont_col in cont_var:
            # Subset the dataframe based on the loop variables
            temp_df = tr_smpl[[cat_col, cont_col]]

            # Create a single figure with subplots
            fig, axes = plt.subplots(2, 1, figsize=(8, 12))

            # Boxplot
            sns.boxplot(x=cat_col, y=cont_col, data=temp_df, ax=axes[0])
            axes[0].set_title(f"{cat_col} vs. {cont_col} (Boxplot)")

            # Histogram
            for category in temp_df[cat_col].unique():
                data = temp_df[temp_df[cat_col] == category][cont_col]
                axes[1].hist(data, bins=num_bins, label=category, alpha=0.7)
            axes[1].set_title(f"{cat_col} vs. {cont_col} (Histogram)")
            axes[1].legend()

            # Adjust spacing between subplots
            plt.tight_layout()

            # Show the figure
            plt.show()
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