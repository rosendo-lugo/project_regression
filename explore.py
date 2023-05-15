import pandas as pd 
# Viz imports
import matplotlib.pyplot as plt
import seaborn as sns


# ----------------------------------------------------------------------------------
def plot_variable_pairs(tr_sub):
    # We can drop the redundant information in the upper right half of the chart if we like.
    g = sns.pairplot(data=tr_sub.drop(columns='property_value'), corner=True, kind='reg')
    g.map_lower(sns.regplot, line_kws={'color': 'orange'})
    plt.show()

# ----------------------------------------------------------------------------------
def plot_categorical_and_continuous_vars(tr_sub, cat_var, cont_var):
    for cat_col in cat_var:
        for cont_col in cont_var:
            # Subset the dataframe based on the loop variables
            temp_df = tr_sub[[cat_col, cont_col]]

            # Boxplot
            plt.figure(figsize=(8,6))
            sns.boxplot(x=cat_col, y=cont_col, data=temp_df)
            plt.title(f"{cat_col} vs. {cont_col} (Boxplot)")
            plt.show()

            # Violin plot
            plt.figure(figsize=(8,6))
            sns.violinplot(x=cat_col, y=cont_col, data=temp_df)
            plt.title(f"{cat_col} vs. {cont_col} (Violin plot)")
            plt.show()

            # Swarm plot
            plt.figure(figsize=(8,6))
            sns.swarmplot(x=cat_col, y=cont_col, data=temp_df)
            plt.title(f"{cat_col} vs. {cont_col} (Swarm plot)")
            plt.show()