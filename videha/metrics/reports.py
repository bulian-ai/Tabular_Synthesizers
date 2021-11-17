import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def plot_distributions(real_data, synthetic_data,discrete_columns,numeric_columns,**kwargs):
    
    print("------------------------------- NUMERIC FEATURE DISTRIBUTIONS -------------------------------")
    for numeric_feat in numeric_columns:
        plt.figure(figsize=(20,4))
        fig = sns.kdeplot(synthetic_data[numeric_feat], shade=True,label='Synthetic Data')
        fig = sns.kdeplot(real_data[numeric_feat], shade=True, label='Real Data')

        fig.figure.suptitle(f"Numeric Density Distribution : {numeric_feat} ", fontsize = 16)

        plt.xlabel(f'{numeric_feat}', fontsize=10)
        plt.ylabel('Distribution', fontsize=10)
        plt.legend(loc='upper right')
        plt.show()

    print("------------------------------- CATEGORICAL FEATURE DISTRIBUTIONS -------------------------------")

    for categ_feat in discrete_columns:
        plt.figure(figsize=(20,4))
        plt.hist(real_data[categ_feat], 
                label='Real Data',alpha=0.2,density=True)

        plt.hist(synthetic_data[categ_feat], 
                label='Synthetic Data',alpha=0.2,density=True)

        plt.legend(loc='upper right')
        plt.title(f'Categorical Density Distribution : {categ_feat}',fontsize=16)
        plt.tick_params(axis='x', rotation=90)
        plt.show()
