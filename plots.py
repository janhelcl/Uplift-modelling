import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def response_plot(results: pd.DataFrame) -> None:
    """Plots response rate by score decile
    
    :param results: output of chosen methodology
    """
    results = results.copy()
    
    results['Decile'] = pd.qcut(results['y_pred'], 10, labels=False)
    results['treatment'] = results['treatment'].map({
        1: 'treatment',
        0: 'control'
    })
    
    plt.figure(figsize=(10, 5))
    chart = sns.barplot(x=results['Decile'],
                        y=results['y'],
                        hue=results['treatment'],
                        ci=False,
                        
                       )
    chart.axhline(results[results['treatment']=='treatment']['y'].mean(),
                  linestyle='--',
                  color='k',
                  alpha=0.5,
                  label='treatment avg')
    chart.axhline(results[results['treatment']=='control']['y'].mean(),
                  linestyle='--',
                  color='r',
                  alpha=0.5,
                  label='control avg')
    plt.title('Response rate by decile')
    plt.legend()
    
    
def true_lift(results: pd.DataFrame) -> None:
    """Plots true lift by deciles
    
    :param results: output of chosen methodology
    """
    results = results.copy()
    
    results['Decile'] = pd.qcut(results['y_pred'], 10, labels=False)
    
    increment_model = results[results['treatment']==1].groupby('Decile').mean()['y'] - \
                      results[results['treatment']==0].groupby('Decile').mean()['y']
    
    increment_random = results[results['treatment']==1]['y'].mean() - \
                       results[results['treatment']==0]['y'].mean()
    
    true_lift = increment_model - increment_random

    true_lift.plot.bar(figsize=(10, 5),
                       title='True lift by decile'
                      )
