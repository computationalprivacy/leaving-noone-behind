import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.distance import top_n_vulnerable_dists


def plot_achilles(distances: dict, n: int):

    sns.set_theme(style='white')

    top_n = top_n_vulnerable_dists(distances, n)

    thresh = np.min(top_n)

    plot_df = pd.DataFrame({
        'Achilles score': distances.values()
    }, index=distances.keys())

    plot_df = plot_df.assign(Vulnerability = plot_df['Achilles score'].map(lambda x: 'Low' if x<thresh else 'High'))
    # plot_df.loc[top_n,'Vulnerability'] = 'High'

    fig, axs = plt.subplots(1,1)

    sns.histplot(plot_df, x='Achilles score', hue='Vulnerability',
                ax=axs, alpha=1, hue_order=['High', 'Low'],
                palette=['#ff0083', '#89d98c'], stat='probability')

    plt.savefig('achilles_scores.png')


def calculate_statistics(distances: dict):
    mean_score = np.mean(list(distances.values()))
    q3 = np.quantile(list(distances.values()), 0.75)
    
    perc_below_avg = len([k for k in distances if distances[k]<mean_score])/len(distances.keys()) * 100
    
    print(f'{perc_below_avg:.2f}% of the records in the target dataset have a below-average Achilles score.')
    print(f'The third quantile is {q3:.2f}, i.e. three quarters (75%) of the records have an Achilles score below {q3:.2f}.')

def plot_risks(df: pd.DataFrame):
    _, axs = plt.subplots(1,2,figsize=(12,4))

    sns.barplot(data=df, x='record', y='auc', ax=axs[0], color='#73d2ff')
    axs[0].set_ylim(0.3, 1.0)
    axs[0].set_ylabel('MIA AUC')

    sns.barplot(data=df, x='record', y='accuracy', ax=axs[1], color='#73d2ff')
    axs[1].set_ylim(0.3, 1.0)
    axs[1].set_ylabel('MIA Accuracy')

    plt.savefig('high_risk_record_mia_scores.png')
