import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

line_x = [i/10 for i in range(0,11)]
line_y = [i/10 for i in range(0,11)]

def plot_spec_vs_any(df, auc_sp, auc_avg, t=None, xmax=None, ymin=None, ymax=None, out_file=None, tpr=False):
    print('ab')
    fig, axs = plt.subplots(1,1, figsize=(8,8))

    if t is not None:
        if ymin is None:
            y_min = 0
        else:
            y_min = ymin
        if ymax is None:
            y_max=t
        else:
            y_max = ymax
        if xmax is None:
            x_max = 1.0
        else:
            x_max = xmin
        plt.axvspan(xmin=t, xmax=x_max, ymin=y_min, ymax=y_max, color='grey', alpha=0.3, label='High-risk records missed')
    sns.scatterplot(df, x=auc_sp, y=auc_avg, ax=axs, s=80)
    #plt.axis('square')

    if tpr:
        axs.set_xlim(0,1.0)
        axs.set_ylim(0,1.0)
    else:
        axs.set_xlim(0.3, 1.0)
        axs.set_ylim(0.3, 1.0)
    axs.set_aspect('equal')
    sns.lineplot(x=line_x, y=line_y, color='black', ax=axs, linestyle='dashed')#, label='y=x')
    
    #axs.axvline(x=t, ymin=0, ymax=1, color='grey', linestyle='dashed', label=f'High-risk threshold={t}')
    #axs.axhline(y=t, xmin=0, xmax=1, color='grey', linestyle='dashed')
    
    axs.set_xlabel(r'$R_{sp}^{AUC}$', fontsize=20)
    axs.set_ylabel(r'$R_{avg}^{AUC}$', fontsize=20)

    axs.set_xticklabels(axs.get_xticklabels(), fontsize=20)
    axs.set_yticklabels(axs.get_yticklabels(), fontsize=20)
    
    plt.legend(loc='upper left', fontsize=15)
    #axs.set_title(f'AUC specific VS any for {len(auc_df)} points')
    plt.tight_layout()
    #plt.savefig(f'../figures/synthetic/auc_spec_vs_any_{len(auc_df)}_missrate.png')
    if out_file is not None:
        plt.savefig(out_file+'.pdf', bbox_inches='tight')
        plt.savefig(out_file+'.svg', bbox_inches='tight')