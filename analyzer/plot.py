import numpy as np
import pandas as pd
import matplotlib.pyplot as plt    

def plot_log_scale_roc_curve(audit_results, logdir, filename):
    mr = audit_results[0]
    fpr = mr.fp / (mr.fp + mr.tn) 
    tpr = mr.tp / (mr.tp + mr.fn) 
    
    df = pd.DataFrame(data={
            'fpr': fpr,
            'tpr': tpr
        })
    
    filepath = logdir+'/data.csv'
    df.to_csv(filepath)
    roc_auc = np.trapz(x=fpr, y=tpr)
    range01 = np.linspace(0, 1)
    
    plt.fill_between(fpr, tpr, alpha=0.15)
    plt.scatter(fpr, tpr)
    plt.plot(fpr, tpr, label="log scale (logit rescaling method)")
    plt.plot(range01, range01, '--', label='Random guess')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim([10**(-5), 1])
    plt.ylim([10**(-5), 1])
    plt.grid()
    plt.legend()
    plt.xlabel('False positive rate (FPR)')
    plt.ylabel('True positive rate (TPR)')
    plt.title('ROC curve')
    plt.text(
                0.4, 0.1,
                f'AUC = {roc_auc:.03f}',
                horizontalalignment='center',
                verticalalignment='center',
                bbox=dict(facecolor='white', alpha=0.5)
            )
    if filename:
        plt.savefig(filename)

    return roc_auc