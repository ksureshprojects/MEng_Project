import matplotlib.pyplot as plt
import numpy as np
def barPlot(results, title, save=None):
    # Sort by rank error
    arg = np.argsort(results[:,0])
    # Store new results
    results_s = results[arg, :]
    # the label locations
    x = np.arange(results_s[:,3].flatten().shape[0])  
    # the width of the bars
    width = 0.65 

    fig, ax = plt.subplots(3,1, sharex=True, figsize=[20,4.8])
    rects1 = ax[0].bar(x, results_s[:,0].astype('float').flatten(), 
                       width, label='Rank Error')
    rects3 = ax[1].bar(x, results_s[:,1].astype('float').flatten(), 
                       width, label='Ellipse MSE')
    rects2 = ax[2].bar(x, results_s[:,2].astype('float').flatten(), 
                       width, label='Ellipse Width')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax[0].set_title(title)
    ax[0].set_ylabel('Rank Error')
    ax[1].set_ylabel('Ellipse MSE')
    ax[2].set_ylabel('Ellipse Width')
    ax[2].set_xlabel('Method')
    ax[2].set_xticks(x)
    ax[2].set_xticklabels(results_s[:,3].flatten(), rotation=90)
    if save:
        fig.savefig(save)