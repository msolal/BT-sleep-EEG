import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def save_score(plots_path, test_bal_acc, test_kappa):
    bal = f'Test balanced accuracy: {test_bal_acc:0.3f}\n'
    kappa = f'Test Cohen\'s kappa: {test_kappa:0.3f}\n'
    print(bal, kappa)
    score_file = open(plots_path+'scores.txt', 'w')
    score_file.write(bal)
    score_file.write(kappa)
    score_file.close()
    
    
def plot_confusion_matrix(plots_path, conf_mat, classes_mapping):
    ticks = list(classes_mapping.keys())
    tick_labels = classes_mapping.values()

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(conf_mat, cmap='Blues')

    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels)
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_title('Confusion matrix')

    for i in range(len(ticks)):
        for j in range(len(ticks)):
            ax.text(j, i, conf_mat[i, j], ha='center', va='center', color='k')

    fig.colorbar(im, ax=ax, fraction=0.05, label='# examples')
    fig.tight_layout()

    plt.savefig(plots_path + 'confusion_matrix', facecolor='w')

    return fig, ax


def plot_history(plots_path, clf):
    # Extract loss and bal accuracy values for plotting from history object
    df = pd.DataFrame(clf.history.to_list())
    df[['train_mis_clf', 'valid_mis_clf']] = 100 - df[
        ['train_bal_acc', 'valid_bal_acc']] * 100

    # Get percent of misclass for better visual comparison to loss
    plt.style.use('seaborn-talk')
    fig, ax1 = plt.subplots(figsize=(8, 3))
    df.loc[:, ['train_loss', 'valid_loss']].plot(
        ax=ax1, style=['-', ':'], marker='o', color='tab:blue', legend=False,
        fontsize=14)

    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=14)
    ax1.set_ylabel("Loss", color='tab:blue', fontsize=14)
    ax2 = ax1.twinx()  # instantiate a second axis that shares the same x-axis
    df.loc[:, ['train_mis_clf', 'valid_mis_clf']].plot(
        ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=14)
    ax2.set_ylabel('Balanced misclassification rate [%]', color='tab:red',
                   fontsize=14)
    ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
    ax1.set_xlabel('Epoch', fontsize=14)

    # where some data has already been plotted to ax
    handles = []
    handles.append(
        Line2D([0], [0], color='black', linewidth=1,
               linestyle='-', label='Train'))
    handles.append(
        Line2D([0], [0], color='black', linewidth=1,
               linestyle=':', label='Valid'))
    plt.legend(handles, [h.get_label() for h in handles], fontsize=14)
    plt.tight_layout()
    plt.savefig(plots_path + 'history_plot', facecolor='w')
