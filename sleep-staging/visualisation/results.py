import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


cmap = sns.cubehelix_palette(50)


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
        fontsize=12)

    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=12)
    ax1.set_ylabel("Loss", color='tab:blue', fontsize=12)
    ax2 = ax1.twinx()  # instantiate a second axis that shares the same x-axis
    df.loc[:, ['train_mis_clf', 'valid_mis_clf']].plot(
        ax=ax2, style=['-', ':'], marker='o', color='tab:red', legend=False)
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=12)
    ax2.set_ylabel('Balanced misclassification rate [%]', color='tab:red',
                   fontsize=10)
    ax2.set_ylim(ax2.get_ylim()[0], 85)  # make some room for legend
    ax1.set_xlabel('Epoch', fontsize=12)

    # where some data has already been plotted to ax
    handles = []
    handles.append(
        Line2D([0], [0], color='black', linewidth=1,
               linestyle='-', label='Train'))
    handles.append(
        Line2D([0], [0], color='black', linewidth=1,
               linestyle=':', label='Valid'))
    plt.legend(handles, [h.get_label() for h in handles], fontsize=12)
    plt.tight_layout()
    plt.savefig(plots_path + 'history_plot', facecolor='w')


def save_score(plots_path, test_bal_acc, test_kappa):
    bal = f'Test balanced accuracy: {test_bal_acc:0.3f}\n'
    kappa = f'Test Cohen\'s kappa: {test_kappa:0.3f}\n'
    print(bal, kappa)
    score_file = open(plots_path+'scores.txt', 'w')
    score_file.write(bal)
    score_file.write(kappa)
    score_file.close()


def plot_confusion_matrix(plots_path, conf_mat, classes_mapping):
    confusion_df = pd.DataFrame(conf_mat, columns=classes_mapping.values(),
                                index=classes_mapping.values())
    plt.figure()
    ax = sns.heatmap(confusion_df, annot=True, cmap=cmap, linewidths=.01, square=True)
    ax.set(xlabel='Predicted Labels', ylabel='True Labels')
    ax.tick_params(left=False, bottom=False)
    plt.yticks(rotation=0) 
    plt.title('Confusion matrix')
    plt.savefig(plots_path + 'confusion_matrix', facecolor='w')


def plot_classification_report(plots_path, class_report, classes_mapping):
    class_report = class_report.replace('\n\n', '\n')
    class_report = class_report.replace(' / ', '/')
    lines = class_report.split('\n')

    classes, values_matrix, support, mask_matrix = [], [], [], []
    for line in lines[1:-1]: 
        splitted_line = line.strip().split()
        support.append(int(splitted_line[-1]))
        if len(splitted_line) == 3:
            classes.append(splitted_line[0])
            values = [0, 0, float(splitted_line[1])]
            mask = [True, True, False]
        elif len(splitted_line) > 5:
            classes.append(splitted_line[0]+'_'+splitted_line[1])
            values = [float(x) for x in splitted_line[2: -1]]
            mask = [False, False, False]
        else:
            classes.append(splitted_line[0])
            values = [float(x) for x in splitted_line[1: -1]]
            mask = [False, False, False]
        values_matrix.append(values)
        mask_matrix.append(mask)

    values_matrix = np.array(values_matrix)
    mask_matrix = np.array(mask_matrix)
    xlabels = ['Precision', 'Recall', 'F1-score']
    ylabels = ['{} ({})'.format(classes_mapping[idx] if idx in classes_mapping else idx, sup)
                   for idx, sup in zip(classes, support)]

    report_df = pd.DataFrame(values_matrix, columns=xlabels, index=ylabels)
    plt.figure()
    ax = sns.heatmap(report_df, annot=True, cmap=cmap, linewidths=.01, square=True, mask=mask_matrix)
    ax.tick_params(left=False, bottom=False)
    # plt.xticks(rotation=0) 
    plt.title('Classification report')

    plt.savefig(plots_path + 'classification_report', facecolor='w')
