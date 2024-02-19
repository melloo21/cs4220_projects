import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

def run_evaluation(filepath):

    df = pd.read_csv(
    filepath,
    delimiter='\t'
    )

    filter_list = [elem for elem in df.columns if "FILTER" in elem]
    df_niave = df[filter_list + ['is_snv']]
    df_niave = df_niave.astype('int32')
    # 1 is True and 0 is false
    df_niave["sum_of_filters"] = df_niave[filter_list].sum(axis=1)
    df_niave["niave_method_1"] = np.where(df_niave["sum_of_filters"] >= 2, 1, 0)

    y_true = np.array(df_niave.is_snv)
    y_pred = np.array(df_niave.niave_method_1)

    # Returns precision/recall/f1 score

    # 'micro':
    # Calculate metrics globally by counting the total true positives, false negatives and false positives.
    # 'macro':
    # Calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
    # 'weighted':
    # Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label). This alters ‘macro’ to account for label imbalance; it can result in an F-score that is not between precision and recall.
    macro_score = precision_recall_fscore_support(y_true, y_pred, average='macro')[0:3]
    micro_score = precision_recall_fscore_support(y_true, y_pred, average='micro')[0:3]
    binary_score = precision_recall_fscore_support(y_true, y_pred, average='binary')[0:3]

    print(" Macro Precision : %5.2f, Recall : %5.2f, F1 : %5.2f" %  macro_score)
    print(" Micro Precision : %5.2f, Recall : %5.2f, F1 : %5.2f" %  micro_score)
    print(" binary Precision : %5.2f, Recall : %5.2f, F1 : %5.2f" %  binary_score)

    cm = confusion_matrix(y_true, y_pred)

    ax= plt.subplot()
    sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

    # labels, title and ticks
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['False', 'True']); ax.yaxis.set_ticklabels(['False', 'True'])

    return macro_score, micro_score, binary_score