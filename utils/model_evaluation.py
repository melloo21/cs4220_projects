import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

def _read_model(
    model_name:str,
    filepath:str
):
    model_name = model_name if ".joblib" in model_name else f"{model_name}.joblib"
    file = f"{filepath}/{model_name}"
    model = joblib.load(file)   

    return model

def _draw_confusion(    
    data_set:tuple,
    model_name:str,
    filepath:str,
    data_type:str
):
    x_val , y_true = data_set
    model = _read_model(
        model_name=model_name,
        filepath=filepath
    )
    
    y_pred = model.predict(x_val)
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
    ax.set_title(f'Confusion Matrix for {data_type}'); 
    ax.xaxis.set_ticklabels(['False', 'True']); ax.yaxis.set_ticklabels(['False', 'True'])
    
    plt.show()

def performance_evaluate(
    train_dataset:tuple,
    valid_dataset:tuple,
    model_name:str,
    filepath:str
    ):
    """
        Summary 
        Args:
        data_set:tuple -- (x_values, y_values) in numpy array format
        model_name:str  -- model file name to extract      
    """

    # Init
    _draw_confusion(
        data_set=train_dataset,
        model_name=model_name,
        filepath=filepath,
        data_type="Train data"       
    )
    _draw_confusion(
        data_set=valid_dataset,
        model_name=model_name,
        filepath=filepath,
        data_type="Valid data"       
    )

    return