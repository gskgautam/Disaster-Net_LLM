import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error, mean_squared_error
from bert_score import score as bert_score

def compute_classification_metrics(y_true, y_pred, average='macro'):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0),
    }
    return metrics

def compute_auc(y_true, y_score, multi_class='ovr'):
    try:
        return roc_auc_score(y_true, y_score, multi_class=multi_class)
    except Exception:
        return None

def compute_bert_score(cands, refs, lang='en'):
    P, R, F1 = bert_score(cands, refs, lang=lang)
    return {'bertscore_precision': float(P.mean()), 'bertscore_recall': float(R.mean()), 'bertscore_f1': float(F1.mean())}

def compute_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return {'mae': mae, 'rmse': rmse} 