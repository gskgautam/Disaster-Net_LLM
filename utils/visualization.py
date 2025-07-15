import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
import numpy as np

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_roc_curve(y_true, y_score, n_classes, save_path=None):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure(figsize=(8, 6))
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_attention_map(attn_weights, save_path=None):
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_weights, cmap='viridis')
    plt.title('Attention Map')
    plt.xlabel('Key')
    plt.ylabel('Query')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_category_metrics(metrics_dict, save_path=None):
    categories = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    plt.figure(figsize=(10, 6))
    sns.barplot(x=categories, y=values)
    plt.title('Category-wise Metrics')
    plt.ylabel('Score')
    if save_path:
        plt.savefig(save_path)
    plt.close() 