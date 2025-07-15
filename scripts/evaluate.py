import torch
from torch.utils.data import DataLoader
from utils.config import Config
from datasets.medic import MedicDataset
from models.disasternet_llm import DisasterNetLLM
from utils.metrics import compute_classification_metrics, compute_auc
from utils.visualization import plot_confusion_matrix, plot_roc_curve
import numpy as np
import os

def evaluate(model_path=None):
    device = Config.device if torch.cuda.is_available() else 'cpu'
    test_set = MedicDataset(Config.medic_dir, split='test')
    test_loader = DataLoader(test_set, batch_size=Config.batch_size)
    num_classes = len(test_set.class_to_idx)
    model = DisasterNetLLM(num_classes=num_classes).to(device)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []


    provides_text = False
    provides_geo = False
    sample = test_set[0]
    if isinstance(sample, (tuple, list)) and len(sample) > 2:
        provides_text = True
    if isinstance(sample, (tuple, list)) and len(sample) > 3:
        provides_geo = True

    with torch.no_grad():
        for batch in test_loader:
            if provides_text and provides_geo:
                imgs, labels, texts, geos = batch
            elif provides_text:
                imgs, labels, texts = batch
                geos = torch.zeros((imgs.size(0), 4, 32, 32)).to(device)
            elif provides_geo:
                imgs, labels, geos = batch
                texts = [''] * imgs.size(0)
            else:
                imgs, labels = batch
                texts = [''] * imgs.size(0)
                geos = torch.zeros((imgs.size(0), 4, 32, 32)).to(device)
            imgs, labels = imgs.to(device), labels.to(device)
            probs = model(texts, imgs, geos)
            preds = probs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.concatenate(all_probs, axis=0)
    metrics = compute_classification_metrics(all_labels, all_preds)
    auc_score = compute_auc(all_labels, all_probs)
    print('Test Metrics:', metrics)
    print('AUC:', auc_score)
    plot_confusion_matrix(all_labels, all_preds, class_names=list(test_set.class_to_idx.keys()), save_path='confusion_matrix.png')
    plot_roc_curve(all_labels, all_probs, n_classes=num_classes, save_path='roc_curve.png')

if __name__ == '__main__':
    evaluate() 