import torch
from torch.utils.data import DataLoader
from datasets.era5 import ERA5Dataset
from models.disasternet_llm import DisasterNetLLM
from utils.metrics import compute_classification_metrics, compute_auc
from utils.visualization import plot_confusion_matrix, plot_roc_curve
from utils.config import Config
import numpy as np

def main():
    device = Config.device if torch.cuda.is_available() else 'cpu'
    test_set = ERA5Dataset(Config.era5_dir, split='test')
    test_loader = DataLoader(test_set, batch_size=Config.batch_size)
    num_classes = 2  # Example: binary prediction (adjust as needed)
    model = DisasterNetLLM(num_classes=num_classes).to(device)
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []


    provides_text = False
    provides_image = False
    provides_geo = False
    sample = test_set[0]
    if isinstance(sample, (tuple, list)) and len(sample) > 2:
        provides_text = True
    if isinstance(sample, (tuple, list)) and len(sample) > 1:
        provides_image = True
    # For ERA5, geo is always provided (single tensor)
    if isinstance(sample, torch.Tensor):
        provides_geo = True

    with torch.no_grad():
        for batch in test_loader:
            if provides_text and provides_image and provides_geo:
                texts, images, geos = batch
            elif provides_image and provides_geo:
                images, geos = batch
                texts = [''] * images.size(0)
            elif provides_geo:
                geos = batch
                batch_size = geos.size(0)
                texts = [''] * batch_size
                images = torch.zeros((batch_size, 3, 224, 224)).to(device)
            else:
                raise ValueError('ERA5 experiment expects at least geo data.')
            geos = geos.to(device)
            
            labels = torch.zeros(geos.size(0), dtype=torch.long).to(device)
            probs = model(texts, images, geos)
            preds = probs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.concatenate(all_probs, axis=0)
    metrics = compute_classification_metrics(all_labels, all_preds)
    auc_score = compute_auc(all_labels, all_probs)
    print('Multimodal (ERA5) Test Metrics:', metrics)
    print('AUC:', auc_score)
    plot_confusion_matrix(all_labels, all_preds, class_names=[str(i) for i in range(num_classes)], save_path='era5_confusion_matrix.png')
    plot_roc_curve(all_labels, all_probs, n_classes=num_classes, save_path='era5_roc_curve.png')

if __name__ == '__main__':
    main() 