import torch
from torch.utils.data import DataLoader
from datasets.disaster_image import DisasterImageDataset
from models.image_encoder import ImageEncoder
from models.classifier import Classifier
from utils.metrics import compute_classification_metrics, compute_auc
from utils.visualization import plot_confusion_matrix, plot_roc_curve
from utils.config import Config
import numpy as np

def main():
    device = Config.device if torch.cuda.is_available() else 'cpu'
    test_set = DisasterImageDataset(Config.disaster_image_dir, split='test')
    test_loader = DataLoader(test_set, batch_size=Config.batch_size)
    num_classes = len(test_set.class_to_idx)
    encoder = ImageEncoder(output_dim=512).to(device)
    classifier = Classifier(512, num_classes).to(device)
    encoder.eval()
    classifier.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            feats = encoder(imgs)
            probs = classifier(feats)
            preds = probs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.concatenate(all_probs, axis=0)
    metrics = compute_classification_metrics(all_labels, all_preds)
    auc_score = compute_auc(all_labels, all_probs)
    print('Image-only Test Metrics:', metrics)
    print('AUC:', auc_score)
    plot_confusion_matrix(all_labels, all_preds, class_names=list(test_set.class_to_idx.keys()), save_path='image_confusion_matrix.png')
    plot_roc_curve(all_labels, all_probs, n_classes=num_classes, save_path='image_roc_curve.png')

if __name__ == '__main__':
    main() 