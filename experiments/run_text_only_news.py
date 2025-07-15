import torch
from torch.utils.data import DataLoader
from datasets.news import NewsDataset
from models.text_encoder import TextEncoder
from models.classifier import Classifier
from utils.metrics import compute_classification_metrics, compute_auc, compute_bert_score
from utils.visualization import plot_confusion_matrix, plot_roc_curve
from utils.config import Config
import numpy as np

def main():
    device = Config.device if torch.cuda.is_available() else 'cpu'
    test_set = NewsDataset(Config.news_dir)
    test_loader = DataLoader(test_set, batch_size=Config.batch_size)
    num_classes = 2  # Example: binary (adjust as needed)
    encoder = TextEncoder(output_dim=768).to(device)
    classifier = Classifier(768, num_classes).to(device)
    encoder.eval()
    classifier.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    all_contents = []
    all_titles = []
    with torch.no_grad():
        for batch in test_loader:
            texts = batch['content']
          
            labels = torch.zeros(len(texts), dtype=torch.long).to(device)
            feats = encoder(texts)
            probs = classifier(feats)
            preds = probs.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_contents.extend(texts)
            all_titles.extend(batch['title'])
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.concatenate(all_probs, axis=0)
    metrics = compute_classification_metrics(all_labels, all_preds)
    auc_score = compute_auc(all_labels, all_probs)
    print('Text-only (News) Test Metrics:', metrics)
    print('AUC:', auc_score)
    plot_confusion_matrix(all_labels, all_preds, class_names=[str(i) for i in range(num_classes)], save_path='news_confusion_matrix.png')
    plot_roc_curve(all_labels, all_probs, n_classes=num_classes, save_path='news_roc_curve.png')
    
    bertscore = compute_bert_score(all_titles, all_contents)
    print('BERTScore:', bertscore)

if __name__ == '__main__':
    main() 