import torch
from torch.utils.data import DataLoader, TensorDataset
from datasets.delhi_urban import DelhiUrbanRiskDataset
from utils.metrics import compute_classification_metrics
from utils.visualization import plot_confusion_matrix
from utils.config import Config
import numpy as np
from torch import nn
from sklearn.preprocessing import LabelEncoder

class SimpleLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        out = self.fc(h[-1])
        return out

def encode_sample(sample, event_encoder, location_encoder, date_encoder):
    # Encode event, date, location as integers or embeddings
    event = event_encoder.transform([sample['event']])[0]
    location = location_encoder.transform([sample['location']])[0]
    date = date_encoder.transform([sample['date']])[0]
    return np.array([event, date, location], dtype=np.float32)

def main():
    device = Config.device if torch.cuda.is_available() else 'cpu'
    test_set = DelhiUrbanRiskDataset(Config.delhi_urban_dir)
    # Fit label encoders on the whole dataset
    events = [row['event'] for row in test_set]
    locations = [row['location'] for row in test_set]
    dates = [row['date'] for row in test_set]
    event_encoder = LabelEncoder().fit(events)
    location_encoder = LabelEncoder().fit(locations)
    date_encoder = LabelEncoder().fit(dates)
    # Prepare data
    X = []
    y = []
    for sample in test_set:
        X.append(encode_sample(sample, event_encoder, location_encoder, date_encoder))
        
        y.append(0)  
    X = np.stack(X)
    y = np.array(y)
    # Reshape for LSTM: [batch, seq_len, input_dim] (seq_len=1 for tabular)
    X = X[:, np.newaxis, :]
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=Config.batch_size)
    num_classes = 4  # Example: fire, unknown bags, building collapse, gas/water leaks
    input_dim = 3  # event, date, location (encoded)
    model = SimpleLSTMClassifier(input_dim, 64, num_classes).to(device)
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for x, labels in test_loader:
            x, labels = x.to(device), labels.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    metrics = compute_classification_metrics(all_labels, all_preds)
    print('Time Series (Delhi Urban) Test Metrics:', metrics)
    plot_confusion_matrix(all_labels, all_preds, class_names=[str(i) for i in range(num_classes)], save_path='delhi_confusion_matrix.png')

if __name__ == '__main__':
    main() 