import os
import pandas as pd
from torch.utils.data import Dataset
from utils.error_handling import log_error, log_warning

class DelhiUrbanRiskDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.csv_path = os.path.join(root_dir, 'processed', 'events.csv')
        self.transform = transform
        if not os.path.exists(self.csv_path):
            log_error(f"Processed events CSV not found: {self.csv_path}")
            raise FileNotFoundError(f"Processed events CSV not found: {self.csv_path}. Please run preprocessing.")
        try:
            self.df = pd.read_csv(self.csv_path)
        except Exception as e:
            log_error(f"Failed to read events CSV {self.csv_path}: {e}")
            raise RuntimeError(f"Failed to read events CSV {self.csv_path}: {e}")
        if self.df.empty:
            log_error(f"Events CSV is empty: {self.csv_path}")
            raise RuntimeError(f"Events CSV is empty: {self.csv_path}. Check preprocessing and data structure.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = {
            'event': row.get('event', ''),
            'date': row.get('date', ''),
            'location': row.get('location', '')
        }
        if self.transform:
            sample = self.transform(sample)
        return sample 