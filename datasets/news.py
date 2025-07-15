import os
import pandas as pd
from torch.utils.data import Dataset
from utils.error_handling import log_error, log_warning

class NewsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.csv_path = os.path.join(root_dir, 'processed', 'news.csv')
        self.transform = transform
        if not os.path.exists(self.csv_path):
            log_error(f"Processed news CSV not found: {self.csv_path}")
            raise FileNotFoundError(f"Processed news CSV not found: {self.csv_path}. Please run preprocessing.")
        try:
            self.df = pd.read_csv(self.csv_path)
        except Exception as e:
            log_error(f"Failed to read news CSV {self.csv_path}: {e}")
            raise RuntimeError(f"Failed to read news CSV {self.csv_path}: {e}")
        if self.df.empty:
            log_error(f"News CSV is empty: {self.csv_path}")
            raise RuntimeError(f"News CSV is empty: {self.csv_path}. Check preprocessing and data structure.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample = {
            'title': row.get('title', ''),
            'content': row.get('content', ''),
            'date': row.get('date', ''),
            'publisher': row.get('publisher', '')
        }
        if self.transform:
            sample = self.transform(sample)
        return sample 