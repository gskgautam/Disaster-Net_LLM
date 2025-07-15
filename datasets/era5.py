import os
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.error_handling import log_error, log_warning

class ERA5Dataset(Dataset):
    def __init__(self, root_dir, split='train', variables=None):
        self.root_dir = os.path.join(root_dir, 'processed')
        self.split = split
        self.variables = variables or ['t2m', 'u10', 'v10', 'tp']
        if not os.path.exists(self.root_dir):
            log_error(f"Processed ERA5 directory not found: {self.root_dir}")
            raise FileNotFoundError(f"Processed ERA5 directory not found: {self.root_dir}. Please run preprocessing.")
        self.files = self._get_split_files()
        if not self.files:
            log_error(f"No processed ERA5 files found in {self.root_dir}")
            raise RuntimeError(f"No processed ERA5 files found in {self.root_dir}. Check preprocessing and data structure.")

    def _get_split_files(self):
        all_files = [os.path.join(self.root_dir, f) for f in os.listdir(self.root_dir) if f.endswith('.npy')]
        all_files.sort()
        n = len(all_files)
        if n == 0:
            log_warning(f"No .npy files found in {self.root_dir}")
        if self.split == 'train':
            return all_files[:int(0.7*n)]
        elif self.split == 'val':
            return all_files[int(0.7*n):int(0.9*n)]
        else:
            return all_files[int(0.9*n):]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        try:
            data = np.load(file_path, allow_pickle=True).item()
        except Exception as e:
            log_error(f"Failed to load ERA5 file {file_path}: {e}")
            raise RuntimeError(f"Failed to load ERA5 file {file_path}: {e}")
        arrs = []
        for var in self.variables:
            if var in data:
                arrs.append(torch.tensor(data[var], dtype=torch.float32))
            else:
                log_warning(f"Variable '{var}' missing in file {file_path}")
        if not arrs:
            log_error(f"No valid variables found in file {file_path}")
            raise RuntimeError(f"No valid variables found in file {file_path}")
        arr = torch.stack(arrs, dim=0)  # shape: [variables, ...]
        return arr 