import os
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from utils.error_handling import log_error, log_warning

class DisasterImageDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, 'processed', split)
        self.transform = transform
        self.samples = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        if not os.path.exists(self.root_dir):
            log_error(f"Dataset split directory not found: {self.root_dir}")
            raise FileNotFoundError(f"Dataset split directory not found: {self.root_dir}. Please run preprocessing.")
        self._load_samples()
        if not self.samples:
            log_error(f"No image samples found in {self.root_dir}")
            raise RuntimeError(f"No image samples found in {self.root_dir}. Check preprocessing and data structure.")

    def _load_samples(self):
        classes = sorted(os.listdir(self.root_dir))
        if not classes:
            log_warning(f"No class folders found in {self.root_dir}")
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}
        self.idx_to_class = {i: cls for cls, i in self.class_to_idx.items()}
        for cls in classes:
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_dir):
                log_warning(f"Skipping non-directory: {cls_dir}")
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    fpath = os.path.join(cls_dir, fname)
                    if not os.path.isfile(fpath):
                        log_warning(f"File not found: {fpath}")
                        continue
                    self.samples.append(fpath)
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except (UnidentifiedImageError, OSError) as e:
            log_error(f"Failed to open image {img_path}: {e}")
            raise RuntimeError(f"Failed to open image {img_path}: {e}")
        if self.transform:
            img = self.transform(img)
        return img, label 