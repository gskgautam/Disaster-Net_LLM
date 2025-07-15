import os
import random
from PIL import Image, UnidentifiedImageError
from torchvision import transforms
from sklearn.model_selection import train_test_split
from utils.error_handling import init_logging, log_error, log_warning, log_info

DATA_DIR = 'Disaster Image Dataset/'
OUTPUT_DIR = 'Disaster Image Dataset/processed/'
IMG_SIZE = 224
SPLITS = {'train': 0.7, 'val': 0.2, 'test': 0.1}
LOG_FILE = 'Disaster Image Dataset/preprocess_error.log'

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
])

def get_image_paths_and_labels(data_dir):
    image_paths = []
    labels = []
    if not os.path.exists(data_dir):
        log_error(f"Data directory not found: {data_dir}")
        return image_paths, labels
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            log_warning(f"Skipping non-directory: {class_dir}")
            continue
        for fname in os.listdir(class_dir):
            if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                fpath = os.path.join(class_dir, fname)
                if not os.path.isfile(fpath):
                    log_warning(f"File not found: {fpath}")
                    continue
                image_paths.append(fpath)
                labels.append(class_name)
    if not image_paths:
        log_error("No images found in dataset.")
    return image_paths, labels

def save_split(split, paths, labels):
    split_dir = os.path.join(OUTPUT_DIR, split)
    os.makedirs(split_dir, exist_ok=True)
    for path, label in zip(paths, labels):
        label_dir = os.path.join(split_dir, label)
        os.makedirs(label_dir, exist_ok=True)
        try:
            img = Image.open(path).convert('RGB')
            img = transform(img)
            img.save(os.path.join(label_dir, os.path.basename(path)))
        except (UnidentifiedImageError, OSError) as e:
            log_error(f"Failed to process image {path}: {e}")
        except Exception as e:
            log_error(f"Unexpected error processing {path}: {e}")

def main():
    init_logging(LOG_FILE)
    image_paths, labels = get_image_paths_and_labels(DATA_DIR)
    if not image_paths or not labels:
        log_error("No valid images or labels found. Exiting.")
        return
    # Check for class consistency
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        log_warning(f"Only {len(unique_labels)} class(es) found. Check dataset structure.")
    X_temp, X_test, y_temp, y_test = train_test_split(image_paths, labels, test_size=SPLITS['test'], stratify=labels, random_state=42)
    val_size = SPLITS['val'] / (SPLITS['train'] + SPLITS['val'])
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=val_size, stratify=y_temp, random_state=42)
    save_split('train', X_train, y_train)
    save_split('val', X_val, y_val)
    save_split('test', X_test, y_test)
    log_info(f'Preprocessing complete. Processed data saved to {OUTPUT_DIR}')
    print('Preprocessing complete. Processed data saved to', OUTPUT_DIR)

if __name__ == '__main__':
    main() 