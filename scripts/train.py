import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from utils.config import Config
from datasets.medic import MedicDataset
from models.disasternet_llm import DisasterNetLLM


def train():
    device = Config.device if torch.cuda.is_available() else 'cpu'
    train_set = MedicDataset(Config.medic_dir, split='train')
    val_set = MedicDataset(Config.medic_dir, split='val')
    train_loader = DataLoader(train_set, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=Config.batch_size)

    num_classes = len(train_set.class_to_idx)
    model = DisasterNetLLM(num_classes=num_classes).to(device)
    optimizer = AdamW(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    criterion = CrossEntropyLoss()

    # Check if dataset provides text or geo data (for future extensibility)
    provides_text = False
    provides_geo = False
    sample = train_set[0]
    if isinstance(sample, (tuple, list)) and len(sample) > 2:
        provides_text = True
    if isinstance(sample, (tuple, list)) and len(sample) > 3:
        provides_geo = True

    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{Config.epochs} - Training'):
            # Unpack batch
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
                texts = [''] * imgs.size(0)  # Explicit dummy text
                geos = torch.zeros((imgs.size(0), 4, 32, 32)).to(device)  # Explicit dummy geo
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(texts, imgs, geos)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1} Train Loss: {total_loss/len(train_loader):.4f}')

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch+1}/{Config.epochs} - Validation'):
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
                outputs = model(texts, imgs, geos)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        print(f'Epoch {epoch+1} Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {correct/total:.4f}')

if __name__ == '__main__':
    train() 