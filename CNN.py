import pandas as pd
import numpy as np
from PIL import Image
import io
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import joblib
import os
import argparse

# ── Argparse für Timeout ──────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--timeout_seconds', type=int, default=1800)
args = parser.parse_args()

# ── Reproduzierbarkeit ────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

# ── Dataset ───────────────────────────────────────────────────────────────────
class ImageDataset(Dataset):
    def __init__(self, df, transform=None):
        self.images = df['image'].tolist()
        self.labels = df['label'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self.images[idx])).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ── Modell (aus Appendix B) ───────────────────────────────────────────────────
k = 32
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, k, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(k, 2*k, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(2*k, 4*k, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4*k, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ── Training ──────────────────────────────────────────────────────────────────
def train():
    # Daten laden
    df = pd.read_parquet('data/train/')
    df['label'] = df['source_class'].apply(lambda x: 0 if x == 0 else 1)

    # Balancieren
    df_real = df[df['label'] == 0]
    df_ai = df[df['label'] == 1].sample(n=len(df_real), random_state=42)
    df_balanced = pd.concat([df_real, df_ai]).sample(frac=1, random_state=42)

    dataset = ImageDataset(df_balanced, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0)

    # Modell
    model = CNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Training Loop
    import time
    start = time.time()

    for epoch in range(20):
        model.train()
        total_loss = 0

        counter = 0
        for images, labels in dataloader:
            if counter % 10 == 0:
                print(f'{epoch}:{counter}')
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = criterion(output, labels.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            counter += 1

        elapsed = time.time() - start
        print(f"Epoch {epoch+1}/20 - Loss: {total_loss/len(dataloader):.4f} - Zeit: {elapsed:.0f}s")

        # Checkpoint speichern
        os.makedirs('artifacts/task02', exist_ok=True)
        torch.save(model.state_dict(), 'artifacts/task02/model.pt')

        # Timeout check
        if elapsed > args.timeout_seconds * 0.9:
            print("Timeout erreicht, stoppe Training")
            break

    print("Training fertig")

if __name__ == '__main__':
    train()