import os
import random
import shutil
import csv
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.autoaugment import RandAugment
from timm import create_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# === Config ===
batch_size = 64
learning_rate = 1e-5
total_epochs = 15
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = 12

# === Transforms ===
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(30),
    RandAugment(num_ops=2, magnitude=9),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3), value='random'),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# === Dataset Split ===
def split_train_to_val(source_dir, val_ratio=0.2, seed=42):
    source_dir = Path(source_dir)
    val_dir = source_dir.parent / "val"
    random.seed(seed)
    classes = ["0_real", "1_fake"]

    for class_name in classes:
        train_class_dir = source_dir / class_name
        val_class_dir = val_dir / class_name

        if not train_class_dir.exists():
            print(f"Skipping missing class folder: {train_class_dir}")
            continue

        if val_class_dir.exists() and any(val_class_dir.iterdir()):
            print(f"Validation folder already exists and is not empty: {val_class_dir}. Skipping.")
            continue

        val_class_dir.mkdir(parents=True, exist_ok=True)

        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        files = [f for f in train_class_dir.iterdir() if f.suffix.lower() in image_extensions and f.is_file()]
        if not files:
            print(f"No images found in {train_class_dir}. Skipping.")
            continue

        num_to_copy = max(1, int(len(files) * val_ratio))
        files_to_copy = random.sample(files, num_to_copy)

        print(f"Copying {num_to_copy}/{len(files)} images from {class_name} to validation")

        for f in files_to_copy:
            dest = val_class_dir / f.name
            shutil.copy(str(f), str(dest))

    print(f"\n✅ Validation set created at: {val_dir}")

# === Dataset Class ===
class data_loader(Dataset):
    def __init__(self, data_dir, transform=None):
        real = os.path.join(data_dir, '0_real')
        fake = os.path.join(data_dir, '1_fake')
        self.full_filenames = [os.path.join(real, f) for f in os.listdir(real)] + \
                              [os.path.join(fake, f) for f in os.listdir(fake)]
        self.labels = [0]*len(os.listdir(real)) + [1]*len(os.listdir(fake))
        self.transform = transform

    def __len__(self):
        return len(self.full_filenames)

    def __getitem__(self, idx):
        image = Image.open(self.full_filenames[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx]

# === Model ===
class CNN(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False, dropout=0.3):
        super(CNN, self).__init__()
        self.vit = create_model('vit_large_patch16_224', pretrained=pretrained, num_classes=0)
        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        feat = self.vit(x)
        return self.fusion(feat)

# === Train Function ===
def train():
    data_root = "data"
    metrics_log = {
        "train_loss": [], "train_acc": [], "train_prec": [], "train_rec": [], "train_f1": [],
        "val_acc": [], "val_prec": [], "val_rec": [], "val_f1": []
    }

    split_train_to_val(os.path.join(data_root, "train"), val_ratio=0.15)
    train_dataset = data_loader(os.path.join(data_root, "train"), transform=train_transform)
    val_dataset = data_loader(os.path.join(data_root, "val"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)

    model = CNN(pretrained=True, freeze_backbone=False).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(total_epochs):
        model.train()
        total_loss, total_correct, total = 0, 0, 0
        train_preds, train_targets = [], []
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch+1}/{total_epochs}]")

        for imgs, labels in loop:
            imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

            train_preds.extend(outputs.argmax(1).detach().cpu().tolist())
            train_targets.extend(labels.detach().cpu().tolist())

        train_loss = total_loss / total
        train_acc = accuracy_score(train_targets, train_preds)
        train_prec, train_rec, train_f1, _ = precision_recall_fscore_support(
            train_targets, train_preds, average='macro', zero_division=0
        )

        # === Validation ===
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(1)
                val_preds.extend(preds.detach().cpu().tolist())
                val_targets.extend(labels.detach().cpu().tolist())

        val_acc = accuracy_score(val_targets, val_preds)
        val_prec, val_rec, val_f1, _ = precision_recall_fscore_support(
            val_targets, val_preds, average='macro', zero_division=0
        )

        metrics_log["train_loss"].append(train_loss)
        metrics_log["train_acc"].append(train_acc)
        metrics_log["train_prec"].append(train_prec)
        metrics_log["train_rec"].append(train_rec)
        metrics_log["train_f1"].append(train_f1)
        metrics_log["val_acc"].append(val_acc)
        metrics_log["val_prec"].append(val_prec)
        metrics_log["val_rec"].append(val_rec)
        metrics_log["val_f1"].append(val_f1)

        print(f"📊 Epoch {epoch+1}/{total_epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | Prec: {train_prec:.4f} | Rec: {train_rec:.4f} | F1: {train_f1:.4f} || "
              f"Val Acc: {val_acc:.4f} | Prec: {val_prec:.4f} | Rec: {val_rec:.4f} | F1: {val_f1:.4f}")

    # === Save model and logs ===
    torch.save(model.state_dict(), "model.pth")
    print("✅ Model saved to model.pth")

    os.makedirs("logs", exist_ok=True)
    csv_path = "logs/vit_all_metrics_15e.csv"
    json_path = "logs/vit_all_metrics_15e.json"

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Epoch", "Train_Loss", "Train_Acc", "Train_Prec", "Train_Rec", "Train_F1",
            "Val_Acc", "Val_Prec", "Val_Rec", "Val_F1"
        ])
        for i in range(total_epochs):
            writer.writerow([
                i + 1,
                f"{metrics_log['train_loss'][i]:.4f}",
                f"{metrics_log['train_acc'][i]:.4f}",
                f"{metrics_log['train_prec'][i]:.4f}",
                f"{metrics_log['train_rec'][i]:.4f}",
                f"{metrics_log['train_f1'][i]:.4f}",
                f"{metrics_log['val_acc'][i]:.4f}",
                f"{metrics_log['val_prec'][i]:.4f}",
                f"{metrics_log['val_rec'][i]:.4f}",
                f"{metrics_log['val_f1'][i]:.4f}"
            ])

    with open(json_path, 'w') as f:
        json.dump(metrics_log, f, indent=2)

    print(f"📁 Metrics saved to {csv_path} and {json_path}")

# === Entry Point ===
if __name__ == "__main__":
    train()