import os
import random
import shutil
import multiprocessing
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.autoaugment import RandAugment
from timm import create_model

# Setup
batch_size_frozen = 64
batch_size_unfrozen = 16  
torch.backends.cudnn.benchmark = True
device = "cuda" if torch.cuda.is_available() else "cpu"
num_workers = 12  # safe high-performance

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
            print(f"❌ Skipping missing class folder: {train_class_dir}")
            continue

        # Skip if already split
        if val_class_dir.exists() and any(val_class_dir.iterdir()):
            print(f"⚠️  Validation folder already exists and is not empty: {val_class_dir}. Skipping.")
            continue

        val_class_dir.mkdir(parents=True, exist_ok=True)

        # Get image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
        files = [f for f in train_class_dir.iterdir() if f.suffix.lower() in image_extensions and f.is_file()]
        if not files:
            print(f"⚠️  No images found in {train_class_dir}. Skipping.")
            continue

        num_to_copy = max(1, int(len(files) * val_ratio))
        files_to_copy = random.sample(files, num_to_copy)

        print(f"📂 Copying {num_to_copy}/{len(files)} images from {class_name} to validation")

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
    def __init__(self, pretrained=True, freeze_backbone=True, dropout=0.3):
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

    def unfreeze_backbone(self):
        for param in self.vit.parameters():
            param.requires_grad = True
        self.freeze_backbone = False

# === Train Function ===
def train():
    data_root = "data"
    total_epochs = 10
    freeze_epochs = 3  # Freeze ViT for first 3 epochs
    lr_frozen = 1e-4
    lr_unfrozen = 1e-5

    # === Data loading and transforms ===
    split_train_to_val(os.path.join(data_root, "train"), val_ratio=0.1)
    train_dataset = data_loader(os.path.join(data_root, "train"), transform=train_transform)
    val_dataset = data_loader(os.path.join(data_root, "val"), transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size_frozen, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size_frozen, shuffle=False,
                            num_workers=num_workers, pin_memory=False)

    # === Model ===
    model = CNN(pretrained=True, freeze_backbone=True).to(device)
    criterion = nn.CrossEntropyLoss()

    # 🔧 Only train fusion head at first
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_frozen)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(total_epochs):
        # 🔓 Unfreeze at specified epoch
        if epoch == freeze_epochs and model.freeze_backbone:
            print(f"🔓 Unfreezing ViT backbone at epoch {epoch+1}")
            model.unfreeze_backbone()
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_unfrozen)
            train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size_unfrozen,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=False
            )

        model.train()
        total_loss, total_correct, total = 0, 0, 0
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

        train_loss = total_loss / total
        train_acc = total_correct / total

        # === Validation ===
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(1)
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
        val_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{total_epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    torch.save(model.state_dict(), "model.pth")
    print("✅ Model saved to model.pth")

# === Entry Point ===
if __name__ == "__main__":
    train()