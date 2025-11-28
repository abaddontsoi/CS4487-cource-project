import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm 
from dataloader import data_loader
from model import Model

device = "cuda" if torch.cuda.is_available() else "cpu"

# ===============================
# Train-Validate
# ===============================
def main():
    data_root = "data"
    batch_size = 32
    epochs = 10
    lr = 1e-4

    train_dataset = data_loader(os.path.join(data_root, "train"))
    val_dataset = data_loader(os.path.join(data_root, "val"))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = Model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        # ---- Train ----
        model.train()
        total_loss, total_correct, total = 0, 0, 0
        loop = tqdm(train_loader, total=len(train_loader), desc=f"Epoch [{epoch+1}/{epochs}]")

        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            total_correct += (outputs.argmax(1) == labels).sum().item()
            total += imgs.size(0)

        train_loss = total_loss / total
        train_acc = total_correct / total

        # ---- Validate ----
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

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    # Save Model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved")

if __name__ == "__main__":
    main()