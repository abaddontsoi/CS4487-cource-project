import os
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

# ===============================
# Self-Defined Dataloader
# ===============================
class data_loader(Dataset):
    def __init__(self, data_dir):

        real = os.path.join(data_dir, '0_real')
        fake = os.path.join(data_dir, '1_fake')

        file_names_real = os.listdir(real)
        file_names_fake = os.listdir(fake)

        self.full_filenames_real = [os.path.join(real, f) for f in file_names_real]
        self.full_filenames_fake = [os.path.join(fake, f) for f in file_names_fake]
        self.full_filenames = self.full_filenames_real + self.full_filenames_fake

        self.labels_real = [0 for _ in file_names_real]
        self.labels_fake = [1 for _ in file_names_fake]
        self.labels = self.labels_real + self.labels_fake

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=(64, 64)),
            transforms.ToTensor(),
        ])
        
    def __len__(self):
        return len(self.full_filenames)

    def __getitem__(self, idx):
        image = Image.open(self.full_filenames[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label
