import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
import rasterio
import matplotlib.pyplot as plt
import random
import sentinel_loader as sl
import numpy as np


class MultiSpectralDataset(Dataset):
    def __init__(self, dataset_path, val=False):
        self.datapoints = []
        classes = os.listdir(dataset_path)
        self.cls_to_int = {label: idx for idx, label in enumerate(set(classes))}
        self.int_to_class = {v: k for k, v in self.cls_to_int.items()}
        
        for cls in classes:
            cls_int = self.cls_to_int[cls]
            cls_imgs = os.listdir(os.path.join(dataset_path, cls))
            
            if val:
                cls_datapoints = [(os.path.join(dataset_path, cls, img), cls_int) for img in cls_imgs[2400:]]
            else:
                cls_datapoints = [(os.path.join(dataset_path, cls, img), cls_int) for img in cls_imgs[:2400]]
            
            self.datapoints += cls_datapoints
        print("Loaded", len(self.datapoints), "datapoints.")

    def __getitem__(self, idx):
        img_path, cls_int = self.datapoints[idx]
        
        # Чтение многоканального GeoTIFF
        with rasterio.open(img_path) as src:
            img = src.read()  # shape: [channels, H, W]
        
        img = torch.tensor(img, dtype=torch.float32)
        
        # Нормализация min-max по каждому каналу
        c_min = img.view(img.shape[0], -1).min(dim=1)[0].unsqueeze(1).unsqueeze(2)
        c_max = img.view(img.shape[0], -1).max(dim=1)[0].unsqueeze(1).unsqueeze(2)
        img = (img - c_min) / (c_max - c_min + 1e-6)
        
        target = torch.tensor(cls_int)
        return img, target

    def __len__(self):
        return len(self.datapoints)


# Загрузка обученной модели
# Настройки эксперимента
# =========================
device = torch.device('cuda')
num_epochs = 3
criterion = nn.CrossEntropyLoss()
val_step = 1
save_path = 'weights/'
os.makedirs(save_path, exist_ok=True)

# =========================
# Инициализация датасетов
# =========================
train_dataset = MultiSpectralDataset('EuroSATallBands/')
val_dataset = MultiSpectralDataset('EuroSATallBands/', val=True)

model = resnet18(num_classes=len(train_dataset.cls_to_int))
model.conv1 = nn.Conv2d(in_channels=13, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
state_dict = torch.load("weights/3_epoch_multispectral_model.pth", map_location=device, weights_only=True)
model.load_state_dict(state_dict)

model.eval().to(device)


# 1. Загрузка данных
multispectral, profile = sl.load_sentinel_data("sentinel_data")
print("Форма массива:", multispectral.shape)  # (13, H, W)

# 2. Нормализация min-max
c_min = multispectral.reshape(multispectral.shape[0], -1).min(axis=1)[:, None, None]
c_max = multispectral.reshape(multispectral.shape[0], -1).max(axis=1)[:, None, None]
multispectral = (multispectral - c_min) / (c_max - c_min + 1e-6)

# 3. Модель
num_classes = 10  # EuroSAT
model = resnet18(num_classes=num_classes)
model.conv1 = nn.Conv2d(in_channels=13, out_channels=64,
                        kernel_size=7, stride=2, padding=3, bias=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("weights/3_epoch_multispectral_model.pth", map_location=device))
model.eval().to(device)

# 4. Классификация тайлами
H, W = multispectral.shape[1], multispectral.shape[2]
tile_size = 64
stride = 64
out_map = np.zeros((H, W), dtype=np.uint8)

for y in range(0, H-tile_size+1, stride):
    for x in range(0, W-tile_size+1, stride):
        patch = multispectral[:, y:y+tile_size, x:x+tile_size]
        inp = torch.tensor(patch, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp)
            pred_class = torch.argmax(logits, dim=1).item()
        out_map[y:y+tile_size, x:x+tile_size] = pred_class

# 5. Визуализация
colormap = {
    0: (255, 255, 0),    # AnnualCrop
    1: (0, 100, 0),      # Forest
    2: (144, 238, 144),  # HerbaceousVegetation
    3: (255, 69, 0),     # Highway
    4: (128, 128, 128),  # Industrial
    5: (0, 255, 0),      # Pasture
    6: (34, 139, 34),    # PermanentCrop
    7: (220, 20, 60),    # Residential
    8: (30, 144, 255),   # River
    9: (0, 191, 255),    # SeaLake
}

rgb_map = np.zeros((H, W, 3), dtype=np.uint8)
for cls, color in colormap.items():
    mask = out_map == cls
    rgb_map[mask] = color

plt.figure(figsize=(10, 10))
plt.imshow(rgb_map)
plt.title("Карта классификации EuroSAT")
plt.axis("off")
plt.show()

# 6. Сохранение GeoTIFF
profile.update(dtype=rasterio.uint8, count=1)
with rasterio.open("classification_map.tif", "w", **profile) as dst:
    dst.write(out_map.astype(np.uint8), 1)
    dst.write_colormap(1, colormap)

