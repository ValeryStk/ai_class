import os
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader
import rasterio
import matplotlib.pyplot as plt
import random

# =========================
# Класс для мультиспектральных данных
# =========================
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

# =========================
# Настройки эксперимента
# =========================
device = torch.device('cuda')
num_epochs = 6
criterion = nn.CrossEntropyLoss()
val_step = 1
save_path = 'weights/'
os.makedirs(save_path, exist_ok=True)

# =========================
# Инициализация датасетов
# =========================
train_dataset = MultiSpectralDataset('EuroSATallBands/')
val_dataset = MultiSpectralDataset('EuroSATallBands/', val=True)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# =========================
# Визуализация примеров
# =========================
shuffled_indices = torch.randperm(len(train_dataset)).tolist()
random_samples = [train_dataset[x] for x in shuffled_indices[:25]]

fig, axs = plt.subplots(5, 5, figsize=(15, 15))
for i, ax in enumerate(axs.flat):
    data, target = random_samples[i]
    
    # Для визуализации используем RGB каналы (Sentinel-2: B4=red, B3=green, B2=blue)
    if data.shape[0] >= 4:
        rgb = data[[3,2,1], :, :].cpu().numpy().transpose(1,2,0)
    else:
        # fallback если каналов меньше
        rgb = data[:3].cpu().numpy().transpose(1,2,0)
    
    ax.imshow(rgb)
    ax.set_title(train_dataset.int_to_class[int(target)], fontsize=10)
    ax.axis('off')

plt.show()

# =========================
# Пример модели
# =========================
# Если хотим использовать все каналы, нужно изменить первый слой ResNet.
# Например, ResNet18 с 13 каналами вместо 3:
from torchvision.models import resnet18

model = resnet18(num_classes=len(train_dataset.cls_to_int))
model.conv1 = nn.Conv2d(in_channels=train_dataset[0][0].shape[0],  # число каналов
                        out_channels=64,
                        kernel_size=7,
                        stride=2,
                        padding=3,
                        bias=False)

model = model.to(device)

# =========================
# Цикл обучения (упрощённый)
# =========================
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(num_epochs):
    model.train()
    for imgs, targets in train_dataloader:
        imgs, targets = imgs.to(device), targets.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    # Валидация
    if (epoch+1) % val_step == 0:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, targets in val_dataloader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()
        print(f"Validation accuracy: {100*correct/total:.2f}%")

# =========================
# Сохранение весов
# =========================
torch.save(model.state_dict(), os.path.join(save_path, "3_epoch_multispectral_model.pth"))
print("Model saved.")
