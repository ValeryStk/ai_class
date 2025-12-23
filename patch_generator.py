from eolearn.core import EOPatch, FeatureType
from eolearn.io import SentinelHubInputTask
from eolearn.features import NormalizedDifferenceIndexTask
from sentinelhub import BBox, CRS, DataCollection
import numpy as np
import os

# 1. Область интереса (пример: Минск)
bbox = BBox([27.55, 53.85, 27.65, 53.95], crs=CRS.WGS84)

# 2. Задача загрузки Sentinel-2
input_task = SentinelHubInputTask(
    data_collection=DataCollection.SENTINEL2_L2A,
    bands_feature=(FeatureType.DATA, 'BANDS'),
    bands=['B03','B04','B08','B11'],  # GREEN, RED, NIR, SWIR
    resolution=20,
    maxcc=0.2
)

# 3. Задачи вычисления индексов (через список индексов каналов)
ndvi_task = NormalizedDifferenceIndexTask(
    (FeatureType.DATA, 'BANDS'), (FeatureType.DATA, 'NDVI'),
    [2, 1]   # NIR (B08), RED (B04)
)

ndwi_task = NormalizedDifferenceIndexTask(
    (FeatureType.DATA, 'BANDS'), (FeatureType.DATA, 'NDWI'),
    [0, 2]   # GREEN (B03), NIR (B08)
)

ndbi_task = NormalizedDifferenceIndexTask(
    (FeatureType.DATA, 'BANDS'), (FeatureType.DATA, 'NDBI'),
    [3, 2]   # SWIR (B11), NIR (B08)
)

# 4. Выполнение пайплайна
eopatch = EOPatch()
eopatch = input_task.execute(eopatch, bbox=bbox, time_interval=('2025-06-01', '2025-06-30'))
eopatch = ndvi_task.execute(eopatch)
eopatch = ndwi_task.execute(eopatch)
eopatch = ndbi_task.execute(eopatch)

# 5. Формирование мультиклассовой маски
ndvi = eopatch.data['NDVI'][0]
ndwi = eopatch.data['NDWI'][0]
ndbi = eopatch.data['NDBI'][0]

mask = np.zeros_like(ndvi, dtype=np.int64)
mask[ndvi > 0.3] = 1   # растительность
mask[ndwi > 0.2] = 2   # вода
mask[ndbi > 0.1] = 3   # урбан

# 6. Нарезка на патчи
patch_size = 64
images, masks = [], []

H, W, C = eopatch.data['BANDS'][0].shape
for i in range(0, H - patch_size, patch_size):
    for j in range(0, W - patch_size, patch_size):
        img_patch = eopatch.data['BANDS'][0][i:i+patch_size, j:j+patch_size, :]
        mask_patch = mask[i:i+patch_size, j:j+patch_size]

        # нормализация
        img_patch = (img_patch - img_patch.mean()) / (img_patch.std() + 1e-6)

        images.append(img_patch.transpose(2,0,1))  # [C,H,W]
        masks.append(mask_patch)

images = np.stack(images, axis=0)
masks = np.stack(masks, axis=0)

# 7. Сохранение датасета
os.makedirs("dataset", exist_ok=True)
np.savez_compressed("dataset/sentinel_multiindex_dataset.npz", images=images, masks=masks)

print(f"✅ Датасет сохранён: images {images.shape}, masks {masks.shape}")
