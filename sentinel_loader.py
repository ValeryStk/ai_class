import os
import xml.etree.ElementTree as ET
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling

def load_sentinel_data(base_path="sentinel_data"):
    """
    Загружает 13 спектральных каналов Sentinel-2,
    приводит их к 20м и возвращает массив [13, H, W].
    """
    xml_path = os.path.join(base_path, "MTD_MSIL2A.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()

    needed_bands = ["B01","B02","B03","B04","B05","B06","B07",
                    "B08","B8A","B09","B10","B11","B12"]

    def find_band_file(band_name):
        for res in ["R20m", "R10m", "R60m"]:
            for elem in root.findall(".//IMAGE_FILE"):
                rel_path = elem.text + ".jp2"
                if res in rel_path and band_name in rel_path:
                    return os.path.join(base_path, rel_path)
        return None

    # эталон — B05 (20м)
    ref_file = find_band_file("B05")
    if ref_file is None:
        raise RuntimeError("Не найден эталонный файл B05")

    with rasterio.open(ref_file) as ref:
        ref_shape = (ref.height, ref.width)
        ref_transform = ref.transform
        ref_crs = ref.crs
        profile = ref.profile

    multispectral = []
    for band in needed_bands:
        f = find_band_file(band)
        if f is None:
            print(f"⚠️ Канал {band} отсутствует, добавляем пустой канал")
            data = np.zeros(ref_shape, dtype=np.float32)
            multispectral.append(data)
            continue

        with rasterio.open(f) as src:
            data = src.read(1).astype(np.float32)
            if src.shape != ref_shape:
                dst = np.empty(ref_shape, dtype=np.float32)
                reproject(
                    source=data,
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.bilinear
                )
                data = dst
            multispectral.append(data)

    multispectral = np.stack(multispectral, axis=0)
    return multispectral, profile
