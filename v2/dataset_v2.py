"""
dataset_v2.py
Carga el dataset balanceado para entrenamiento.
CORRECCIÓN v2:
- Eliminado RandomHorizontalFlip (neuroanatómicamente inválido en RM cerebral)
- Eliminado RandomRotation del transform_train (ya está en disco desde preparacion_DS.py)
- Transform_train solo normaliza — el augmentation ya está en las imágenes en disco
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from collections import Counter

DATASET_DIR = "/home/davfy/Escritorio/Vision/dataset_balanceado"
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# ── Transformaciones ───────────────────────────────────────
# NOTA: NO se aplica augmentation en RAM porque ya está en disco
# desde preparacion_DS.py. Solo se normaliza.
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

transform_val_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

def get_dataloaders(batch_size=32, num_workers=4):
    train_ds = datasets.ImageFolder(os.path.join(DATASET_DIR, "train"), transform_train)
    val_ds   = datasets.ImageFolder(os.path.join(DATASET_DIR, "val"),   transform_val_test)
    test_ds  = datasets.ImageFolder(os.path.join(DATASET_DIR, "test"),  transform_val_test)

    # Orden médico real (PyTorch lee alfabético por defecto)
    logica_medica = {
        "Non Demented":      0,
        "Very mild Dementia": 1,
        "Mild Dementia":      2,
        "Moderate Dementia":  3,
    }

    def remap(ds):
        m = {i: logica_medica[c] for i, c in enumerate(ds.classes)}
        ds.targets = [m[t] for t in ds.targets]
        ds.samples = [(p, m[l]) for p, l in ds.samples]
        ds.class_to_idx = logica_medica

    for ds in [train_ds, val_ds, test_ds]:
        remap(ds)

    # WeightedRandomSampler — balancea los batches de train
    targets = np.array(train_ds.targets)
    counts  = np.bincount(targets)
    counts  = np.where(counts == 0, 1, counts)
    w       = 1.0 / counts
    sw      = torch.tensor([w[t] for t in targets], dtype=torch.double)
    sampler = WeightedRandomSampler(sw, len(sw))

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    clases_ordenadas = ["Non Demented", "Very mild Dementia", "Mild Dementia", "Moderate Dementia"]
    return train_loader, val_loader, test_loader, clases_ordenadas