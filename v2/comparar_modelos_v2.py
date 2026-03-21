"""
run_all_v2.py
Script maestro v2 — entrena los 3 modelos corregidos uno por uno.
El dataset ya existe, no se regenera.
"""

import os, sys, time, subprocess

BASE    = "/home/davfy/Escritorio/Vision"
V2_DIR  = os.path.join(BASE, "v2")

def log(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}\n")

def correr(script):
    subprocess.run([sys.executable, os.path.join(V2_DIR, script)], check=True)

t_total = time.time()

log("PASO 1: Entrenando EfficientNetV2S + CBAM v2")
correr("train_efficientnet_v2.py")

log("PASO 2: Entrenando MobileNetV3 + CBAM v2")
correr("train_mobilenet_v2.py")

log("PASO 3: Entrenando ResNet50 + CBAM v2")
correr("train_resnet_v2.py")

log("PASO 4: Generando comparativa v2")
correr("comparar_modelos_v2.py")

mins = (time.time() - t_total) / 60
log(f"TODO LISTO en {mins:.1f} minutos")
print(f"Resultados en: {V2_DIR}/resultados/")