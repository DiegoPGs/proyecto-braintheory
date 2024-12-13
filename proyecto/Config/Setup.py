"""
HCP Task Analysis and Neural Network Modeling
==========================================

Esta libreta analiza datos de fMRI del Human Connectome Project (HCP) para predecir
estados mentales y comportamientos usando redes neuronales.

Requisitos:
-----------
- Python 3.7+
- PyTorch
- Nilearn
- Pandas
- NumPy
- Seaborn
- Matplotlib

Autores: Ari, Ignacio, y Alejandro 👾
"""

# ==== Importaciones Básicas ====
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union

# ==== Importaciones de PyTorch ====
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ==== Configuración de Visualización ====
plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/nma.mplstyle")
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

# ==== Configuración Global ====
# Constantes para el dataset HCP
CONSTANTS = {
    'N_SUBJECTS': 100,
    'N_PARCELS': 360,
    'TR': 0.72,  # Resolución temporal en segundos
    'HEMIS': ["Right", "Left"],
    'RUNS': ['LR', 'RL'],
    'N_RUNS': 2,
}

# Definición de experimentos y condiciones
EXPERIMENTS = {
    'MOTOR': {
        'cond': ['lf', 'rf', 'lh', 'rh', 't', 'cue'],
        'description': 'Movimientos de pie izquierdo, derecho, mano izquierda, derecha, lengua y señal'
    },
    'GAMBLING': {
        'cond': ['loss', 'win'],
        'description': 'Respuestas a pérdidas y ganancias en juegos de azar'
    },
    'EMOTION': {
        'cond': ['fear', 'neut'],
        'description': 'Respuestas a expresiones de miedo y neutrales'
    },
    # ... [otros experimentos]
}

# Configuración de semilla aleatoria para reproducibilidad
def set_seed(seed: int = 42) -> None:
    """
    Configura las semillas aleatorias para reproducibilidad.
    
    Args:
        seed (int): Valor de la semilla aleatoria
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
set_seed()

# Configuración de dispositivo para PyTorch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Usando dispositivo: {DEVICE}")

# Configuración de rutas
HCP_DIR = "./hcp_task"
if not os.path.exists(HCP_DIR):
    os.makedirs(HCP_DIR)

# TODO: Sustituir con el setup en la libreta de proyecto