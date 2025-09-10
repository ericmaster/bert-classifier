import pandas as pd
import numpy as np

from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import random_split, Dataset, DataLoader
from torch.optim import AdamW

from transformers import BertTokenizerFast as BertTokenizer, BertModel, get_linear_schedule_with_warmup

import pytorch_lightning as pl
import torchmetrics
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import TQDMProgressBar, RichProgressBar

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, multilabel_confusion_matrix

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from torchmetrics import AUROC, Accuracy

#nuevas librerias
import kagglehub
from kagglehub import KaggleDatasetAdapter
import os
import glob
import zipfile
import tempfile

# Configuración de entrenamiento actualizada para clasificación binaria IMDB
N_EPOCHS = 10  # Reducido para pruebas más rápidas
BATCH_SIZE = 32
# NUM_WORKERS = 4

# ============ Para compatibilidad con VSCode en ambientes basados en colab ===========

from google.colab import userdata

# Mock userdata.get to return None immediately
_original_get = userdata.get
userdata.get = lambda key: None  # Skip all secret prompts

# ============ Fin de compatibilidad con VSCode en ambientes basados en colab ===========

# Descargar la última versión
dataset_path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
print("Dataset path:", dataset_path)

# Si es un ZIP, extraer a directorio temporal
if os.path.isfile(dataset_path) and dataset_path.lower().endswith(".zip"):
    extract_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(dataset_path, "r") as z:
        z.extractall(extract_dir)
    search_dir = extract_dir
else:
    search_dir = dataset_path

# Buscar archivos CSV
csv_files = glob.glob(os.path.join(search_dir, "**", "*.csv"), recursive=True)
if not csv_files:
    raise FileNotFoundError(f"No CSV files found under {search_dir}")

csv_file = csv_files[0]
print("Using CSV:", csv_file)

# Leer con pandas (fallback de encoding si falla)
try:
    df = pd.read_csv(csv_file)
except UnicodeDecodeError:
    df = pd.read_csv(csv_file, encoding="latin1")

print(df.shape)
df.head()

df['is_offensive'] = df['sentiment'].map({'negative': 1, 'positive': 0})

# Renombrar columnas para compatibilidad con el código existente
df = df.rename(columns={'review': 'comment_text'})

# Crear columna de etiquetas numéricas para el modelo
df['label'] = df['is_offensive']

# Filtra por sentimiento
df_origin = df[df['is_offensive'] == 1]  # Negativos (tratados como ofensivos)
df_clean = df[df['is_offensive'] == 0]   # Positivos (tratados como no ofensivos)

filter_df = pd.concat([
    df_origin,      # Todas las reseñas negativas (25,000)
    df_clean        # Todas las reseñas positivas (25,000)
])

train_frac, val_frac, test_frac = 0.7, 0.15, 0.15

# Shuffle the DataFrame
filter_df = filter_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Calcular los indices para separar los subsets
train_end = int(train_frac * len(filter_df))
val_end = train_end + int(val_frac * len(filter_df))

# Split the DataFrame
train_df = filter_df[:train_end]
val_df = filter_df[train_end:val_end]
test_df = filter_df[val_end:]

print("Train DataFrame:\n", len(train_df))
print("Validation DataFrame:\n", len(val_df))
print("Test DataFrame:\n", len(test_df))

BERT_MODEL_NAME = 'bert-base-cased'
tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)

from utils.data import SentimentDataModule

MAX_TOKEN_COUNT = 512

# Crear DataModule
data_module = SentimentDataModule(
    train_df,
    val_df, 
    test_df,
    tokenizer,
    batch_size=BATCH_SIZE,
    max_token_len=MAX_TOKEN_COUNT
)

# Calcular pasos de entrenamiento
steps_per_epoch = len(train_df) // BATCH_SIZE
total_training_steps = steps_per_epoch * N_EPOCHS
warmup_steps = total_training_steps // 5

print(f"Steps per epoch: {steps_per_epoch}")
print(f"Total training steps: {total_training_steps}")
print(f"Warmup steps: {warmup_steps}")

from utils.model import SentimentClassifier

# Crear modelo
model = SentimentClassifier(
    n_warmup_steps=warmup_steps,
    n_training_steps=total_training_steps
)

print("Modelo y DataModule creados exitosamente para clasificación de sentimientos IMDB")

# Configurar Trainer de Lightning
checkpoint_callback = ModelCheckpoint(
    dirpath="sentiment_checkpoints",
    filename="best-checkpoint-{epoch:02d}-{val/loss:.2f}",
    save_top_k=1,
    verbose=True,
    monitor="val/loss",
    mode="min"
)

logger = CSVLogger("sentiment_logs", name="imdb_sentiment")

early_stopping_callback = EarlyStopping(
    monitor="val/loss",
    patience=3,
    verbose=True,
    mode="min"
)

trainer = pl.Trainer(
    logger=logger,
    callbacks=[early_stopping_callback, checkpoint_callback],
    max_epochs=N_EPOCHS,
    accelerator="auto",  # Usa GPU si está disponible
    devices="auto",
    log_every_n_steps=10,
    deterministic=True
)

print("Trainer configurado para clasificación de sentimientos IMDB")

# Entrenar el modelo
print("Iniciando entrenamiento del modelo de clasificación de sentimientos...")
trainer.fit(model, data_module)
print("Entrenamiento completado!")
