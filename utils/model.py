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
from pytorch_lightning.callbacks import ModelCheckpoint
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

BERT_MODEL_NAME = 'bert-base-cased'

# Modelo Lightning actualizado para clasificación binaria de sentimientos IMDB
class SentimentClassifier(pl.LightningModule):
    def __init__(self, n_training_steps=None, n_warmup_steps=None):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        # Cambio principal: clasificación binaria (1 clase en lugar de 6)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.n_training_steps = n_training_steps
        self.n_warmup_steps = n_warmup_steps
        # Usar Binary Cross Entropy para clasificación binaria
        self.criterion = nn.BCEWithLogitsLoss()
        # Métricas para clasificación binaria
        self.train_auroc = AUROC(task="binary")
        self.val_auroc = AUROC(task="binary")

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.bert(input_ids, attention_mask=attention_mask)
        output = self.classifier(output.pooler_output)
        loss = 0
        if labels is not None:
            loss = self.criterion(output.squeeze(), labels.squeeze())
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        
        # Calcular métricas
        predictions = torch.sigmoid(outputs.squeeze())
        self.train_auroc(predictions, labels.squeeze())
        
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("train/auroc", self.train_auroc, on_step=False, on_epoch=True)
        return {"loss": loss, "predictions": predictions, "labels": labels}

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        
        # Calcular métricas
        predictions = torch.sigmoid(outputs.squeeze())
        self.val_auroc(predictions, labels.squeeze())
        
        self.log("val/loss", loss, prog_bar=True, logger=True, on_epoch=True)
        self.log("val/auroc", self.val_auroc, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test/loss", loss, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=2e-5)
        if self.n_training_steps:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.n_warmup_steps,
                num_training_steps=self.n_training_steps
            )
            return dict(
                optimizer=optimizer,
                lr_scheduler=dict(
                    scheduler=scheduler,
                    interval='step'
                )
            )
        return dict(optimizer=optimizer)