import pandas as pd

import torch
from torch.utils.data import random_split, Dataset, DataLoader

from transformers import BertTokenizerFast as BertTokenizer

import pytorch_lightning as pl

# Dataset class actualizada para clasificación binaria de sentimientos IMDB
class IMDBDataset(Dataset):
  def __init__(
    self,
    data: pd.DataFrame,
    tokenizer: BertTokenizer,
    max_token_len: int = 512
  ):
    self.tokenizer = tokenizer
    self.data = data
    self.max_token_len = max_token_len

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    comment_text = data_row.comment_text
    # Cambio principal: usar etiqueta binaria en lugar de multi-etiqueta
    label = data_row.label  # 0 para positive, 1 para negative

    encoding = self.tokenizer.encode_plus(
      comment_text,
      add_special_tokens=True,
      max_length=self.max_token_len,
      return_token_type_ids=False,
      padding="max_length",
      truncation=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return dict(
      comment_text=comment_text,
      input_ids=encoding["input_ids"].flatten(),
      attention_mask=encoding["attention_mask"].flatten(),
      labels=torch.FloatTensor([label])  # Solo una etiqueta binaria
    )

# DataModule actualizado para dataset IMDB
class SentimentDataModule(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, tokenizer, batch_size=8, max_token_len=512):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def setup(self, stage=None):
        # Usar IMDBDataset en lugar de ToxicCommentsDataset
        self.train_dataset = IMDBDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len
        )

        self.val_dataset = IMDBDataset(
            self.val_df,
            self.tokenizer,
            self.max_token_len
        )

        self.test_dataset = IMDBDataset(
            self.test_df,
            self.tokenizer,
            self.max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=2
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=2
        )