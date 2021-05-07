import torch
import pandas as pd
import numpy as np
from torch import nn
from sklearn.model_selection import train_test_split
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
from model import BioASQModel
from dataset import BioASQDataset
import engine

def run():
    df = pd.read_csv(config.DATA_FILE).dropna().reset_index(drop=True)
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    train_dataset = BioASQDataset(df_train['context'].values, df_train['question'].values,
                                  df_train['answer'].values, df_train['answer_start_id'].values)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE)

    valid_dataset = BioASQDataset(df_test['context'].values, df_test['question'].values,
                                  df_test['answer'], df_test['answer_start_id'].values)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE)

    model = BioASQModel()
    model.to(config.DEVICE)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int((len(df_train)/config.TRAIN_BATCH_SIZE) * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=num_train_steps)

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_dataloader, model, optimizer, scheduler)
        #outputs, targets = engine.valid_fn(valid_dataloader, model)


if __name__ == '__main__':
    run()
