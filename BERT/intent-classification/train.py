import torch
import config
import engine
import dataset
from model import IntentModel
from tqdm import tqdm
from torch import nn

import numpy as np
import pandas as pd
from sklearn import metrics
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.preprocessing import LabelEncoder

def run():
    train_df = pd.read_csv(config.TRAIN_DATA)
    valid_df = pd.read_csv(config.VALID_DATA)
    le = LabelEncoder()
    #le.fit_transform()
    train_df['targets'] = le.fit_transform(train_df['intent'])
    valid_df['targets'] = le.transform(valid_df['intent'])
    n_classes = len(train_df['targets'].unique())

    train_dataset = dataset.IntentDataset(train_df['text'].values, train_df['targets'].values)
    train_dataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE)

    valid_dataset = dataset.IntentDataset(valid_df['text'].values, valid_df['targets'].values)
    valid_dataLoader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE)

    model = IntentModel(n_classes)
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

    num_train_steps = int((len(train_df)/ config.TRAIN_BATCH_SIZE) * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        loss = engine.train_fn(train_dataLoader, model, optimizer, scheduler,
                               config.DEVICE, len(train_dataLoader))
        print('Epoch: {} Train Loss{}'.format(epoch, loss))

        outputs, targets, valid_loss = engine.valid_fn(valid_dataLoader, model,
                                                       config.DEVICE, len(valid_dataLoader))

        accuracy = metrics.accuracy_score(targets, outputs)
        print('Valid Accuracy: ', accuracy)


        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy

if __name__ == '__main__':
    run()
