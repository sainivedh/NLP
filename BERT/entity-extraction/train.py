import numpy as np
import IPython
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

import config
import dataset
import engine
from model import EntityModel


def process_data(data_path):
    df = pd.read_csv(data_path, encoding='latin-1')
    IPython.embed()
    exit(1)
    df = df.sample(50000)
    df['Sentence #'] = df['Sentence #'].fillna(method='ffill')

    enc_pos = LabelEncoder()
    enc_tag = LabelEncoder()

    df.loc[:, 'POS'] = enc_pos.fit_transform(df['POS'])
    df.loc[:, 'Tag'] = enc_tag.fit_transform(df['Tag'])

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    sentences = df.groupby('Sentence #')['Word'].apply(list).values
    pos = df.groupby('Sentence #')['POS'].apply(list).values
    tags = df.groupby('Sentence #')['Tag'].apply(list).values

    return sentences, pos, tags, enc_pos, enc_tag, num_pos, num_tag

if __name__ == '__main__':
    sentences, pos, tags, enc_pos, enc_tag, num_pos, num_tag = process_data(config.TRAINING_FILE)

    meta_data = {'enc_pos': enc_pos, 'enc_tag': enc_tag}
    joblib.dump(meta_data, 'encoding.bin')

    (
        train_sentences, test_sentences,
        train_pos, test_pos,
        train_tag, test_tag
    ) = train_test_split(sentences, pos, tags, test_size=0.1, random_state=56)

    train_dataset = dataset.EntityDataset(train_sentences, train_pos, train_tag)
    valid_dataset = dataset.EntityDataset(test_sentences, test_pos, test_tag)

    train_data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE)

    valid_data_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE)

    device = config.device
    model = EntityModel(num_tag, num_pos)
    model = model.to(device)

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


    num_steps = int(len(sentences) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_steps
    )

    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        print(f'Epoch {epoch}')
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        valid_loss = engine.eval_fn(valid_data_loader, model, device)
        print(f'Train Loss: {train_loss} Valid Loss: {valid_loss}')
        if valid_loss < best_loss:
            torch.save(model.state_dict(), 'Entity_model.bin')
            best_loss = valid_loss






