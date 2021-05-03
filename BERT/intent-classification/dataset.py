import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import IPython
from sklearn.preprocessing import LabelEncoder
import joblib
import config
'''
df = pd.read_csv('./Data/train.csv')
le = LabelEncoder()
#IPython.embed(); exit(1)
df['targets'] = le.fit_transform(df['intent'])
joblib.dump(le,'labelencoder.bin')
n_classes = len(df['intent'].unique())
print(df.columns)
'''

class IntentDataset:
    def __init__(self, texts, targets):
        self.texts = texts
        self.targets = targets
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LENGTH

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        text = " ".join(text.split())
        target = self.targets[item]
        inputs = self.tokenizer.encode_plus(text, None, add_special_tokens=True,
                                               max_length=self.max_len, pad_to_max_length=True)

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(target, dtype=torch.float)
        }


#train_dataLoader = torch.utils.data.data