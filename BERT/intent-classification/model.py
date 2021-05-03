import torch
from torch import nn
from config import BERT_MODEL_PATH, TOKENIZER_PATH
import transformers

class IntentModel(nn.Module):
    def __init__(self, n_classes):
        super(IntentModel, self).__init__()
        self.bert_model = transformers.BertModel.from_pretrained(
            BERT_MODEL_PATH
        )
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(768, n_classes)

    def forward(self, ids, mask, token_type_ids):
        bert_out = self.bert_model(ids, mask, token_type_ids)
        x = self.dropout(bert_out[1])
        x = self.out(x)
        return x
