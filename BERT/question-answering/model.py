import config
import torch
from torch import nn


class BioASQModel(nn.Module):
    def __init__(self):
        super(BioASQModel, self).__init__()
        self.bert_model = config.BERT_MODEL
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(768, 2)

    def forward(self, ids, mask, token_type_ids):
        x = self.bert_model(ids, mask, token_type_ids)
        x = self.dropout(x[0])
        x = self.out(x[0])
        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

model = BioASQModel()
ids = torch.randint(1, 10000, (4,512), dtype=torch.long)
mask = torch.ones((4,512), dtype=torch.long)
token_type_ids = torch.zeros((4,512), dtype=torch.long)

_, _ = model(ids, mask, token_type_ids)
