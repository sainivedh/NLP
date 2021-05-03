import torch
import config
from model import IntentModel
from sklearn.preprocessing import LabelEncoder
import joblib
from torch import nn

m = nn.Softmax(-1)
model = IntentModel(n_classes=7)
model.load_state_dict(torch.load(config.MODEL_PATH))
model.to(config.DEVICE)
le = LabelEncoder()
le = joblib.load('labelencoder.bin')
print(le.classes_)
text = "Rate this book as awful"
input = config.TOKENIZER(text, None, add_special_tokens=True, max_length=config.MAX_LENGTH, pad_to_max_length=True)

ids = torch.tensor(input['input_ids']).unsqueeze(0).to(config.DEVICE)
mask = torch.tensor(input['attention_mask']).unsqueeze(0).to(config.DEVICE)
token_type_ids = torch.tensor(input['token_type_ids']).unsqueeze(0).to(config.DEVICE)


outputs = model(ids, mask, token_type_ids)

category = m(outputs[0]).argmax()
print(le.classes_[category])