import joblib
import numpy as np
import torch

import config
import engine
from model import EntityModel
from dataset import EntityDataset

if __name__ == '__main__':

    meta_data = joblib.load('encoding.bin')
    enc_pos = meta_data['enc_pos']
    enc_tag = meta_data['enc_tag']

    num_pos = len(list(enc_pos.classes_))
    num_tag = len(list(enc_tag.classes_))

    text = "thousands of protestors have marched through london to protest the war in iraq and demand the withdrawl of british troops from that country."
    token_ids = config.TOKENIZER.encode(text)

    text = text.split()
    print(text, token_ids)

    test_dataset = EntityDataset(texts=[text], pos=[[0]*len(text)], tags=[[0]*len(text)])

    device = config.device
    model = EntityModel(num_tag, num_pos)
    model.state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        for k,v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        pos, tag, _ = model(**data)

        print(enc_pos.inverse_transform(
            pos.argmax(2).cpu().numpy().reshape(-1))[:len(token_ids)])

        print(enc_tag.inverse_transform(
            tag.argmax(2).cpu().numpy().reshape(-1))[:len(token_ids)])

