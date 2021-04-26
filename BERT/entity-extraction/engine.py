import torch
from tqdm import tqdm

def train_fn(dataloader, model, optimizer, device, scheduler):
    model.train()
    total_loss = 0

    for data in tqdm(dataloader, total=len(dataloader)):
        for k, v in data.items():
            data[k] = v.to(device)
        optimizer.zero_grad()
        _ , _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def eval_fn(dataloader, model, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            for k, v in data.items():
                data[k] = v.to(device)
            _ , _, loss = model(**data)
            total_loss += loss.item()
    return total_loss / len(dataloader)