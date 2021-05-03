import torch
from torch import nn
from tqdm import tqdm

def criterion(outputs, targets):
    return nn.CrossEntropyLoss()(outputs, targets.flatten())



def train_fn(dataloader, model, optimizer, scheduler, device, n_samples):
    model.train()
    total_loss = 0

    for data in tqdm(dataloader, total=len(dataloader)):
        ids = data['ids']
        mask = data['mask']
        token_type_ids = data['token_type_ids']
        targets = data['targets']

        ids = ids.to(device)
        mask = mask.to(device)
        token_type_ids = token_type_ids.to(device)
        targets = targets.to(device, dtype=torch.long)


        optimizer.zero_grad()
        output = model(ids, mask, token_type_ids)
        #print(output.dtype, targets.dtype)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()


    return total_loss / n_samples

def valid_fn(dataloader, model, device, n_samples):
    model.eval()
    m = nn.Softmax(-1)
    total_loss = 0
    fin_outputs = []
    fin_targets = []

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            ids = data['ids']
            mask = data['mask']
            token_type_ids = data['token_type_ids']
            targets = data['targets']

            ids = ids.to(device)
            mask = mask.to(device)
            token_type_ids = token_type_ids.to(device)
            targets = targets.to(device, dtype=torch.long)

            output = model(ids, mask, token_type_ids)
            #print(output.shape)
            loss = criterion(output, targets)

            total_loss += loss.item()
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(m(output).argmax(-1).view(-1,1).cpu().detach().numpy().tolist())

    return fin_outputs, fin_targets, total_loss/n_samples



