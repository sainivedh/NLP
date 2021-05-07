import torch
from torch import nn
import config
from tqdm import tqdm
import utils


def criterion(o1, t1, o2, t2):
    loss_start = nn.CrossEntropyLoss()(o1, t1)
    loss_end = nn.CrossEntropyLoss()(o2, t2)

    return loss_start + loss_end



def train_fn(dataloader, model, optimizer, scheduler):
    model.train()
    losses = utils.AverageMeter()
    tk0 = tqdm(dataloader, total=len(dataloader))
    for data in tk0:
        ids = data['ids']
        mask = data['mask']
        token_type_ids = data['token_type_ids']
        start_target = data['targets_start']
        end_target = data['targets_end']

        #print(ids.shape, mask.shape, token_type_ids.shape)

        ids = ids.to(config.DEVICE, dtype=torch.long)
        mask = mask.to(config.DEVICE, dtype=torch.long)
        token_type_ids = token_type_ids.to(config.DEVICE, dtype=torch.long)
        start_target = start_target.to(config.DEVICE, dtype=torch.float)
        end_target = end_target.to(config.DEVICE, dtype=torch.float)

        optimizer.zero_grad()
        start_out, end_out = model(ids, mask, token_type_ids)
        loss = criterion(start_out, start_target, end_out, end_target)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)
        break



