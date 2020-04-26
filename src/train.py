import torch
from tqdm import tqdm


def train_fn(dataloader, model, criterion, optimizer, device, scheduler=None, accumulation_steps=None):
    model.train()
    epoch_loss = []

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        ids = data['ids'].to(device)
        mask = data['mask'].to(device)
        token_type_ids = data['token_type_ids'].to(device)
        targets = data['targets'].to(device)

        model.zero_grad()
        output = model(ids)
        #print(output.size(), targets.size())
        loss = criterion(output, targets)
        epoch_loss.append(loss.item())
        loss.backward()
        optimizer.step()

    return sum(epoch_loss)/len(epoch_loss)


def eval_fn(dataloader, model, criterion, device):
    model.eval()
    epoch_loss = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            ids = data['ids'].to(device)
            mask = data['mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            targets = data['targets'].to(device)

            output = model(ids)
            loss = criterion(output, targets)
            epoch_loss.append(loss.item())
        
        return sum(epoch_loss)/len(epoch_loss)