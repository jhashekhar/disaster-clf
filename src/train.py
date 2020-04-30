import torch
from tqdm import tqdm


def train(
        dataloader,
        model,
        criterion,
        optimizer,
        device,
        scheduler=None,
        accumulation_step=None):

    model.train()
    epoch_loss = []
    model.zero_grad()

    for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
        ids = data['ids'].to(device)
        targets = data['targets'].to(device)

        output = model(ids)
        loss = criterion(output, targets)
        epoch_loss.append(loss.item())

        if (i+1) % accumulation_step == 0:
            loss.backward()
            optimizer.step()

            if scheduler:
                scheduler.step()

            model.zero_grad()

    return sum(epoch_loss)/len(epoch_loss)


def eval(
        dataloader,
        model,
        criterion,
        device):
        
    model.eval()
    epoch_loss = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=len(dataloader)):
            ids = data['ids'].to(device)
            targets = data['targets'].to(device)

            output = model(ids)
            loss = criterion(output, targets)
            epoch_loss.append(loss.item())

    return sum(epoch_loss)/len(epoch_loss)
