import pandas as pd
import torch

import argparse

import sys
sys.path.insert(0, '/content/src')

import config
from train import train, eval
from dataset import DisasterDataset
from model import Model


parser = argparse.ArgumentParser(
    description='arguments for training and inference')

parser.add_argument(
    "-file_path", type=str, help="file path")
parser.add_argument(
    "-epochs", type=int, help="Number of epochs")
parser.add_argument(
    "-model", type=str, help="'bert', 'roberta', 'distilbert', 'xlnet'")
parser.add_argument(
    "-optimizer", type=str, help="'Adam' and 'AdamW' ")
parser.add_argument(
    "-scheduler", type=str, help="'StepLR' and 'MultiStepLR'")
parser.add_argument(
    "-bs", type=int, help="batch size")


args = parser.parse_args()

config.TRAIN_BATCH_SIZE = args.bs
config.VALID_BATCH_SIZE = args.bs

df = pd.read_csv(args.file_path)

t = int(0.9 * len(df))
train_df = df[:t]
valid_df = df[t:]

trainset = DisasterDataset(
    train_df.text.values,
    train_df.target.values,
    args.model)

validset = DisasterDataset(
    valid_df.text.values,
    valid_df.target.values,
    args.model)

trainloader = torch.utils.data.DataLoader(
                                trainset,
                                batch_size=config.TRAIN_BATCH_SIZE,
                                shuffle=True)

validloader = torch.utils.data.DataLoader(
                                validset,
                                batch_size=config.VALID_BATCH_SIZE,
                                shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if args.model == 'xlnet':
    model = Model[args.model](dropout=0.3, max_len=config.MAX_LEN).to(device)

if args.model == 'bert':
    model = Model[args.model](dropout=0.3).to(device)

else:
    model = Model[args.model](dropout=0.3).to(device)

criterion = config.loss
optimizer = config.optimizer[args.optimizer](model.parameters(), config.LR)


if args.scheduler == 'steplr':
    scheduler = config.scheduler['steplr'](optimizer, config.STEP_SIZE)

if args.scheduler == 'multisteplr':
    scheduler = config.scheduler['multisteplr'](optimizer, config.MILESTONES)


def save_checkpoint(model, PATH):
    torch.save(model, PATH)


def run():
    for epoch in range(args.epochs):
        if args.scheduler:
            train_loss = train(
                trainloader,
                model,
                criterion,
                optimizer,
                device,
                scheduler=scheduler,
                accumulation_step=config.ACCUMULATION)
        else:
            train_loss = train(
                trainloader,
                model,
                criterion,
                optimizer,
                device,
                accumulation_step=config.ACCUMULATION)

        val_loss = eval(
            validloader,
            model,
            criterion,
            device)

        print(f"Epoch: {epoch+1}/{args.epochs}, train_loss: {train_loss:.5f}, val_loss: {val_loss:.5f}")


if __name__ == "__main__":
    run()
