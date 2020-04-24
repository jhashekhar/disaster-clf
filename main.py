import numpy as np 
import pandas as pd

import argparse
import sys
sys.path.insert(0, '/content/src')

import torch

import config
from train import train_fn, eval_fn
from dataset import DisasterDataset
from model import BERTBaseUncased


# add arguments
# test it on colab --- how to run python program in colab/jupyter notebook
# TODO:// add scheduler
# TODO:// add option of gradient accumulation

parser = argparse.ArgumentParser(description='arguments for training and inference')

parser.add_argument("-file_path", type=str, help="file path")
#parser.add_argument("-mode", typt=str, help="'train', 'eval'")
parser.add_argument("-epochs", type=int, help="Number of epochs")
parser.add_argument("-lr", type=float, help="learning rate")
parser.add_argument("-model", type=str, help="'bert', 'roberta', 'xlnet'")
#parser.add_argument("-device", type=str, help="gpu, tpu, cpu")
parser.add_argument("-bs", type=int, help="batch size")


args = parser.parse_args()

config.TRAIN_BATCH_SIZE = args.bs
config.VALID_BATCH_SIZE = args.bs

df = pd.read_csv(args.file_path)

t = int(0.9 * len(df))
train = df[:t]
valid = df[t:]

trainset = DisasterDataset(train.text.values, train.target.values, args.model)
validset = DisasterDataset(valid.text.values, valid.target.values, args.model)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
validloader = torch.utils.data.DataLoader(validset, batch_size=config.VALID_BATCH_SIZE, shuffle=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BERTBaseUncased(dropout=0.3).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
criterion = torch.nn.CrossEntropyLoss()

def run():
    for epoch in range(args.epochs):
        train_loss = train_fn(trainloader, model, criterion, optimizer, device)
        val_loss = eval_fn(validloader, model, criterion, device)
        print(f"Epoch: {epoch+1}/{args.epochs}, train_loss: {train_loss:.5f}, val_loss: {val_loss:.5f}")


if __name__ == "__main__":
    run()