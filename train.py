# to measure progress
from tqdm import tqdm



def train_fn(dataloader, model, optimizer, device, accumulation_steps):
    pass
    
    #model.train()

    #for bi, d in tqdm(enumerate(dataloader), total=len(dataloader)):
    #    ...


def eval_fn(dataloader, model, device):
    pass
    #model.eval()

    #for bi, d in tqdm(enumerate(dataloader), total=len(dataloader)):
    #    ...
