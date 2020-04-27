# learn to use tokenizer dispatcher
from transformers import BertTokenizer
from transformers import RobertaTokenizer
from transformers import XLNetTokenizer
from transformers import DistilBertTokenizer
from transformers import AdamW

import torch
import torch.nn as nn

# default values of hyper-parameters
MAX_LEN = 160
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
ACCUMULATION = 4   # gradient accumulation step
LR = 0.001
STEP_SIZE = 5      # for StepLR
MILESTONES = 10    # for MultiStepLR


# paths for tokenizers and pre-trained models
PATHS = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'distilbert': 'distilbert-base-cased',
    'xlnet': 'xlnet-base-cased'}

# tokenizer paths
TOKENIZER = {
    'bert': BertTokenizer.from_pretrained(
                            PATHS['bert'],
                            do_lower_case=True),

    'roberta': RobertaTokenizer.from_pretrained(
                            PATHS['roberta'],
                            do_lower_case=True),

    'distilbert': DistilBertTokenizer.from_pretrained(
                            PATHS['distilbert'],
                            do_lower_case=True),

    'xlnet': XLNetTokenizer.from_pretrained(
                            PATHS['xlnet'],
                            do_lower_case=True)
            }


# criterion
loss = nn.CrossEntropyLoss()

# optimizers and schedulers
optimizer = {
    'adam': torch.optim.Adam,
    'adamw': AdamW
}

scheduler = {
    'steplr': torch.optim.lr_scheduler.StepLR,
    'multisteplr': torch.optim.lr_scheduler.MultiStepLR
}
