import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import utils
import config
from contractions import contractions


class DisasterDataset(object):
    def __init__(self, text, target, tokenizer, transforms=None):
        super(DisasterDataset, self).__init__()

        self.text = text
        self.target = target
        self.transforms = transforms        
        self.tokenizer = config.TOKENIZER[tokenizer]
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):

        text = self.text[idx]
        target = self.target[idx]

        inputs = self.tokenizer.encode_plus(
            text, 
            None,
            add_special_tokens=True,
            max_length=self.max_len
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
    
        # add padding
        padding_len = self.max_len - len(ids)
        ids = ids + [0] * padding_len
        mask = mask + [0] * padding_len
        token_type_ids = token_type_ids + [0] * padding_len
        
        return {
            'ids': torch.tensor(ids, dtype=torch.long), 
            'mask': torch.tensor(mask, dtype=torch.long), 
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(target, dtype=torch.long)
               }
               

class Transforms(object):
    def __init__(self, embedding=None, vector_size=None, hidden_dim=None):
        self.embedding = nn.Embedding(vector_size, hidden_dim)

    def pipeline(self, text):
        text = utils.remove_space(text)
        text = utils.remove_punct(text)
        text = utils.remove_contractions(text.lower(), contractions)
        text = utils.remove_url(text)
        text = utils.remove_html(text)
        text = utils.correct_spellings(text)

        return text

    def __call__(self, sample):
        tweets, targets = sample['tweets'], sample['targets']
        transformed_tweets = self.pipeline(tweets)
        sample = {'tweets': transformed_tweets, 'targets': targets}
        return sample
