import config

import numpy as np
import pandas as pd



import torch
from torch.utils.data import Dataset

import utils
from contractions import contractions

class DisasterDataset(Dataset):
    def __init__(self, csv_file, transforms=None):
        super(DisasterDataset, self).__init__()

        self.df = pd.read_csv(csv_file)
        self.transforms = transforms


        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        tweets = self.df.loc[idx, 'text'].values
        targets = self.df.loc[idx, 'target'].values

        inputs = self.tokenizer.encode_plus(
            tweets, 
            None,
            add_special_tokens=True,
            max_length=self.max_len
        )

        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs['token_type_ids']
    
        return {'ids': ids, 'mask': mask, 'token_type_ids': token_type_ids}


class Transforms(object):
    def __init__(self, embedding=None, vector_size=None):
        self.embedding = embedding
        self.vector_size = vector_size

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


transforms = Transforms()
print(transforms({'tweets': "I love India! #IndiaMeriJaannnn ##Sunday https://github.com", 
'targets': 1}))