from transformers import BertModel, RobertaModel, XLNetModel

import torch.nn as nn
import config


class BERTBaseUncased(nn.Module):
    def __init__(self, dropout):
        super(BERTBaseUncased, self).__init__()
        self.bert = BertModel.from_pretrained(config.PATHS['bert'])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, 2)  # 1 for binary classifier, can use more than 1 need different loss function

    def forward(self, input):
        # out1 = sequence of hidden states for batch shape (BATCH_SIZE * 512) vectors with each vector of (1 * 768) length
        # out2 = CLS pooler output (BATCH_SIZE * 1) with each having shape of (1 * 768)
        _, out2 = self.bert(input)
        bo = self.dropout(out2)
        output = self.fc(bo)
        return output

        
class ROBERTAModel(nn.Module):
    def __init__(self, dropout):
        super(ROBERTAModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(config.PATHS['roberta'])
        self.fc = nn.Linear(768, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids, mask, token_type_ids):
        hidden_staes, scores = self.roberta(
            ids,
            attention_mask = mask,
            token_type_ids = token_type_ids
        )

        return hidden_staes, scores


class XLNETModel(nn.Module):
    def __init__(self, dropout):
        super(XLNETModel, self).__init__()
        self.xlnet = XLNetModel.from_pretrained(config.PATHS['xlnet'])
        self.fc = nn.Linear(768, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids, mask, token_type_ids):
        out = self.xlnet(
            ids,
            attention_mask = mask,
            token_type_ids = token_type_ids
        )
        return out
