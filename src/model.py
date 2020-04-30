from transformers import (
    BertConfig,
    BertModel,
    RobertaConfig,
    RobertaModel,
    DistilBertConfig,
    DistilBertModel,
    XLNetConfig,
    XLNetModel,
    )

import torch.nn as nn
import config


class BERTModel(nn.Module):
    def __init__(self, dropout):
        super(BERTModel, self).__init__()

        self.bert = BertModel.from_pretrained(
            config.PATHS['bert'],
            config=BertConfig())

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(768, 2)

    def forward(self, input):
        _, out2 = self.bert(input)
        bo = self.dropout(out2)
        output = self.fc(bo)
        return output


class ROBERTAModel(nn.Module):
    def __init__(self, dropout):
        super(ROBERTAModel, self).__init__()

        self.roberta = RobertaModel.from_pretrained(
            config.PATHS['roberta'],
            config=RobertaConfig())

        self.fc = nn.Linear(768, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids, mask, token_type_ids):
        roberta_out = self.roberta(ids)
        out = self.dropout(roberta_out)
        out = self.fc(out)
        return out


class DISTILBertModel(nn.Module):
    def __init__(self, dropout):
        super(DISTILBertModel, self).__init__()

        self.distilbert = DistilBertModel.from_pretrained(
            config.PATHS['distilbert'],
            config=DistilBertConfig())

        self.fc = nn.Linear(768, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids):
        output = self.distilbert(ids)
        distil_out = self.dropout(output[1])
        out = self.fc(distil_out)
        return out


class XLNETModel(nn.Module):
    def __init__(self, dropout, max_len=None):
        super(XLNETModel, self).__init__()

        self.xlnet = XLNetModel.from_pretrained(
            config.PATHS['xlnet'],
            config=XLNetConfig())

        self.fc = nn.Linear(768 * max_len, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids):
        xlnet_out = self.xlnet(ids)
        out = self.dropout(xlnet_out[0])
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


Model = {
    'bert': BERTModel,
    'roberta': ROBERTAModel,
    'xlnet': XLNETModel,
    'distilbert': DISTILBertModel
    }
