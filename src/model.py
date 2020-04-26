from transformers import BertModel, RobertaModel, XLNetModel

import torch.nn as nn
import config


class BERTBaseUncased(nn.Module):
    def __init__(self, dropout):
        super(BERTBaseUncased, self).__init__()
        self.bert = BertModel.from_pretrained(config.PATHS['bert'])
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
        self.roberta = RobertaModel.from_pretrained(config.PATHS['roberta'])
        self.fc = nn.Linear(768, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids, mask, token_type_ids):
        hidden_state, scores = self.roberta(
            ids,
            attention_mask = mask,
            token_type_ids = token_type_ids
        )

        return hidden_state, scores


class XLNETModel(nn.Module):
    def __init__(self, dropout, max_len=None):
        super(XLNETModel, self).__init__()
        self.xlnet = XLNetModel.from_pretrained(config.PATHS['xlnet'])
        self.fc = nn.Linear(768 * max_len, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, ids):
        out = self.xlnet(ids)
        out = self.dropout(out[0])
        xl_out = out.reshape(out.size(0), -1)
        fc_out = self.fc(xl_out)
        return fc_out


Model = {
    'bert': BERTBaseUncased,
    'roberta': ROBERTAModel,
    'xlnet': XLNETModel
}