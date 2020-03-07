import config
import transformers
import torch.nn as nn

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)  # 1 for binary classifier, can use more than 1 need different loss function


    def forward(self, ids, mask, token_type_ids):
        # out1 = sequence of hidden states for batch shape (BATCH_SIZE * 512) vectors with each vector of (1 * 768) length
        # out2 = CLS pooler output (BATCH_SIZE * 1) with each having shape of (1 * 768)
        _, out2 = self.bert(
            ids, 
            attention_mask = mask,
            token_type_ids = token_type_ids
        )
        bo = self.bert_drop(out2)
        output = self.out(bo)
        return output

        
