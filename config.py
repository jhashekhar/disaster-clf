# learn to use tokenizer dispatcher

import transformers


MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
ACCUMULATION = 2
BERT_PATH = 'bert-base-uncased'
MODEL_PATH = "model.bin"
tokenizer  = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
