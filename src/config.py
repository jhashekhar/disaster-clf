# learn to use tokenizer dispatcher
from transformers import BertTokenizer, RobertaTokenizer, XLNetTokenizer

# hyper-parameters
MAX_LEN = 160
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
ACCUMULATION = 4

MODEL_PATH = "model.bin"

# model paths
PATHS = {
    'bert': 'bert-base-uncased',
    'roberta': 'roberta-base',
    'xlnet': 'xlnet-base-cased'
        }

# tokenizer paths
TOKENIZER = {
    'bert': BertTokenizer.from_pretrained(PATHS['bert'], do_lower_case=True),
    'roberta': RobertaTokenizer.from_pretrained(PATHS['roberta'], do_lower_case=True),
    'xlnet': XLNetTokenizer.from_pretrained(PATHS['xlnet'], do_lower_case=True)
            }