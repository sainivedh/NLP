import transformers

BERT_MODEL_PATH = './Models/'
TOKENIZER_PATH = './Models/Tokenizer/'
MAX_LENGTH = 32
DEVICE = 'cuda'
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
EPOCHS = 10
TRAIN_DATA = './Data/train.csv'
VALID_DATA = './Data/valid.csv'
MODEL_PATH = 'model.bin'
TOKENIZER = transformers.BertTokenizer.from_pretrained(
    TOKENIZER_PATH, do_lower_case=True
)