import transformers
import torch

MAX_LEN = 128
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = './Models'
TOKENIZER_MODEL_PATH = './Models/Tokenizer/'
MODEL_PATH = 'Entity_model.bin'
TRAINING_FILE = 'ner_dataset.csv'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

TOKENIZER = transformers.BertTokenizer.from_pretrained(TOKENIZER_MODEL_PATH, do_lower_case=True)


