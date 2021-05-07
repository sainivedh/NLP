import transformers
import tokenizers
import os
import torch

MAX_LENGTH = 500
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MODEL_PATH = 'model.bin'
DATA_FILE = './BioASQ_data.csv'
BERT_MODEL = transformers.BertModel.from_pretrained('./Models')
TOKENIZER  = tokenizers.BertWordPieceTokenizer('./Models/Tokenizer/vocab.txt', lowercase=True)