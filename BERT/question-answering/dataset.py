import torch
import config
import pandas as pd
from tqdm import tqdm

def preprocess_data(context, question, answer, answer_start_id, tokenizer, max_len):
    context = context[2:-1]
    question = question[2:-1]
    answer = answer[2:-1]
    len_ans = len(answer)
    '''
    start_idx = answer_start_id
    end_idx = answer_start_id + len_ans - 1
    print(answer, context[start_idx:end_idx+1])
    assert answer == context[start_idx:end_idx+1]
    '''
    start_idx = None
    end_idx = None
    for i in (i for i, e in enumerate(context) if context[i] == answer[0]):
        if answer == context[i: i+len(answer)]:
            start_idx = i
            end_idx = i + len(answer) - 1
    #print(start_idx, end_idx)
    #print(answer, context[start_idx:end_idx + 1])
    assert answer == context[start_idx:end_idx + 1]
    char_targets = [0] * len(context)
    for ct in range(start_idx, end_idx+1):
        char_targets[ct] = 1

    context_ids = tokenizer.encode(context)
    question_ids = tokenizer.encode(question).ids
    answer_ids = tokenizer.encode(answer).ids

    input_ids_orig = context_ids.ids[1:-1]
    context_offsets = context_ids.offsets[1:-1]

    target_idx = []
    for ind, (start, end) in enumerate(context_offsets):
        if sum(char_targets[start:end]) >= 1:
            target_idx.append(ind)

    targets_start = target_idx[0]
    targets_end = target_idx[-1]


    input_ids = question_ids + input_ids_orig + [102]
    token_type_ids = [0] * len(question_ids) + [1] * (len(input_ids_orig) + 1)
    assert len(token_type_ids) <= 512
    mask = [1] * len(token_type_ids)
    context_offsets = [(0, 0)] * len(question_ids) +  context_offsets + [(0, 0)]
    targets_start = targets_start + len(question_ids)
    targets_end = targets_end + len(question_ids)

    padding_length = max_len - len(input_ids)
    if padding_length > 0:
        input_ids = input_ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)
        context_offsets = context_offsets + ([(0,0)] * padding_length)

    return {
        "ids": input_ids,
        "mask": mask,
        "token_type_ids": token_type_ids,
        "targets_start": targets_start,
        "targets_end": targets_end,
        "orig_context": context,
        "orig_answer": answer,
        "question": question,
        "offsets": context_offsets
    }

class BioASQDataset:
    def __init__(self, context, question, answer, answer_start_id):
        self.context = context
        self.question = question
        self.answer = answer
        self.answer_start_id = answer_start_id
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LENGTH

    def __len__(self):
        return len(self.context)

    def __getitem__(self, item):
        data = preprocess_data(self.context[item], self.question[item],
                               self.answer[item], self.answer_start_id[item], self.tokenizer,
                               self.max_len)

        return {
            "ids": torch.tensor(data['ids'], dtype=torch.long),
            "mask": torch.tensor(data['mask'], dtype=torch.long),
            "token_type_ids": torch.tensor(data['token_type_ids'], dtype=torch.long),
            "targets_start": torch.tensor(data['targets_start'], dtype=torch.long),
            "tragets_end": torch.tensor(data['targets_end'], dtype=torch.long),
            "orig_context": data['orig_context'],
            "orig_answer": data['orig_answer'],
            "question": data['question'],
            "offsets": torch.tensor(data['offsets'], dtype=torch.long)
        }

def select_rows(df):
    rows_index = []
    for i in range(len(df)):
        if len(df['context'][i].split()) < 100:
            rows_index.append(i)
    return df.iloc[rows_index]




if __name__ == '__main__':
    df = pd.read_csv('BioASQ_data_mod.csv')
    print(len(df))
    df = select_rows(df)
    print(len(df))
    dataset = BioASQDataset(df['context'].values, df['question'].values, df['answer'].values,
                            df['answer_start_id'].values)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.TRAIN_BATCH_SIZE)
    for d in tqdm(dataloader, total=len(dataloader)):
        print(d['ids'].shape)
        print(d['mask'].shape)
        print(d['token_type_ids'].shape)
        print('-------------------')



