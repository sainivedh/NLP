import json
import pandas as pd
import os
import csv
from tqdm import tqdm
from string import digits
import re

json_files = os.listdir('./Factoid')
output_file = open('BioASQ_data.csv','w')
csv_writer = csv.writer(output_file)
remove_digits = str.maketrans('','',digits)


header = ['context','question', 'answer','answer_start_id']
csv_writer.writerow(header)

for json_file in json_files:
    input_file = open(os.path.join('./Factoid', json_file), 'r')
    data_file = json.load(input_file)
    input_file.close()

    for x in tqdm(data_file['data'][0]['paragraphs'], total=len(data_file['data'][0]['paragraphs'])):
        context = x['context'].encode('utf-8')
        #context = x['context']
        #print(context)
        #context = context.translate(remove_digits)
        #context = re.sub(r'\S*\d+\S*', '', context)
        #print(context); exit(1)
        question = x['qas'][0]['question'].encode('utf-8')
        answer_start_id = x['qas'][0]['answers'][0]['answer_start']
        answer = x['qas'][0]['answers'][0]['text'].encode('utf-8')
        #if len(answer) <= 3:
        #    continue
        csv_writer.writerow([context, question, answer, answer_start_id])

output_file.close()
