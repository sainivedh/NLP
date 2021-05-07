import config
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

text = "where the fuck is ids and offsets"

def length_tokens(text):
    return len(config.TOKENIZER.encode(text).ids)

df = pd.read_csv('BioASQ_data.csv')

a= []
b = []
for i in range(len(df['context'])):
    b.append(length_tokens(df['context'][i]))
    a.append(len(df['context'][i].split()))

f, axes = plt.subplots(1,2)
sns.boxplot(b,orient='v', ax=axes[0])
sns.boxplot(a, orient='v', ax=axes[1])
plt.show()
