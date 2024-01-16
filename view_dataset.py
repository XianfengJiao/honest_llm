from datasets import load_dataset
import pandas as pd

url = "https://huggingface.co/api/datasets/OamPatel/iti_nq_open_val/parquet/default/validation/0.parquet"
dataset = load_dataset('parquet', data_files=url)['train']

df = pd.DataFrame(dataset)

df['Correct Answers'] = df['answer'].apply(lambda x: x[0])
df['Best Answer'] = df['answer'].apply(lambda x: x[0])
df['Incorrect Answers'] = df['false_answer']
print(dataset)


url = "https://huggingface.co/api/datasets/OamPatel/iti_trivia_qa_val/parquet/default/validation/0.parquet"
dataset = load_dataset('parquet', data_files=url)['train']

df = pd.DataFrame(dataset)
df['Correct Answers'] = df['answer'].apply(lambda x: ';'.join(x['normalized_aliases']))
df['Best Answer'] = df['answer'].apply(lambda x: x['normalized_value'])
df['Incorrect Answers'] = df['false_answer']
print(dataset)


df = pd.read_csv('./TruthfulQA/data/v0/TruthfulQA.csv')