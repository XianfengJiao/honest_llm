from datasets import load_dataset
import pandas as pd

url = "https://huggingface.co/api/datasets/openbookqa/parquet/additional/train/0.parquet"
dataset = load_dataset('parquet', data_files=url)
# 定义一个函数来提取正确和错误的答案
def extract_answers(row):
    # 获取正确答案的索引
    correct_index = row['choices']['label'].index(row['answerKey'])
    # 提取正确答案
    correct_answer = row['choices']['text'][correct_index]
    # 提取错误答案
    incorrect_answers = [ans for i, ans in enumerate(row['choices']['text']) if i != correct_index]
    incorrect_answers = '; '.join(incorrect_answers)

    return pd.Series([correct_answer, incorrect_answers])

df = pd.DataFrame(dataset['train'])
df['Question'] = df['question_stem']
df[['Correct Answers', 'Incorrect Answers']] = df.apply(extract_answers, axis=1)
df['Best Answer'] = df['Correct Answers']
print(dataset)


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
df['Correct Answers'] = df['answer'].apply(lambda x: ';'.join(x['aliases']))
df['Best Answer'] = df['answer'].apply(lambda x: x['value'])
df['Incorrect Answers'] = df['false_answer']
print(dataset)


df = pd.read_csv('./TruthfulQA/data/v0/TruthfulQA.csv')