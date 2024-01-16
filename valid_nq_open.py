from datasets import load_dataset
import pandas as pd

url = "https://huggingface.co/api/datasets/OamPatel/iti_nq_open_val/parquet/default/validation/0.parquet"
dataset = load_dataset('parquet', data_files=url)['train']

df = pd.DataFrame(dataset)
print(dataset)


url = "https://huggingface.co/api/datasets/OamPatel/iti_trivia_qa_val/parquet/default/validation/0.parquet"
dataset = load_dataset('parquet', data_files=url)['train']

df = pd.DataFrame(dataset)
print(dataset)