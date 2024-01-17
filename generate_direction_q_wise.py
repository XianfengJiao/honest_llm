import pickle as pkl
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from datasets import load_dataset
from einops import rearrange
from utils import get_separated_activations, flattened_idx_to_layer_head
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import requests


# 'llama_7B', 'llama2_7B', 'llama2_chat_7B', 'alpaca_7B', 'vicuna_7B'
model_name = 'llama2_chat_7B'

head_wise_activations = pkl.load(open(f'/data/wtl/honest_llm/activations/{model_name}_tqa_mc2_all_100_head_wise.pkl', 'rb'))
labels = np.load(f'/data/wtl/honest_llm/activations/{model_name}_tqa_mc2_all_100_labels.npy')
activation_categories = pkl.load(open(f'/data/wtl/honest_llm/activations/{model_name}_tqa_mc2_all_100_categories.pkl', 'rb'))
tokens = pkl.load(open(f'/data/wtl/honest_llm/activations/{model_name}_tqa_mc2_all_100_tokens.pkl', 'rb'))

# head_wise_activations = pkl.load(open(f'/data/wtl/honest_llm/activations/{model_name}_tqa_gen_end_q_all_100_head_wise.pkl', 'rb'))
# labels = np.load(f'/data/wtl/honest_llm/activations/{model_name}_tqa_gen_end_q_all_100_labels.npy')
# activation_categories = pkl.load(open(f'/data/wtl/honest_llm/activations/{model_name}_tqa_gen_end_q_all_100_categories.pkl', 'rb'))
# tokens = pkl.load(open(f'/data/wtl/honest_llm/activations/{model_name}_tqa_gen_end_q_all_100_tokens.pkl', 'rb'))
num_heads = 32
head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)


def get_separated_activations(labels, head_wise_activations, categories): 

    # separate activations by question
    url = "https://huggingface.co/api/datasets/truthful_qa/parquet/multiple_choice/validation/0.parquet"
    dataset = load_dataset('parquet', data_files=url)['train']
    actual_labels = []
    for i in range(len(dataset)):
        actual_labels.append(dataset[i]['mc2_targets']['labels'])
    

    idxs_to_split_at = np.cumsum([len(x) for x in actual_labels])        

    labels = list(labels)
    categories = list(categories)
    separated_labels = []
    separated_categories = []
    for i in range(len(idxs_to_split_at)):
        if i == 0:
            separated_labels.append(labels[:idxs_to_split_at[i]])
            separated_categories.append(categories[:idxs_to_split_at[i]])
        else:
            separated_labels.append(labels[idxs_to_split_at[i-1]:idxs_to_split_at[i]])
            separated_categories.append(categories[idxs_to_split_at[i-1]:idxs_to_split_at[i]])

    separated_head_wise_activations = np.split(head_wise_activations, idxs_to_split_at)

    return separated_head_wise_activations, separated_labels, separated_categories, idxs_to_split_at

separated_head_wise_activations, separated_labels, separated_categories, idxs_to_split_at = get_separated_activations(labels, head_wise_activations, activation_categories)


# 获得每一个问题的方向
head_wise_activation_directions = np.array([a[np.array(l) == 1].mean(axis=0) - a[np.array(l) == 0].mean(axis=0) for a, l in zip(separated_head_wise_activations, separated_labels)])
os.makedirs('/data/wtl/honest_llm/directions', exist_ok=True)
np.save(f'/data/wtl/honest_llm/directions/{model_name}_head_wise_activation_directions.npy', head_wise_activation_directions)


