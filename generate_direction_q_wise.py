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

import sys
sys.path.append('../')
from utils import *

HF_NAMES = {
        # 'llama_7B': 'decapoda-research/llama-7b-hf',
        'llama_7B': 'yahma/llama-7b-hf',
        'alpaca_7B': 'circulus/alpaca-7b', 
        'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
        # 'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
        'llama2_chat_7B': 'daryl149/llama-2-7b-chat-hf',
        'llama_13B': 'luodian/llama-13b-hf',
        'llama_33B': 'alexl83/LLaMA-33B-HF',
    }


# 'llama_7B', 'llama2_7B', 'llama2_chat_7B', 'alpaca_7B', 'vicuna_7B'
model_name = 'llama_33B'
model_name_hf = HF_NAMES[model_name]

head_wise_activations = pkl.load(open(f'/data/jxf/activations/{model_name}_tqa_mc2_all_100_head_wise.pkl', 'rb'))
labels = np.load(f'/data/jxf/activations/{model_name}_tqa_mc2_all_100_labels.npy')
activation_categories = pkl.load(open(f'/data/jxf/activations/{model_name}_tqa_mc2_all_100_categories.pkl', 'rb'))
tokens = pkl.load(open(f'/data/jxf/activations/{model_name}_tqa_mc2_all_100_tokens.pkl', 'rb'))
model = llama.LLaMAForCausalLM.from_pretrained(model_name_hf, low_cpu_mem_usage = True, torch_dtype=torch.float16, device_map="auto")

num_heads = model.config.num_attention_heads
head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', d = num_heads)


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
os.makedirs('/data/jxf/honest_llm/directions', exist_ok=True)
np.save(f'/data/jxf/honest_llm/directions/{model_name}_head_wise_activation_directions.npy', head_wise_activation_directions)

