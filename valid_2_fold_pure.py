import torch
from einops import rearrange
import numpy as np
import pickle
import os
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset
import pickle as pkl
import torch.nn.functional as F

import sys
sys.path.append('../')
from utils import *

HF_NAMES = {
    'llama_7B': 'yahma/llama-7b-hf',
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf',
    'llama_13B': 'luodian/llama-13b-hf',
    'llama_33B': 'alexl83/LLaMA-33B-HF',
}


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--device', type=int, default=1, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--judge_name', type=str, default='ft:davinci-002:university-of-edinburgh::8ejp8D64')
    parser.add_argument('--info_name', type=str, default='ft:davinci-002:university-of-edinburgh:info:8ejuTaQe')
    args = parser.parse_args()

    print('Running:\n{}\n'.format(' '.join(sys.argv)))
    print(args)
    experiment_name = f'valid_2_fold_{args.model_name}_pure'
    experiments_path = f'/data/jxf/honest_llm/validation/{experiment_name}'
    os.makedirs(experiments_path, exist_ok=True)
    print(f'experiments_path: {experiments_path}')

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load dataframe and activations direcitons
    df = pd.read_csv('./TruthfulQA/data/v0/TruthfulQA.csv')

    # order csv by huggingface order, the order used to save activations
    # dataset = load_dataset("truthful_qa", "multiple_choice")['validation']
    url = "https://huggingface.co/api/datasets/truthful_qa/parquet/multiple_choice/validation/0.parquet"
    dataset = load_dataset('parquet', data_files=url)['train']
    golden_q_order = list(dataset["question"])
    df = df.sort_values(by='Question', key=lambda x: x.map({k: i for i, k in enumerate(golden_q_order)}))

    dictionary = {k: i for i, k in enumerate(golden_q_order)}
    for q in df['Question']:
        assert q in dictionary
    
    # get two folds using numpy
    fold_idxs = np.array_split(np.arange(len(df)), args.num_fold)

    # create model
    model_name = HF_NAMES[args.model_name]
    tokenizer = llama.LLaMATokenizer.from_pretrained(model_name)
    if args.model_name == 'llama_7B':
        model = llama.LLaMAForCausalLM.from_pretrained(model_name, low_cpu_mem_usage = True, torch_dtype=torch.float16, device_map=args.device)
        r = model.to(args.device)
        device = args.device
    else:
        model = llama.LLaMAForCausalLM.from_pretrained(model_name, low_cpu_mem_usage = True, torch_dtype=torch.float16, device_map="auto")
        device = 'cuda'

    # run k-fold cross validation
    results = []
    for i in range(args.num_fold):

        train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != i])
        test_idxs = fold_idxs[i]

        print(f"Running fold {i}")

        # pick a val set using numpy
        train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
        val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

        # save train and test splits
        df.iloc[train_set_idxs].to_csv(f"{experiments_path}/fold_{i}_train_seed_{args.seed}.csv", index=False)
        df.iloc[val_set_idxs].to_csv(f"{experiments_path}/fold_{i}_val_seed_{args.seed}.csv", index=False)
        df.iloc[test_idxs].to_csv(f"{experiments_path}/fold_{i}_test_seed_{args.seed}.csv", index=False)

        filename = f'{args.model_name}_seed_{args.seed}_fold_{i}'
                    
        curr_fold_results = alt_tqa_evaluate(
            {args.model_name: model}, 
            ['mc','bleu','bleurt', 'judge', 'info'], 
            f'{experiments_path}/fold_{i}_test_seed_{args.seed}.csv', 
            f'{experiments_path}/answer_{filename}.csv', 
            f'{experiments_path}/summary_{filename}.csv', 
            device=device, 
            interventions={},
            intervention_fn=None, 
            judge_name=args.judge_name, 
            info_name=args.info_name,
            use_cluster=False,
            sample_directions = None
        )

        print(f"FOLD {i}")
        print(curr_fold_results)

        curr_fold_results = curr_fold_results.to_numpy()[0].astype(float)
        results.append(curr_fold_results)
    
    results = np.array(results)
    final = results.mean(axis=0)

    # print(f'BLEURT acc: {final[0]:.4f}, MC1: {final[1]:.4f}, MC2: {final[2]:.4f}, bleu acc: {final[3]:.4f}, rouge1 acc: {final[4]:.4f}, CE Loss: {final[5]}, KL wrt Original: {final[6]}')
    print(f'True*Info Score: {final[1]*final[2]}, True Score: {final[2]}, Info Score: {final[1]}, BLEURT acc: {final[0]:.4f}, MC1: {final[3]:.4f}, MC2: {final[4]:.4f}, bleu acc: {final[5]:.4f}, rouge1 acc: {final[6]:.4f}, CE Loss: {final[7]}, KL wrt Original: {final[8]}')

if __name__ == "__main__":
    main()
