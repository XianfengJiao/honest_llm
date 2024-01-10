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
}


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument('--probe_base_weight', type=float, default=0)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--probe_type', type=str, default='prob')
    parser.add_argument('--pure', action='store_true', default=False)
    parser.add_argument('--method', type=str, default='icl')
    parser.add_argument("--num_fold", type=int, default=1, help="number of folds")
    parser.add_argument('--fewshot_ratio', type=float, default=0.053)
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.05)
    parser.add_argument('--device', type=int, default=2, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--n_clusters', type=int, default=3)
    parser.add_argument('--judge_name', type=str, required=False)
    parser.add_argument('--info_name', type=str, required=False)
    args = parser.parse_args()

    print('Running:\n{}\n'.format(' '.join(sys.argv)))
    print(args)

    if args.pure:
        experiment_name = f'fewshot_pure_{args.method}'
    else:
        experiment_name = f'fewshot_cluster_probe_num_heads{args.num_heads}_alpha{args.alpha}_n_clusters{args.n_clusters}_baseW{args.probe_base_weight}_{args.probe_type}_{args.method}'
    experiments_path = f'/data/jxf/honest_llm/cluster_experiments/{experiment_name}'
    os.makedirs(experiments_path, exist_ok=True)
    print(f'experiments_path: {experiments_path}')

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load dataframe and activations direcitons
    df = pd.read_csv('./TruthfulQA/data/v0/TruthfulQA.csv')
    head_wise_activation_directions = np.load('/data/jxf/honest_llm/directions/head_wise_activation_directions.npy')
    # norms = np.linalg.norm(head_wise_activation_directions, axis=-1, keepdims=True)
    # # 避免除以零的情况
    # norms[norms == 0] = np.inf  # 将为0范数设置为无穷大，这样除以无穷大会得到0
    # head_wise_activation_directions = head_wise_activation_directions / norms


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
    model = llama.LLaMAForCausalLM.from_pretrained(model_name, low_cpu_mem_usage = True, torch_dtype=torch.float16)
    r = model.to(args.device)
    device = args.device
    
    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # load activations 
    head_wise_activations = pkl.load(open('/data/jxf/activations/llama_7B_tqa_mc2_all_100_head_wise.pkl', 'rb'))
    labels = np.load('/data/jxf/activations/llama_7B_tqa_mc2_all_labels.npy')
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)

    # separated_head_wise_activations: shape(question_nums, answer_nums, layer_nums, head_nums, 128)
    separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations)

    results = []
    # run k-fold cross validation
    for i in range(args.num_fold):

        all_idxs = fold_idxs[i]

        print(f"Running fold {i}")

        # pick a val set using numpy
        val_set_idxs = np.random.choice(all_idxs, size=int(len(all_idxs)*(args.val_ratio)), replace=False)
        test_set_idxs = np.array([x for x in all_idxs if x not in val_set_idxs])

        train_set_idxs = np.random.choice(test_set_idxs, size=int(len(test_set_idxs)*(args.fewshot_ratio)), replace=False)
        test_set_idxs = np.array([x for x in test_set_idxs if x not in train_set_idxs])

        # test_set_idxs = np.random.choice(test_set_idxs, size=int(len(test_set_idxs)*(0.05/0.9)), replace=False)

        many_shot_prefix = None
        if args.method == 'icl':
            many_shot_prefix = fewshot_eqipped(df.iloc[train_set_idxs])

        # save train and test splits
        df.iloc[train_set_idxs].to_csv(f"{experiments_path}/fold_{i}_train_seed_{args.seed}.csv", index=False)

        df.iloc[val_set_idxs].to_csv(f"{experiments_path}/fold_{i}_val_seed_{args.seed}.csv", index=False)
        df.iloc[test_set_idxs].to_csv(f"{experiments_path}/fold_{i}_test_seed_{args.seed}.csv", index=False)

        # get directions
        if args.use_center_of_mass:
            print('use mean direction')
            com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels)
        else:
            com_directions = None

        top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir)

        print("Heads intervened: ", sorted(top_heads))
    
        interventions = get_interventions_dict(top_heads, probes, head_wise_activations, num_heads, args.use_center_of_mass, args.use_random_dir, com_directions)

        # sample_directions
        sample_directions = head_wise_activation_directions[test_set_idxs]


        def lt_modulated_vector_add(head_output, layer_name, start_edit_location='lt'): 
            head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
            for head, direction, proj_val_std in interventions[layer_name]:
                direction_to_add = torch.tensor(direction).to(args.device)
                if start_edit_location == 'lt': 
                    head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add
                else: 
                    head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add
            head_output = rearrange(head_output, 'b s h d -> b s (h d)')
            return head_output

        filename = f'{args.model_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{int(args.alpha)}_fold_{i}'
                    
        curr_fold_results = alt_tqa_evaluate(
            {args.model_name: model}, 
            ['mc','bleu','bleurt'], 
            f'{experiments_path}/fold_{i}_test_seed_{args.seed}.csv', 
            f'{experiments_path}/answer_{filename}.csv', 
            f'{experiments_path}/summary_{filename}.csv', 
            device=args.device, 
            interventions=interventions if not args.pure else {},
            intervention_fn=lt_modulated_vector_add if not args.pure else None,
            judge_name=args.judge_name, 
            info_name=args.info_name,
            use_cluster=False,
            sample_directions = sample_directions,
            many_shot_prefix = many_shot_prefix,
        )

        print(f"FOLD {i}")
        print(curr_fold_results)

        curr_fold_results = curr_fold_results.to_numpy()[0].astype(float)
        results.append(curr_fold_results)
    
    results = np.array(results)
    final = results.mean(axis=0)

    print(f'BLEURT acc: {final[0]:.4f}, MC1: {final[1]:.4f}, MC2: {final[2]:.4f}, bleu acc: {final[3]:.4f}, rouge1 acc: {final[4]:.4f}, CE Loss: {final[5]}, KL wrt Original: {final[6]}')

    # print(f'True*Info Score: {final[1]*final[0]}, True Score: {final[1]}, Info Score: {final[0]}, MC1 Score: {final[2]}, MC2 Score: {final[3]}, CE Loss: {final[4]}, KL wrt Original: {final[5]}')

if __name__ == "__main__":
    main()