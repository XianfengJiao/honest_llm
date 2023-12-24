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
from utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_cluster_interventions_dict, get_top_heads, get_separated_activations, get_cluster_mean_directions
import llama

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
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--n_clusters', type=int, default=4)
    parser.add_argument('--judge_name', type=str, required=False)
    parser.add_argument('--info_name', type=str, required=False)
    args = parser.parse_args()

    print('Running:\n{}\n'.format(' '.join(sys.argv)))
    print(args)

    experiment_name = f'num_heads{args.num_heads}_alpha{args.alpha}_n_clusters{args.n_clusters}'
    experiments_path = f'/data/wtl/honest_llm/cluster_experiments/{experiment_name}'
    os.makedirs(experiments_path, exist_ok=True)
    print(f'experiments_path: {experiments_path}')

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load dataframe and activations direcitons
    df = pd.read_csv('./TruthfulQA/data/v0/TruthfulQA.csv')
    head_wise_activation_directions = np.load('/data/wtl/honest_llm/directions/head_wise_activation_directions.npy')

    # order csv by huggingface order, the order used to save activations
    dataset = load_dataset("truthful_qa", "multiple_choice")['validation']
    golden_q_order = list(dataset["question"])
    df = df.sort_values(by='Question', key=lambda x: x.map({k: i for i, k in enumerate(golden_q_order)}))
    
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

        # get direction of cluster center
        com_directions = get_cluster_mean_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, n_clusters=args.n_clusters)
        
        top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, use_random_dir=False)
        print("Heads intervened: ", sorted(top_heads))
    
        interventions = get_cluster_interventions_dict(top_heads, probes, head_wise_activations, num_heads, use_center_of_mass=True, use_random_dir=None, com_directions=com_directions)

        # sample_directions
        sample_directions = head_wise_activation_directions[test_idxs] 

        def lt_modulated_cluster_add(head_output, layer_name, sample_direction, start_edit_location='lt'): 
            head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
            layer = int(re.search(r'\d+', layer_name).group())
            for head, direction, proj_val_std in interventions[layer_name]:
                sample_head_direction = sample_direction[layer, head]
                direction_to_add = torch.tensor(direction *  proj_val_std.reshape(-1, 1)).to(args.device)
                distances = torch.cdist(sample_head_direction.reshape(1, -1).float(), direction_to_add.float())
                weights = F.softmax((-distances).clone().detach(), dim=1)
                weighted_direction = torch.matmul(weights, direction_to_add.float())
                # weighted_direction += direction_to_add.mean(dim=0, keepdim=True)
                # weighted_direction /= 2
                if start_edit_location == 'lt': 
                    head_output[:, -1, head, :] += args.alpha * weighted_direction
                else: 
                    head_output[:, start_edit_location:, head, :] += args.alpha * weighted_direction
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
            interventions=interventions, 
            intervention_fn=lt_modulated_cluster_add, 
            judge_name=args.judge_name, 
            info_name=args.info_name,
            use_cluster=True,
            sample_directions = sample_directions
        )

        print(f"FOLD {i}")
        print(curr_fold_results)

        curr_fold_results = curr_fold_results.to_numpy()[0].astype(float)
        results.append(curr_fold_results)
    
    results = np.array(results)
    final = results.mean(axis=0)

    print(f'BLEURT acc: {final[0]}, MC1: {final[1]}, MC2: {final[2]}, bleu acc: {final[3]}, rouge1 acc: {final[4]}, CE Loss: {final[5]}, KL wrt Original: {final[6]}')

    # print(f'True*Info Score: {final[1]*final[0]}, True Score: {final[1]}, Info Score: {final[0]}, MC1 Score: {final[2]}, MC2 Score: {final[3]}, CE Loss: {final[4]}, KL wrt Original: {final[5]}')

if __name__ == "__main__":
    main()
