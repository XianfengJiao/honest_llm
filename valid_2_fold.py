import random
import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np
import pickle as pkl
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from datasets import load_dataset

import sys
sys.path.append('../')
from utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions, get_pca_directions
import llama
from utils import train_probes

HF_NAMES = {
    'llama_7B': 'yahma/llama-7b-hf',
    'llama2_7B': 'meta-llama/Llama-2-7b-hf', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b'
}

def get_top_heads_and_save_accs(experiments_path, fold, train_idxs, val_idxs, separated_activations, separated_labels, num_layers, num_heads, seed, num_to_intervene, use_random_dir=False):

    probes, all_head_accs_np = train_probes(seed, train_idxs, val_idxs, separated_activations, separated_labels, num_layers=num_layers, num_heads=num_heads)
    all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads)
    save_path = f'{experiments_path}/fold_{fold}_pca_head_accs.npy'
    np.save(save_path, all_head_accs_np)
    top_heads = []

    top_accs = np.argsort(all_head_accs_np.reshape(num_heads*num_layers))[::-1][:num_to_intervene] # 排序后反转取索引
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]  # 准确率最高的层和head的索引
    if use_random_dir: 
        # overwrite top heads with random heads, no replacement
        random_idxs = np.random.choice(num_heads*num_layers, num_heads*num_layers, replace=False)
        top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in random_idxs[:num_to_intervene]]

    return top_heads, probes


def get_top_heads_by_pca_and_save_accs(experiments_path, fold, num_layers, num_heads, all_head_accs_np, num_to_intervene):
    save_path = f'{experiments_path}/fold_{fold}_pca_head_accs.npy'
    all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads)
    np.save(save_path, all_head_accs_np)
    top_heads = []

    top_accs = np.argsort(all_head_accs_np.reshape(num_heads*num_layers))[::-1][:num_to_intervene] # 排序后反转取索引
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]  # 准确率最高的层和head的索引

    return top_heads
    


def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--use_honest', action='store_true', help='use local editted version of the model', default=False)
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    parser.add_argument('--activations_dataset', type=str, default='tqa_gen_end_q', help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--n_components', type=int, default=1)
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--pure', action='store_true', default=False)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--direction_type', type=str, default='pca')
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--device', type=int, default=1, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--judge_name', type=str, required=False)
    parser.add_argument('--info_name', type=str, required=False)
    parser.add_argument('--collect', type=str, default='all')
    parser.add_argument('--cut_type', type=str, default='')
    parser.add_argument('--stimulus_pos', type=int, default='6')
    parser.add_argument('--cur_rate', type=float, default='1')
    parser.add_argument('--cut_random', action='store_true', help='Randomly truncate at answer position', default=False)
    parser.add_argument('--random_lower_bound', type=float, default='0.5')
    parser.add_argument('--choose_heads_by_pca', action='store_true', default=False)

    args = parser.parse_args()

    if args.pure:
        experiment_name = f'{args.model_name}_pure'
    elif args.collect == 'all':
        if args.cut_random:
            experiment_name = f'{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_random_lower_bound{str(int(args.random_lower_bound * 100)).zfill(3)}'
        else:
            experiment_name = f'{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_{str(int(args.cur_rate * 100)).zfill(3)}'
    elif args.collect == 'stimulus':
        experiment_name = f'{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_{str(args.stimulus_pos)}'
    else:
        experiment_name = f'{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_iti'
    if args.use_center_of_mass:
        experiment_name += f'_{args.direction_type}_alpha{int(args.alpha)}'
        if args.direction_type == 'pca':
            experiment_name += f'_n{args.n_components}'
            if args.choose_heads_by_pca:
                experiment_name += '_choose_by_pca'

    print('Running:\n{}\n'.format(' '.join(sys.argv)))
    print(args)
    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    df = pd.read_csv('./TruthfulQA/data/v0/TruthfulQA.csv')

    # order csv by huggingface order, the order used to save activations
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
    model_name = HF_NAMES["honest_" + args.model_name if args.use_honest else args.model_name]
    tokenizer = llama.LLaMATokenizer.from_pretrained(model_name)
    model = llama.LLaMAForCausalLM.from_pretrained(model_name, low_cpu_mem_usage = True, torch_dtype=torch.float16)
    r = model.to(args.device)
    device = args.device
    
    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # load activations 
    tokens = pkl.load(open(f"/data/wtl/honest_llm/activations/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_tokens.pkl", 'rb'))
    head_wise_activations = pkl.load(open(f"/data/wtl/honest_llm/activations/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_head_wise.pkl", 'rb'))
    labels = np.load(f"/data/wtl/honest_llm/activations/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_labels.npy")

    def find_ans_positions(tokens):
        positions = []
        tokens_len = []
        for token_list in tokens:
            for i in range(1, len(token_list)):
                if token_list[i] == ':' and token_list[i-1] == '▁A':
                    positions.append(i + 1)  # 将找到的位置添加到列表中
                    tokens_len.append(len(token_list) - (i + 1))
                    break  # 假设每个列表中只有一个满足条件的位置
        return positions, tokens_len
    
    if args.collect == 'all':
        ans_start_pos, ans_len = find_ans_positions(tokens)
        if args.cut_random:
            head_wise_activations = [activations[:, pos + int(l * random.uniform(args.random_lower_bound, 1)) - 1,:] for activations, pos, l in zip(head_wise_activations, ans_start_pos, ans_len)]    
            # np.save(f'/data/wtl/honest_llm/activations/llama_7B_tqa_mc2_random_from{str(int(args.random_lower_bound * 100)).zfill(3)}_head_wise.npy', head_wise_activations)
            # print('save success!')
            # exit(0)
        else:
            head_wise_activations = [activations[:, pos + int(l * args.cur_rate) - 1,:] for activations, pos, l in zip(head_wise_activations, ans_start_pos, ans_len)]
    elif args.collect == 'stimulus':
        head_wise_activations = [activations[:, args.stimulus_pos, :] for activations in head_wise_activations]
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)
    # separated_head_wise_activations: shape(question_nums, answer_nums, layer_nums, head_nums, 128)
    separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations)

    experiments_path = f'/data/wtl/honest_llm/cluster_probe_experiments/{experiment_name}'
    os.makedirs(experiments_path, exist_ok=True)
    print(f'experiments_path: {experiments_path}')

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

        # get directions
        if args.use_center_of_mass:
            if args.direction_type == 'mean':
                print('use mean direction')
                com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels)
            elif args.direction_type == 'pca':
                print('use pca direction') 
                com_directions, pca_accs = get_pca_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, n_components=args.n_components)
                print('top pca accs: ', sorted(pca_accs, reverse=True)[:5])
                print('bottom pca accs: ', sorted(pca_accs)[:5])
            elif args.direction_type == 'pca_mean':
                print('use pca_mean direction')
                mean_direction = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels)
                pca_directions, pca_accs = get_pca_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, n_components=args.n_components)
                com_directions = (mean_direction + pca_directions) / 2
        else:
            com_directions = None

        if args.choose_heads_by_pca:
            top_heads = get_top_heads_by_pca_and_save_accs(experiments_path, i, num_layers, num_heads, pca_accs, args.num_heads)
            probes = None
        else:
            top_heads, probes = get_top_heads_and_save_accs(experiments_path, i, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir)

        print("Heads intervened: ", sorted(top_heads))
    
        interventions = get_interventions_dict(top_heads, probes, head_wise_activations, num_heads, args.use_center_of_mass, args.use_random_dir, com_directions)

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

        if args.use_random_dir:
            filename += '_random'
        if args.use_honest:
            filename = 'honest_' + filename
                    
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
            info_name=args.info_name
        )

        print(f"FOLD {i}")
        print(curr_fold_results)

        curr_fold_results = curr_fold_results.to_numpy()[0].astype(float)
        results.append(curr_fold_results)
    
    results = np.array(results)
    final = results.mean(axis=0)

    print(f'BLEURT acc: {final[0]:.4f}, MC1: {final[1]:.4f}, MC2: {final[2]:.4f}, bleu acc: {final[3]:.4f}, rouge1 acc: {final[4]:.4f}, CE Loss: {final[5]:.4f}, KL wrt Original: {final[6]:.4f}')

if __name__ == "__main__":
    main()
