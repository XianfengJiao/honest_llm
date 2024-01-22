import torch
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
from utils import *
import llama

HF_NAMES = {
    'llama_7B': 'yahma/llama-7b-hf',
    'llama2_7B': 'meta-llama/Llama-2-7b-hf', 
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'alpaca_7B': 'circulus/alpaca-7b', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b'
}

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=5, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--pure', action='store_true', default=False)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--use_pca_dir', action='store_true', help='use pca direction', default=False)
    parser.add_argument('--device', type=int, default=1, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--judge_name', type=str, required=False)
    parser.add_argument('--info_name', type=str, required=False)
    args = parser.parse_args()
    
    print('Running:\n{}\n'.format(' '.join(sys.argv)))
    print(args)

    if args.pure:
        experiment_name = f'{args.model_name}_pure'
    elif args.use_center_of_mass:
        experiment_name = f'{args.model_name}_com_heads{args.num_heads}_alpha{args.alpha}'
    elif args.use_random_dir:
        experiment_name = f'{args.model_name}_random_dir_heads{args.num_heads}_alpha{args.alpha}'
    elif args.use_pca_dir:
        experiment_name = f'{args.model_name}_pca_dir_heads{args.num_heads}_alpha{args.alpha}'

    experiments_path = f'/data/wtl/honest_llm/baseline/{experiment_name}'
    os.makedirs(experiments_path, exist_ok=True)
    print(f'experiments_path: {experiments_path}')

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
    model_name = HF_NAMES[args.model_name]
    tokenizer = llama.LLaMATokenizer.from_pretrained(model_name)
    model = llama.LLaMAForCausalLM.from_pretrained(model_name, low_cpu_mem_usage = True, torch_dtype=torch.float16)
    r = model.to(args.device)
    device = args.device
    
    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # load activations 
    head_wise_activations = pkl.load(open(f'/data/wtl/honest_llm/activations/{args.model_name}_{args.dataset_name}_all_100_head_wise.pkl', 'rb'))
    labels = np.load(f'/data/wtl/honest_llm/activations/{args.model_name}_{args.dataset_name}_all_100_labels.npy')
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)


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

        # get directions
        if args.use_center_of_mass:
            print("Using center of mass directions")
            com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels)
        elif args.use_random_dir:
            print("Using random directions")
            com_directions = None
        elif args.use_pca_dir:
            print("Using PCA directions")
            com_directions, pca_accs = get_pca_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, n_components=1)
        else:
            com_directions = None
        top_heads, probes = get_top_heads(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, args.use_random_dir)

        print("Heads intervened: ", sorted(top_heads))
    
        interventions = get_baseline_interventions_dict(top_heads, probes, head_wise_activations, num_heads, args.use_center_of_mass, args.use_random_dir, com_directions)

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

        filename = f'{args.model_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{args.alpha}_fold_{i}'

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
    
    print(f'MC1: {final[1]:.3f}, MC2: {final[2]:.3f}, BLEURT acc: {final[0]:.3f}, bleu acc: {final[3]:.3f}, rouge1 acc: {final[4]:.3f}, CE Loss: {final[5]:.3f}, KL wrt Original: {final[6]:.3f}')
    


if __name__ == "__main__":
    main()
