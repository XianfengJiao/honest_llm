import logging

logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

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
from utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions
import llama

HF_NAMES = {
    'llama_7B': 'yahma/llama-7b-hf',
    'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
}

def get_top_heads_from_probes_and_accs(probes, all_head_accs_np, num_layers, num_heads, num_to_intervene, use_random_dir=False):
    top_heads = []

    top_accs = np.argsort(all_head_accs_np.reshape(num_heads*num_layers))[::-1][:num_to_intervene] # 排序后反转取索引
    top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in top_accs]  # 准确率最高的层和head的索引
    if use_random_dir: 
        # overwrite top heads with random heads, no replacement
        random_idxs = np.random.choice(num_heads*num_layers, num_heads*num_layers, replace=False)
        top_heads = [flattened_idx_to_layer_head(idx, num_heads) for idx in random_idxs[:num_to_intervene]]

    return top_heads, probes



def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--use_honest', action='store_true', help='use local editted version of the model', default=False)
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    parser.add_argument('--activations_dataset', type=str, default='tqa_gen_end_q', help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--device', type=int, default=1, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--judge_name', type=str, required=False)
    parser.add_argument('--info_name', type=str, required=False)
    parser.add_argument('--collect', type=str, default='stimulus')
    parser.add_argument('--cut_type', type=str, default='')
    parser.add_argument('--stimulus_pos', type=int, default='6')
    parser.add_argument('--cur_rate', type=float, default='1')
    args = parser.parse_args()

    # Add file logging besides stdout
    if args.collect == 'all':
        experiment_name = f'{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_{str(int(args.cur_rate * 100)).zfill(3)}'
    elif args.collect == 'stimulus':
        experiment_name = f'{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_{str(args.stimulus_pos)}'
    else:
        experiment_name = f'{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}'
    
    os.makedirs(f'/data/wtl/honest_llm/validation/{experiment_name}',exist_ok=True)
    log_path = f'logs/{experiment_name}.log'
    file_handler = logging.FileHandler(log_path)
    logger.addHandler(file_handler)

    logger.info('Running:\n{}\n'.format(' '.join(sys.argv))) # command

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    df = pd.read_csv('TruthfulQA/TruthfulQA.csv')

    # order csv by huggingface order, the order used to save activations
    dataset = load_dataset("truthful_qa", "multiple_choice")['validation']
    golden_q_order = list(dataset["question"])
    df = df.sort_values(by='Question', key=lambda x: x.map({k: i for i, k in enumerate(golden_q_order)}))
    
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
    tokens = pkl.load(open(f"/data/jxf/activations/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_tokens.pkl", 'rb'))
    head_wise_activations = pkl.load(open(f"/data/jxf/activations/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_head_wise.pkl", 'rb'))
    labels = np.load(f"/data/jxf/activations/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_labels.npy")

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
        head_wise_activations = [activations[:, pos + int(l * args.cur_rate) - 1,:] for activations, pos, l in zip(head_wise_activations, ans_start_pos, ans_len)]
    elif args.collect == 'stimulus':
        head_wise_activations = [activations[:, args.stimulus_pos, :] for activations in head_wise_activations]
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)


    # load probes and accs
    if args.collect == 'all':
        probes = pkl.load(open(f'/data/jxf/probes/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_{str(int(args.cur_rate * 100)).zfill(3)}_probes.pkl', 'rb'))
        all_head_accs_np = pkl.load(open(f'/data/jxf/probes/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_{str(int(args.cur_rate * 100)).zfill(3)}_head_accs.pkl', 'rb'))
    elif args.collect == 'stimulus':
        probes = pkl.load(open(f'/data/jxf/probes/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_{str(args.stimulus_pos)}_probes.pkl', 'rb'))
        all_head_accs_np = pkl.load(open(f'/data/jxf/probes/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_{str(args.stimulus_pos)}_head_accs.pkl', 'rb'))
    else:
        probes = pkl.load(open(f'/data/jxf/probes/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_probes.pkl', 'rb'))
        all_head_accs_np = pkl.load(open(f'/data/jxf/probes/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_head_accs.pkl', 'rb'))

    separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations)

    # run k-fold cross validation
    results = []
    experiments_path = f'/data/wtl/honest_llm/validation/'
    for i in range(args.num_fold):

        train_idxs = np.concatenate([fold_idxs[j] for j in range(args.num_fold) if j != i])
        test_idxs = fold_idxs[i]

        print(f"Running fold {i}")

        # pick a val set using numpy
        train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
        val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])

        # save train and test splits
        df.iloc[train_set_idxs].to_csv(f"/data/wtl/honest_llm/validation/{experiment_name}/fold_{i}_train_seed_{args.seed}.csv", index=False)
        df.iloc[val_set_idxs].to_csv(f"/data/wtl/honest_llm/validation/{experiment_name}/fold_{i}_val_seed_{args.seed}.csv", index=False)
        df.iloc[test_idxs].to_csv(f"/data/wtl/honest_llm/validation/{experiment_name}/fold_{i}_test_seed_{args.seed}.csv", index=False)

        # get directions
        if args.use_center_of_mass:
            com_directions = get_com_directions(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels)
        else:
            com_directions = None
        top_heads, probes = get_top_heads_from_probes_and_accs(probes, all_head_accs_np, num_layers, num_heads, args.num_heads, args.use_random_dir)

        logger.info("Heads intervened: ", sorted(top_heads))
    
        interventions = get_interventions_dict(top_heads, probes, head_wise_activations, num_heads, args.use_center_of_mass, args.use_random_dir, com_directions)

        # edit model
        for head_out_name, list_int_vec in interventions.items():
            layer_no = int(head_out_name.split('.')[2])
            displacement = np.zeros((int(num_heads), int(model.config.hidden_size / num_heads)))
            for head_no, head_vec, std in list_int_vec:
                displacement[head_no] = args.alpha * std * head_vec
            displacement = torch.tensor(rearrange(displacement, 'h d -> (h d)'), device=device)
            bias_tobe = F.linear(displacement.to(torch.float16), model.model.layers[layer_no].self_attn.o_proj.weight).to(device)
            model.model.layers[layer_no].self_attn.o_proj.bias = torch.nn.parameter.Parameter(bias_tobe)


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

        if args.use_center_of_mass:
            filename += '_com'
        if args.use_random_dir:
            filename += '_random'
        if args.use_honest:
            filename = 'honest_' + filename
                    
        curr_fold_results = alt_tqa_evaluate(
            {args.model_name: model}, 
            ['judge', 'info', 'mc'], 
            f'/data/wtl/honest_llm/validation/{experiment_name}/fold_{i}_test_seed_{args.seed}.csv', 
            f'results_dump/answer_dump/{filename}.csv', 
            f'results_dump/summary_dump/{filename}.csv', 
            device=args.device, 
            interventions=interventions, 
            intervention_fn=lt_modulated_vector_add, 
            judge_name=args.judge_name, 
            info_name=args.info_name
        )

        print(f"FOLD {i}")
        print(curr_fold_results)

        curr_fold_results = curr_fold_results.to_numpy()[0].astype(float)
        results.append(curr_fold_results)
    
    results = np.array(results)
    final = results.mean(axis=0)

    # print(f'True*Info Score: {final[1]*final[0]}, True Score: {final[1]}, Info Score: {final[0]}, MC1 Score: {final[2]}, MC2 Score: {final[3]}, CE Loss: {final[4]}, KL wrt Original: {final[5]}')

if __name__ == "__main__":
    main()
