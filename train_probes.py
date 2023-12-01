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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


import sys
from utils import alt_tqa_evaluate, flattened_idx_to_layer_head, layer_head_to_flattened_idx, get_interventions_dict, get_top_heads, get_separated_activations, get_com_directions
import llama

HF_NAMES = {
    # 'llama_7B': 'decapoda-research/llama-7b-hf',
    'llama_7B': 'yahma/llama-7b-hf',
    'honest_llama_7B': 'results_dump/llama_7B_seed_42_top_48_heads_alpha_15', 
    'alpaca_7B': 'circulus/alpaca-7b', 
    'honest_alpaca_7B': 'results_dump/alpaca_7B_seed_42_top_48_heads_alpha_15', 
    'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
    'honest_vicuna_7B': 'results_dump/vicuna_7B_seed_42_top_48_heads_alpha_15', 
    # 'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
    'llama2_chat_7B': 'daryl149/llama-2-7b-chat-hf',
    'honest_llama2_chat_7B': 'results_dump/llama2_chat_7B_seed_42_top_48_heads_alpha_15', 
}

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='llama_7B', choices=HF_NAMES.keys(), help='model name')
    parser.add_argument('--use_honest', action='store_true', help='use local editted version of the model', default=False)
    parser.add_argument('--collect', type=str, default='stimulus')
    parser.add_argument('--stimulus_pos', type=int, default='6')
    parser.add_argument('--cur_rate', type=float, default='1')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2', help='feature bank for training probes')
    parser.add_argument('--activations_dataset', type=str, default='tqa_gen_end_q', help='feature bank for calculating std along direction')
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument("--num_fold", type=int, default=2, help="number of folds")
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.2)
    parser.add_argument('--use_center_of_mass', action='store_true', help='use center of mass direction', default=False)
    parser.add_argument('--use_random_dir', action='store_true', help='use random direction', default=False)
    parser.add_argument('--device', type=int, default=3, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--judge_name', type=str, required=False)
    parser.add_argument('--info_name', type=str, required=False)
    args = parser.parse_args()

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    df = pd.read_csv('./TruthfulQA/TruthfulQA.csv')
    
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
    
    device = args.device
    r = model.to(device)
    
    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    
    # load activations 
    head_wise_activations = pkl.load(open(f"/data1/jxf/activations/{args.model_name}_{args.dataset_name}_{args.collect}_head_wise.pkl", 'rb'))
    labels = np.load(f"/data1/jxf/activations/{args.model_name}_{args.dataset_name}_{args.collect}_labels.npy")
    tokens = pkl.load(open(f"/data1/jxf/activations/{args.model_name}_{args.dataset_name}_{args.collect}_tokens.pkl", 'rb'))
    
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
    else:
        head_wise_activations = [activations[:, args.stimulus_pos, :] for activations in head_wise_activations]
        
    
    
    head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)

    separated_head_wise_activations, separated_labels, idxs_to_split_at = get_separated_activations(labels, head_wise_activations)
    
    all_head_accs = []
    probes = []
    
    train_idxs = np.arange(len(df))
    train_set_idxs = np.random.choice(train_idxs, size=int(len(train_idxs)*(1-args.val_ratio)), replace=False)
    val_set_idxs = np.array([x for x in train_idxs if x not in train_set_idxs])
        
    all_X_train = np.concatenate([separated_head_wise_activations[i] for i in train_set_idxs], axis = 0)
    all_X_val = np.concatenate([separated_head_wise_activations[i] for i in val_set_idxs], axis = 0)
    y_train = np.concatenate([separated_labels[i] for i in train_set_idxs], axis = 0)
    y_val = np.concatenate([separated_labels[i] for i in val_set_idxs], axis = 0)
    
    for layer in tqdm(range(num_layers)): 
        for head in range(num_heads): 
            X_train = all_X_train[:,layer,head,:]
            X_val = all_X_val[:,layer,head,:]

            clf = LogisticRegression(random_state=args.seed, max_iter=1000).fit(X_train, y_train)
            y_pred = clf.predict(X_train)
            y_val_pred = clf.predict(X_val)
            all_head_accs.append(accuracy_score(y_val, y_val_pred))
            probes.append(clf)

    all_head_accs_np = np.array(all_head_accs)
    all_head_accs_np = all_head_accs_np.reshape(num_layers, num_heads)
    if args.collect == 'all':
        pkl.dump(probes, open(f'/data1/jxf/probes/{args.model_name}_{args.dataset_name}_{args.collect}_{str(int(args.cur_rate * 100)).zfill(3)}_probes.pkl', 'wb'))
        pkl.dump(all_head_accs_np, open(f'/data1/jxf/probes/{args.model_name}_{args.dataset_name}_{args.collect}_{str(int(args.cur_rate * 100)).zfill(3)}_head_accs.pkl', 'wb'))
    elif args.collect == 'stimulus':
        pkl.dump(probes, open(f'/data1/jxf/probes/{args.model_name}_{args.dataset_name}_{args.collect}_{str(args.stimulus_pos)}_probes.pkl', 'wb'))
        pkl.dump(all_head_accs_np, open(f'/data1/jxf/probes/{args.model_name}_{args.dataset_name}_{args.collect}_{str(args.stimulus_pos)}_head_accs.pkl', 'wb'))
    
    
if __name__ == "__main__":
    main()
    