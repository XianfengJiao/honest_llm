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
    parser.add_argument('--num_heads', type=int, default=48, help='K, number of top heads to intervene on')
    parser.add_argument('--alpha', type=float, default=15, help='alpha, intervention strength')
    parser.add_argument('--cut_rate', type=float, default=0.9)
    parser.add_argument('--probe_base_weight', type=float, default=0.5)
    parser.add_argument('--probe_type', type=str, default='prob')
    parser.add_argument('--method', type=str, default='icl')
    parser.add_argument("--num_fold", type=int, default=1, help="number of folds")
    parser.add_argument('--fewshot_ratio', type=float, default=0.053)
    parser.add_argument('--val_ratio', type=float, help='ratio of validation set size to development set size', default=0.05)
    parser.add_argument('--device', type=int, default=3, help='device')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--n_clusters', type=int, default=3)
    parser.add_argument('--judge_name', type=str, default='ft:davinci-002:university-of-edinburgh::8ejp8D64')
    parser.add_argument('--info_name', type=str, default='ft:davinci-002:university-of-edinburgh:info:8ejuTaQe')
    args = parser.parse_args()

    print('Running:\n{}\n'.format(' '.join(sys.argv)))
    print(args)

    experiment_name = f'fewshot_{args.model_name}_cluster_probe_upsample_num_heads{args.num_heads}_alpha{args.alpha}_n_clusters{args.n_clusters}_baseW{args.probe_base_weight}_cut{args.cut_rate}_{args.probe_type}_{args.method}'
    experiments_path = f'/data/jxf/honest_llm/cluster_experiments/{experiment_name}'
    os.makedirs(experiments_path, exist_ok=True)
    print(f'experiments_path: {experiments_path}')

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # load dataframe and activations direcitons
    df = pd.read_csv('./TruthfulQA/data/v0/TruthfulQA.csv')
    head_wise_activation_directions = np.load(f'/data/jxf/honest_llm/directions/{args.model_name}_head_wise_activation_directions.npy')
    # norms = np.linalg.norm(head_wise_activation_directions, axis=-1, keepdims=True)
    # # 避免除以零的情况
    # norms[norms == 0] = np.inf  # 将为0范数设置为无穷大，这样除以无穷大会得到0
    # head_wise_activation_directions = head_wise_activation_directions / norms


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
    model = llama.LLaMAForCausalLM.from_pretrained(model_name, low_cpu_mem_usage = True, torch_dtype=torch.float16, device_map="auto")
    # r = model.to(args.device)
    device = args.device
    
    # define number of layers and heads
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    # load activations 
    head_wise_activations = pkl.load(open(f'/data/jxf/activations/{args.model_name}_llama_7B_tqa_mc2_all_head_wise.pkl', 'rb'))
    labels = np.load(f'/data/jxf/activations/{args.model_name}_llama_7B_tqa_mc2_all_labels.npy')
    # head_wise_activations = rearrange(head_wise_activations, 'b l (h d) -> b l h d', h = num_heads)

    # separated_head_wise_activations: shape(question_nums, answer_nums, layer_nums, head_nums, 128)
    separated_head_wise_activations, separated_labels, _ = get_separated_upsample_activations(labels, head_wise_activations, args.cut_rate, num_heads)
    

    # run k-fold cross validation
    results = []
    for i in range(args.num_fold):

        all_idxs = fold_idxs[i]

        print(f"Running fold {i}")

        # pick a val set using numpy
        val_set_idxs = np.random.choice(all_idxs, size=int(len(all_idxs)*(args.val_ratio)), replace=False)
        test_set_idxs = np.array([x for x in all_idxs if x not in val_set_idxs])

        train_set_idxs = np.random.choice(test_set_idxs, size=int(len(test_set_idxs)*(args.fewshot_ratio)), replace=False)
        test_set_idxs = np.array([x for x in test_set_idxs if x not in train_set_idxs])
        # test_set_idxs = val_set_idxs

        many_shot_prefix = None
        if args.method == 'icl':
            many_shot_prefix = fewshot_eqipped(df.iloc[train_set_idxs])

        # save train and test splits
        df.iloc[train_set_idxs].to_csv(f"{experiments_path}/fold_{i}_train_seed_{args.seed}.csv", index=False)
        df.iloc[val_set_idxs].to_csv(f"{experiments_path}/fold_{i}_val_seed_{args.seed}.csv", index=False)
        df.iloc[test_set_idxs].to_csv(f"{experiments_path}/fold_{i}_test_seed_{args.seed}.csv", index=False)

        # get direction of cluster center
        cluster_idxs = get_cluster_idxs(num_layers, num_heads, train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, n_clusters=args.n_clusters, directions=head_wise_activation_directions)

        top_heads, probes = get_top_heads_cluster(train_set_idxs, val_set_idxs, separated_head_wise_activations, separated_labels, num_layers, num_heads, args.seed, args.num_heads, cluster_idxs, use_random_dir=False)
        # print("Heads intervened: ", sorted(top_heads))

        tuning_activations = np.array([token_ac for ans_ac in separated_head_wise_activations for token_ac in ans_ac])
        interventions = get_cluster_probe_interventions_dict(top_heads, probes, tuning_activations, num_heads, use_center_of_mass=True, use_random_dir=None, com_directions=None)

        # sample_directions
        sample_directions = head_wise_activation_directions[test_set_idxs]

        if args.probe_type == 'prob':
            def lt_modulated_cluster_probe_add(head_output, layer_name, start_edit_location='lt'):
                head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
                for head, direction, proj_val_std, probe in interventions[layer_name]:
                    direction_to_add = torch.tensor(direction).to(head_output.device.index)
                    if args.probe_base_weight == -1:
                        weight = 1
                    else:
                        weight = 1 + args.probe_base_weight - probe.predict_proba(head_output[:, -1, head, :].detach().cpu().numpy())[0][1]

                    if start_edit_location == 'lt': 
                        head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add * weight
                    else: 
                        head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add * weight
                    
                head_output = rearrange(head_output, 'b s h d -> b s (h d)')
                return head_output
        else:
            def lt_modulated_cluster_probe_add(head_output, layer_name, start_edit_location='lt'):
                head_output = rearrange(head_output, 'b s (h d) -> b s h d', h=num_heads)
                for head, direction, proj_val_std, probe in interventions[layer_name]:
                    direction_to_add = torch.tensor(direction).to(head_output.device.index)
                    if args.probe_base_weight == -1:
                        weight = 1
                    else:
                        weight = 1 + args.probe_base_weight - probe.predict(head_output[:, -1, head, :].detach().cpu().numpy())[0]

                    if start_edit_location == 'lt': 
                        head_output[:, -1, head, :] += args.alpha * proj_val_std * direction_to_add * weight
                    else: 
                        head_output[:, start_edit_location:, head, :] += args.alpha * proj_val_std * direction_to_add * weight
                    
                head_output = rearrange(head_output, 'b s h d -> b s (h d)')
                return head_output

        filename = f'{args.model_name}_seed_{args.seed}_top_{args.num_heads}_heads_alpha_{int(args.alpha)}_fold_{i}'
                    
        curr_fold_results = alt_tqa_evaluate(
            {args.model_name: model}, 
            ['mc','bleu','bleurt', 'judge', 'info'], 
            f'{experiments_path}/fold_{i}_test_seed_{args.seed}.csv', 
            f'{experiments_path}/answer_{filename}.csv', 
            f'{experiments_path}/summary_{filename}.csv', 
            device="cuda", 
            interventions=interventions, 
            intervention_fn=lt_modulated_cluster_probe_add, 
            judge_name=args.judge_name, 
            info_name=args.info_name,
            use_cluster=False,
            sample_directions = sample_directions,
            many_shot_prefix = many_shot_prefix,
        )

        print(f"FOLD {i}")
    
        curr_fold_results = curr_fold_results.to_numpy()[0].astype(float)
        results.append(curr_fold_results)
        print(curr_fold_results)
    
    results = np.array(results)
    final = results.mean(axis=0)

    print(f'True*Info Score: {final[1]*final[2]}, True Score: {final[2]}, Info Score: {final[1]}, BLEURT acc: {final[0]:.4f}, MC1: {final[3]:.4f}, MC2: {final[4]:.4f}, bleu acc: {final[5]:.4f}, rouge1 acc: {final[6]:.4f}, CE Loss: {final[7]}, KL wrt Original: {final[8]}')

    # print(f'True*Info Score: {final[1]*final[0]}, True Score: {final[1]}, Info Score: {final[0]}, MC1 Score: {final[2]}, MC2 Score: {final[3]}, CE Loss: {final[4]}, KL wrt Original: {final[5]}')

if __name__ == "__main__":
    main()
