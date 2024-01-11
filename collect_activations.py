import os
import torch
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pickle
from utils import *
import llama
import pickle
import argparse
import random
from functools import partial


def main(): 
    """
    Specify dataset name as the first command line argument. Current options are 
    "tqa_mc2", "piqa", "rte", "boolq", "copa". Gets activations for all prompts in the 
    validation set for the specified dataset on the last token for llama-7B. 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama2_7B')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--collect', type=str, default='all_100')
    parser.add_argument('--cut_type', type=str, default='')
    parser.add_argument('--device', type=int, default=3)
    args = parser.parse_args()
    HF_NAMES = {
        'llama_7B': 'yahma/llama-7b-hf',
        'llama2_7B': 'meta-llama/Llama-2-7b-hf', 
        'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
        'alpaca_7B': 'circulus/alpaca-7b', 
        'vicuna_7B': 'AlekseyKorshuk/vicuna-7b'
    }

    MODEL = HF_NAMES[args.model_name]

    tokenizer = llama.LLaMATokenizer.from_pretrained(MODEL)
    model = llama.LLaMAForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map=args.device)
    device = args.device
    r = model.to(device)

    if args.dataset_name == "tqa_mc2":
        url = "https://huggingface.co/api/datasets/truthful_qa/parquet/multiple_choice/validation/0.parquet"
        dataset = load_dataset('parquet', data_files=url)['train'] 
        # dataset = load_dataset("truthful_qa", "multiple_choice")['validation']
        ref_df = pd.read_csv('/data/wtl/code/honest_llm/TruthfulQA/data/v0/TruthfulQA.csv')
        if 'all' in args.collect:
            formatter = tokenized_tqa_all
        elif 'stimulus' in args.collect:  # 加入stimulus模板
            formatter = tokenized_tqa_stimulus
        elif args.collect == 'cut':  # 截断
            if args.cut_type == 'random':
                pos_fn = lambda x: random.randint(1,len(x))
            elif args.cut_type == '05':
                pos_fn = lambda x: len(x)//2
            else:
                pos_fn = lambda x: len(x)
            formatter = partial(tokenized_tqa_cut, pos=pos_fn)
        formatter = partial(formatter, ref_df=ref_df)
    elif args.dataset_name == "tqa_gen": 
        dataset = load_dataset("truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen
    elif args.dataset_name == 'tqa_gen_end_q': 
        dataset = load_dataset("truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen_end_q
    else:
        raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    prompts, labels, categories, tokens = formatter(dataset, tokenizer)

    # with open(f'/data/jxf/activations/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_categories.pkl', 'wb') as f:
    #     pickle.dump(categories, f)
    #     exit(0)

    all_layer_wise_activations = []
    all_head_wise_activations = []

    print("Getting activations")
    for prompt, token in tqdm(zip(prompts, tokens), total=len(prompts)):
        # layer_wise_activations (33, 42, 4096) num_hidden_layers + last, seq_len, hidden_size
        # head_wise_activations (32, 42, 4096) num_hidden_layers, seq_len, hidden_size
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_bau(model, prompt, device)
        if 'ans' in args.collect:
            pos = get_ans_pos(token)
            layer_wise_activations = layer_wise_activations[:, pos:, :]
            head_wise_activations = head_wise_activations[:, pos:, :]
        if 'mean' in args.collect:
            all_layer_wise_activations.append(np.mean(layer_wise_activations, axis=1))
            all_head_wise_activations.append(np.mean(head_wise_activations, axis=1))
        elif args.collect == 'stimulus':
            all_layer_wise_activations.append(layer_wise_activations[:,[5,6,7,-4,-3,-2,-1],:])
            all_head_wise_activations.append(head_wise_activations[:,[5,6,7,-4,-3,-2,-1],:])
        elif args.collect == 'cut':
            all_layer_wise_activations.append(layer_wise_activations[:, -1, :])
            all_head_wise_activations.append(head_wise_activations[:, -1, :])
        elif args.collect == 'all':
            all_layer_wise_activations.append(layer_wise_activations)
            all_head_wise_activations.append(head_wise_activations)
        elif args.collect == 'all_100':
            print("collect type: all_100")
            all_layer_wise_activations.append(layer_wise_activations[:, -1, :])
            all_head_wise_activations.append(head_wise_activations[:, -1, :])

    print("saving categories")
    pickle.dump(categories, open(f'/data/wtl/honest_llm/activations/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_categories.pkl', 'wb'))

    print("Saving labels")
    np.save(f'/data/wtl/honest_llm/activations/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_labels.npy', labels)
    
    print("Saving tokens")
    pickle.dump(tokens, open(f'/data/wtl/honest_llm/activations/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_tokens.pkl', 'wb'))

    print("Saving layer wise activations")
    pickle.dump(all_layer_wise_activations, open(f'/data/wtl/honest_llm/activations/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_layer_wise.pkl', 'wb'))
    # np.save(f'features/all_activations/{args.model_name}_{args.dataset_name}_layer_wise.npy', all_layer_wise_activations)
    
    print("Saving head wise activations")
    pickle.dump(all_head_wise_activations, open(f'/data/wtl/honest_llm/activations/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_head_wise.pkl', 'wb'))
    # np.save(f'features/all_activations/{args.model_name}_{args.dataset_name}_head_wise.npy', all_head_wise_activations)

    print("All saved successfully")


if __name__ == '__main__':
    main()