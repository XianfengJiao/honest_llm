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
    parser.add_argument('--model_name', type=str, default='llama_7B')
    parser.add_argument('--dataset_name', type=str, default='tqa_mc2')
    parser.add_argument('--collect', type=str, default='cut')
    parser.add_argument('--cut_type', type=str, default='')
    parser.add_argument('--device', type=int, default=1)
    args = parser.parse_args()
    HF_NAMES = {
        # 'llama_7B': 'decapoda-research/llama-7b-hf',
        'llama_7B': 'yahma/llama-7b-hf',
        'alpaca_7B': 'circulus/alpaca-7b', 
        'vicuna_7B': 'AlekseyKorshuk/vicuna-7b', 
        # 'llama2_chat_7B': 'meta-llama/Llama-2-7b-chat-hf', 
        'llama2_chat_7B': 'daryl149/llama-2-7b-chat-hf',
    }

    MODEL = HF_NAMES[args.model_name]

    tokenizer = llama.LLaMATokenizer.from_pretrained(MODEL)
    model = llama.LLaMAForCausalLM.from_pretrained(MODEL, low_cpu_mem_usage=True, torch_dtype=torch.float16, device_map=args.device)
    device = args.device
    # device = "cuda"
    r = model.to(device)

    if args.dataset_name == "tqa_mc2": 
        dataset = load_dataset("truthful_qa", "multiple_choice")['validation']
        if args.collect == 'all':
            formatter = tokenized_tqa_all
        elif args.collect == 'stimulus':
            formatter = tokenized_tqa_stimulus
        elif args.collect == 'cut':
            if args.cut_type == 'random':
                pos_fn = lambda x: random.randint(1,len(x))
            elif args.cut_type == '05':
                pos_fn = lambda x: random.randint(1,len(x)//2)
            formatter = partial(tokenized_tqa_cut, pos=pos_fn)
    elif args.dataset_name == "tqa_gen": 
        dataset = load_dataset("truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen
    elif args.dataset_name == 'tqa_gen_end_q': 
        dataset = load_dataset("truthful_qa", 'generation')['validation']
        formatter = tokenized_tqa_gen_end_q
    else:
        raise ValueError("Invalid dataset name")

    print("Tokenizing prompts")
    if args.dataset_name == "tqa_gen" or args.dataset_name == "tqa_gen_end_q": 
        prompts, labels, categories = formatter(dataset, tokenizer)
        with open(f'/data1/jxf/activations/{args.model_name}_{args.dataset_name}_{args.collect}_categories.pkl', 'wb') as f:
            pickle.dump(categories, f)
    else: 
        prompts, labels, tokens = formatter(dataset, tokenizer)

    all_layer_wise_activations = []
    all_head_wise_activations = []

    print("Getting activations")
    for prompt in tqdm(prompts):
        layer_wise_activations, head_wise_activations, _ = get_llama_activations_bau(model, prompt, device)
        if args.collect == 'stimulus':
            all_layer_wise_activations.append(layer_wise_activations[:,[5,6,7,-4,-3,-2,-1],:])
            all_head_wise_activations.append(head_wise_activations[:,[5,6,7,-4,-3,-2,-1],:])
        elif args.collect == 'cut':
            all_layer_wise_activations.append(layer_wise_activations[:, -1, :])
            all_head_wise_activations.append(head_wise_activations[:, -1, :])
        elif args.collect == 'all':
            all_layer_wise_activations.append(layer_wise_activations)
            all_head_wise_activations.append(head_wise_activations)

    print("Saving labels")
    np.save(f'/data1/jxf/activations/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_labels.npy', labels)
    
    print("Saving tokens")
    pickle.dump(tokens, open(f'/data1/jxf/activations/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_tokens.pkl', 'wb'))

    print("Saving layer wise activations")
    pickle.dump(all_layer_wise_activations, open(f'/data1/jxf/activations/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_layer_wise.pkl', 'wb'))
    # np.save(f'features/all_activations/{args.model_name}_{args.dataset_name}_layer_wise.npy', all_layer_wise_activations)
    
    print("Saving head wise activations")
    pickle.dump(all_head_wise_activations, open(f'/data1/jxf/activations/{args.model_name}_{args.dataset_name}_{args.collect}{args.cut_type}_head_wise.pkl', 'wb'))
    # np.save(f'features/all_activations/{args.model_name}_{args.dataset_name}_head_wise.npy', all_head_wise_activations)

if __name__ == '__main__':
    main()