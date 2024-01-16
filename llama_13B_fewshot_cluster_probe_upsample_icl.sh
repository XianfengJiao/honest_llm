#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3

alpha=(15)
probe_base_weights=(0.1 0.2 0.3 0.4 0.5)
n_clusters=(2)
num_heads=(24)
cut_rates=(0.9)

echo "Running: fewshot_llama_13B_pure_icl"
nohup python -u fewshot_cluster_probe.py --model_name='llama_13B' --method='icl' --pure > "./logs/fewshot_llama_13B_pure_icl.log" 2>&1 &
wait

for c in "${cut_rates[@]}"; do
    for a in "${alpha[@]}"; do
        for num_head in "${num_heads[@]}"; do
            for weight in "${probe_base_weights[@]}"; do
                # 内循环遍历 n_clusters
                for cluster in "${n_clusters[@]}"; do
                    # 显示正在执行的命令
                    echo "Running: fewshot_llama_13B_cluster${cluster}_probe_cut${c}_heads${num_head}_alpha${a}_baseW${weight//.}_icl"
                    nohup python -u fewshot_cluster_probe_upsample.py --model_name='llama_13B' --method='icl' --probe_type=prob --probe_base_weight="$weight" --n_clusters="$cluster" --num_heads="$num_head" --alpha="$a" --cut_rate="$c" > "./logs/fewshot_llama_13B_cluster${cluster}_probe_cut${c}_heads${num_head}_alpha${a}_baseW${weight//.}_prob_icl.log" 2>&1 &
                    wait

                    # echo "Running: fewshot_llama_7B_cluster${cluster}_probe_cut${c}_heads${num_head}_alpha${a}_baseW${weight//.}_none"
                    # nohup python -u fewshot_cluster_probe_upsample.py --method='none' --device=0 --probe_type=prob --probe_base_weight="$weight" --n_clusters="$cluster" --num_heads="$num_head" --alpha="$a" --cut_rate="$c" > "./logs/fewshot_llama_7B_cluster${cluster}_probe_cut${c}_heads${num_head}_alpha${a}_baseW${weight//.}_prob_none.log" 2>&1 &
                    # wait
                done
            done
        done
    done
done