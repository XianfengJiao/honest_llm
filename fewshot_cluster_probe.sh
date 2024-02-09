#!/bin/bash


alpha=(10 12 15)
probe_base_weights=(0)
n_clusters=(3)
num_heads=(16 24 32)

for a in "${alpha[@]}"; do
    for num_head in "${num_heads[@]}"; do
        for weight in "${probe_base_weights[@]}"; do
            # 内循环遍历 n_clusters
            for cluster in "${n_clusters[@]}"; do
                # 显示正在执行的命令
                echo "Running: fewshot_llama_7B_cluster${cluster}_probe_heads${num_head}_alpha${a}_baseW${weight//.}_icl"
                nohup python -u fewshot_cluster_probe.py --method='icl' --device=0 --probe_type=prob --probe_base_weight="$weight" --n_clusters="$cluster" --num_heads="$num_head" --alpha="$a" > "./logs/fewshot_llama_7B_cluster${cluster}_probe_heads${num_head}_alpha${a}_baseW${weight//.}_prob_icl.log" 2>&1 &
                wait

                echo "Running: fewshot_llama_7B_cluster${cluster}_probe_heads${num_head}_alpha${a}_baseW${weight//.}_none"
                nohup python -u fewshot_cluster_probe.py --method='none' --device=0 --probe_type=prob --probe_base_weight="$weight" --n_clusters="$cluster" --num_heads="$num_head" --alpha="$a" > "./logs/fewshot_llama_7B_cluster${cluster}_probe_heads${num_head}_alpha${a}_baseW${weight//.}_prob_none.log" 2>&1 &
                wait
            done
        done
    done
done
