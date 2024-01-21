#!/bin/bash

alpha=(5 7 9 10)
probe_base_weights=(0)
n_clusters=(2)
num_heads=(8 10 12 14 16)
device=0

for a in "${alpha[@]}"; do
    for num_head in "${num_heads[@]}"; do
        for weight in "${probe_base_weights[@]}"; do
            # 内循环遍历 n_clusters
            for cluster in "${n_clusters[@]}"; do
                # 显示正在执行的命令
                echo "Running: nq_open_valid_1_fold_llama_7B_cluster${cluster}_probe_heads${num_head}_alpha${a}_baseW${weight//.}"

                # 构建并执行命令
                nohup python -u valid_cluster_probe_external.py --device=${device} --dataset_name=nq_open --probe_type=prob --probe_base_weight="$weight" --n_clusters="$cluster" --num_heads="$num_head" --alpha="$a" > "./logs/0117_nq_open_valid_1_fold_llama_7B_cluster${cluster}_probe_heads${num_head}_alpha${a}_baseW${weight//.}_prob.log" 2>&1 &
                
                # 等待上一个命令完成
                wait
            done
        done
    done
done
