#!/bin/bash

# 定义 probe_base_weight 和 n_clusters 的数组
alpha=(10 13 14 15 17)
probe_base_weights=(0)
n_clusters=(2 3 4)
num_heads=(16 24 32 48)
device=1

# 外循环遍历 probe_base_weight

nohup python -u valid_cluster_probe_external.py --device=${device} --dataset_name=nq_open --pure > "./logs/nq_open_valid_1_fold_llama_7B_pure.log" 2>&1 &
wait

for a in "${alpha[@]}"; do
    for num_head in "${num_heads[@]}"; do
        for weight in "${probe_base_weights[@]}"; do
            # 内循环遍历 n_clusters
            for cluster in "${n_clusters[@]}"; do
                # 显示正在执行的命令
                echo "Running: nq_open_valid_1_fold_llama_7B_cluster${cluster}_probe_heads${num_head}_alpha${a}_baseW${weight//.}"

                # 构建并执行命令
                nohup python -u valid_cluster_probe_external.py --device=${device} --dataset_name=nq_open --probe_type=prob --probe_base_weight="$weight" --n_clusters="$cluster" --num_heads="$num_head" --alpha="$a" > "./logs/nq_open_valid_1_fold_llama_7B_cluster${cluster}_probe_heads${num_head}_alpha${a}_baseW${weight//.}_prob.log" 2>&1 &
                
                # 等待上一个命令完成
                wait
            done
        done
    done
done
