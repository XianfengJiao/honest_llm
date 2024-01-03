#!/bin/bash

# 定义 probe_base_weight 和 n_clusters 的数组
alpha=(15 10 5)
probe_base_weights=(0 0.5)
n_clusters=(3)
num_heads=(16 24)

# 外循环遍历 probe_base_weight

for a in "${alpha[@]}"; do
    for num_head in "${num_heads[@]}"; do
        for weight in "${probe_base_weights[@]}"; do
            # 内循环遍历 n_clusters
            for cluster in "${n_clusters[@]}"; do
                # 显示正在执行的命令
                echo "Running: valid_2_fold_llama_7B_cluster${cluster}_probe_heads${num_head}_alpha${a}_baseW${weight//.}"

                # 构建并执行命令
                nohup python -u valid_2_fold_cluster_probe_proj.py --device=3 --probe_base_weight="$weight" --n_clusters="$cluster" --num_heads="$num_head" --alpha="$a" > "./logs/valid_2_fold_llama_7B_cluster${cluster}_probe_proj_heads${num_head}_alpha${a}_baseW${weight//.}.log" 2>&1 &
                
                # 等待上一个命令完成
                wait
            done
        done
    done
done
