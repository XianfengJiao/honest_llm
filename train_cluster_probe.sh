#!/bin/bash

# 定义 probe_base_weight 和 n_clusters 的数组
probe_base_weights=(0 0.3 0.4 0.5)
n_clusters=(2 3 4)

# 外循环遍历 probe_base_weight
for weight in "${probe_base_weights[@]}"; do
    # 内循环遍历 n_clusters
    for cluster in "${n_clusters[@]}"; do
        # 显示正在执行的命令
        echo "Running: valid_2_fold_llama_7B_cluster${cluster}_probe_heads48_alpha15_baseW${weight//.}"

        # 构建并执行命令
        nohup python -u valid_2_fold_cluster_probe.py --device=0 --probe_base_weight="$weight" --n_clusters="$cluster" > "./logs/valid_2_fold_llama_7B_cluster${cluster}_probe_heads48_alpha15_baseW${weight//.}.log" 2>&1 &
        
        # 等待上一个命令完成
        wait
    done
done
