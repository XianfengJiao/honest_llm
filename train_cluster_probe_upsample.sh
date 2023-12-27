#!/bin/bash

# 定义 probe_base_weight 和 n_clusters 的数组
alpha=(15 10)
probe_base_weights=(0)
n_clusters=(3 4)
num_heads=(48 24)
cut_rates=(0.75 0.9 0.5)

# 外循环遍历 probe_base_weight
for c in "${cut_rates[@]}"; do
    for a in "${alpha[@]}"; do
        for num_head in "${num_heads[@]}"; do
            for weight in "${probe_base_weights[@]}"; do
                # 内循环遍历 n_clusters
                for cluster in "${n_clusters[@]}"; do
                    # 显示正在执行的命令
                    echo "Running: valid_2_fold_llama_7B_cluster${cluster}_probe_cut${c}_heads${num_head}_alpha${a}_baseW${weight//.}"

                    # 构建并执行命令
                    nohup python -u valid_2_fold_cluster_probe_upsample.py --device=0 --probe_base_weight="$weight" --n_clusters="$cluster" --num_heads="$num_head" --alpha="$a" --cut_rate="$c" > "./logs/valid_2_fold_llama_7B_cluster${cluster}_probe_cut${c}_heads${num_head}_alpha${a}_baseW${weight//.}.log" 2>&1 &
                    
                    # 等待上一个命令完成
                    wait
                done
            done
        done
    done
done
