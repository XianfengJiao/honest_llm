#!/bin/bash

# 定义 probe_base_weight 和 n_clusters 的数组
alpha=(24 32 48)
layers=(25 30 15 10 5)

# 外循环遍历 probe_base_weight

for a in "${alpha[@]}"; do
    for l in "${layers[@]}"; do
        # 显示正在执行的命令
        echo "Running: valid_2_fold_llama_7B_alpha${a}_layer${l}"

        # 构建并执行命令
        nohup python -u valid_2_fold_mean_centring.py --model_name=llama_7B --alpha="$a" --select_layer="$l" > "./logs/mean_centring_valid_2_fold_llama_7B_alpha${a}_layer${l}.log" 2>&1 &
        
        # 等待上一个命令完成
        wait
    done
done
