#!/bin/bash

# 定义数组
alpha=(12)
probe_base_weights=(0)
n_clusters=(3)
num_heads=(24 32)
model_names=('llama2_7B' 'llama2_chat_7B')  # 添加模型名称数组

# 外层循环遍历 alpha
for a in "${alpha[@]}"; do
    for num_head in "${num_heads[@]}"; do
        for weight in "${probe_base_weights[@]}"; do
            for cluster in "${n_clusters[@]}"; do
                # 新增循环遍历模型名称
                for model_name in "${model_names[@]}"; do
                    # 显示正在执行的命令
                    echo "Running: valid_2_fold_${model_name}_cluster${cluster}_probe_heads${num_head}_alpha${a}_baseW${weight//.}"

                    # 构建并执行命令，添加 --model_name 参数
                    nohup python -u valid_2_fold_cluster_probe.py --probe_base_weight="$weight" --n_clusters="$cluster" --num_heads="$num_head" --alpha="$a" --model_name="$model_name" --device=3 --probe_type=mean > "./logs/version/valid_2_fold_${model_name}_cluster${cluster}_heads${num_head}_alpha${a}_baseW${weight//.}_end_q_mean.log" 2>&1 &
                    
                    # 等待上一个命令完成
                    wait
                done
            done
        done
    done
done
