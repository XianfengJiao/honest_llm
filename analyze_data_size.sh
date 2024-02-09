#!/bin/bash
export CUDA_VISIBLE_DEVICES=6,7
# 定义 probe_base_weight 和 n_clusters 的数组
alpha=(12)
probe_base_weights=(0)
n_clusters=(3)
num_heads=(32)
train_rate=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
seed=42
init_seed=42

# 外循环遍历 probe_base_weight

for a in "${alpha[@]}"; do
    for num_head in "${num_heads[@]}"; do
        for weight in "${probe_base_weights[@]}"; do
            # 内循环遍历 n_clusters
            for cluster in "${n_clusters[@]}"; do
                for rate in "${train_rate[@]}"; do
                    # 显示正在执行的命令
                    echo "Running: analyze_data_size_llama_7B_cluster${cluster}_probe_heads${num_head}_alpha${a}_baseW${weight//.}_train${rate}"

                    # 构建并执行命令
                    nohup python -u data_size_analysis_act.py --init_seed=${init_seed} --seed=${seed} --train_ratio="$rate" --probe_type=prob --probe_base_weight="$weight" --n_clusters="$cluster" --num_heads="$num_head" --alpha="$a" > "./logs/0209_seed${seed}_analyze_data_size_llama_7B_cluster${cluster}_probe_heads${num_head}_alpha${a}_baseW${weight//.}_prob_train${rate}.log" 2>&1 &
                    
                    # 等待上一个命令完成
                    wait
                done
            done
        done
    done
done
