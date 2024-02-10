#!/bin/bash
alpha=(15)
probe_base_weights=(0)

n_clusters=(3)
num_heads=(16)
train_rate=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
seed=5
model=vicuna_7B

echo "Running: analyze_data_size_${model}_seed${seed}_pure"
nohup python -u data_size_analysis_act.py --seed=${seed} --pure > "./logs/0211_${model}_data_size_analysis_pure_seed${seed}.log" 2>&1 &
wait

for a in "${alpha[@]}"; do
    for num_head in "${num_heads[@]}"; do
        for weight in "${probe_base_weights[@]}"; do
            # 内循环遍历 n_clusters
            for cluster in "${n_clusters[@]}"; do
                for rate in "${train_rate[@]}"; do
                    # 显示正在执行的命令
                    echo "Running: analyze_data_size_${model}_cluster${cluster}_probe_heads${num_head}_alpha${a}_baseW${weight//.}_train${rate}_seed${seed}"

                    # 构建并执行命令
                    nohup python -u data_size_analysis_act.py --model_name=${model} --seed=${seed} --train_ratio="$rate" --probe_type=prob --probe_base_weight="$weight" --n_clusters="$cluster" --num_heads="$num_head" --alpha="$a" > "./logs/0211_seed${seed}_analyze_data_size_${model}_cluster${cluster}_probe_heads${num_head}_alpha${a}_baseW${weight//.}_prob_train${rate}.log" 2>&1 &
                    
                    # 等待上一个命令完成
                    wait
                done
            done
        done
    done
done
