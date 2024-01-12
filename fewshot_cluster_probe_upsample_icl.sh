#!/bin/bash


alpha=(13)
probe_base_weights=(0.0 0.1 0.2 0.3 0.4)
n_clusters=(3)
num_heads=(24)
cut_rates=(0.9)

for c in "${cut_rates[@]}"; do
    for a in "${alpha[@]}"; do
        for num_head in "${num_heads[@]}"; do
            for weight in "${probe_base_weights[@]}"; do
                # 内循环遍历 n_clusters
                for cluster in "${n_clusters[@]}"; do
                    # 显示正在执行的命令
                    echo "Running: fewshot_llama_7B_cluster${cluster}_probe_cut${c}_heads${num_head}_alpha${a}_baseW${weight//.}_icl"
<<<<<<< HEAD
<<<<<<< HEAD:fewshot_cluster_probe_upsample.sh
                    nohup python -u fewshot_cluster_probe_upsample.py --method='icl' --device=0 --probe_type=prob --probe_base_weight="$weight" --n_clusters="$cluster" --num_heads="$num_head" --alpha="$a" --cut_rate="$c" > "./logs/fewshot_llama_7B_cluster${cluster}_probe_cut${c}_heads${num_head}_alpha${a}_baseW${weight//.}_prob_icl.log" 2>&1 &
=======
                    nohup python -u fewshot_cluster_probe_upsample.py --method='icl' --device=1 --probe_type=prob --probe_base_weight="$weight" --n_clusters="$cluster" --num_heads="$num_head" --alpha="$a" --cut_rate="$c" > "./logs/fewshot_llama_7B_cluster${cluster}_probe_cut${c}_heads${num_head}_alpha${a}_baseW${weight//.}_prob_icl.log" 2>&1 &
>>>>>>> c88fd54bad25f970334ccecd08242e1fa3f7c00b:fewshot_cluster_probe_upsample_icl.sh
=======
                    nohup python -u fewshot_cluster_probe_upsample.py --method='icl' --device=0 --probe_type=prob --probe_base_weight="$weight" --n_clusters="$cluster" --num_heads="$num_head" --alpha="$a" --cut_rate="$c" > "./logs/fewshot_llama_7B_cluster${cluster}_probe_cut${c}_heads${num_head}_alpha${a}_baseW${weight//.}_prob_icl.log" 2>&1 &
>>>>>>> ea95241fd5f3e3898120cfe14dd2ffc1eefa8510
                    wait

                    # echo "Running: fewshot_llama_7B_cluster${cluster}_probe_cut${c}_heads${num_head}_alpha${a}_baseW${weight//.}_none"
                    # nohup python -u fewshot_cluster_probe_upsample.py --method='none' --device=0 --probe_type=prob --probe_base_weight="$weight" --n_clusters="$cluster" --num_heads="$num_head" --alpha="$a" --cut_rate="$c" > "./logs/fewshot_llama_7B_cluster${cluster}_probe_cut${c}_heads${num_head}_alpha${a}_baseW${weight//.}_prob_none.log" 2>&1 &
                    # wait
                done
            done
        done
    done
done