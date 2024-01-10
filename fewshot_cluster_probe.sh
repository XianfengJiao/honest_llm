#!/bin/bash

echo "Running: fewshot_llama7B_pure_none"
nohup python -u fewshot_cluster_probe.py --pure --method='none' > ./logs/fewshot_llama7B_pure_none.log 2>&1 &
wait

echo "Running: fewshot_llama7B_pure_icl"
nohup python -u fewshot_cluster_probe.py --pure --method='icl' > ./logs/fewshot_llama7B_pure_icl.log 2>&1 &
wait

echo "Running: fewshot_llama7B_cluster_probe_c3_alpha12_prob_baseW0_head24_icl"
nohup python -u fewshot_cluster_probe.py --method='icl' --alpha=12 --num_heads=24 --n_clusters=3  > ./logs/fewshot_llama7B_cluster_probe_c3_alpha12_prob_baseW0_head24_icl.log 2>&1 &
wait

echo "Running: fewshot_llama7B_cluster_probe_c3_alpha12_prob_baseW0_head24_none"
nohup python -u fewshot_cluster_probe.py --method='none' --alpha=12 --num_heads=24 --n_clusters=3  > ./logs/fewshot_llama7B_cluster_probe_c3_alpha12_prob_baseW0_head24_none.log 2>&1 &