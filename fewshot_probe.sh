#!/bin/bash

echo "Running: fewshot_llama7B_probe_alpha15_prob_head48_icl"
nohup python -u fewshot_probe.py --method='icl' --alpha=15 --num_heads=48  > ./logs/fewshot_llama7B_probe_alpha15_prob_head48_icl.log 2>&1 &
wait

echo "Running: fewshot_llama7B_probe_alpha15_prob_head48_none"
nohup python -u fewshot_probe.py --method='none' --alpha=15 --num_heads=48  > ./logs/fewshot_llama7B_probe_alpha15_prob_head48_none.log 2>&1 &
wait

echo "Running: fewshot_llama7B_probe_alpha15_mean_head48_icl"
nohup python -u fewshot_probe.py --method='icl' --alpha=20 --num_heads=48 --use_center_of_mass  > ./logs/fewshot_llama7B_probe_alpha15_mean_head48_icl.log 2>&1 &
wait

echo "Running: fewshot_llama7B_probe_alpha15_mean_head48_none"
nohup python -u fewshot_probe.py --method='none' --alpha=20 --num_heads=48 --use_center_of_mass  > ./logs/fewshot_llama7B_probe_alpha15_mean_head48_none.log 2>&1 &
wait