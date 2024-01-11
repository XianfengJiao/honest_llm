#!/bin/bash

model_names=('llama_7B' 'llama2_7B' 'llama2_chat_7B' 'alpaca_7B' 'vicuna_7B')  # 添加模型名称数组


for model_name in "${model_names[@]}"; do

    # 构建并执行命令，添加 --model_name 参数
    nohup python -u collect_activations.py --model_name="$model_name" > "./logs/collect/collect_${model_name}_activations.log" 2>&1 &
    
    # 等待上一个命令完成
    wait
done
