nohup python valid_2_fold_cluster_only.py --alpha 5 --n_clusters 3 > logs/cluster/num_heads48_alpha5_n_clusters3_std.log 2>&1 &
wait
nohup python valid_2_fold_cluster_only.py --alpha 5 --n_clusters 3 --num_heads 12 > logs/cluster/num_heads12_alpha5_n_clusters3_std.log 2>&1 &
wait
nohup python valid_2_fold_cluster_only.py --alpha 5 --n_clusters 3 --num_heads 64 > logs/cluster/num_heads64_alpha5_n_clusters3_std.log 2>&1 &
wait
