nohup python valid_2_fold_cluster_only.py --alpha 15 > logs/cluster/num_heads48_alpha15_n_clusters4.log 2>&1 &
wait
nohup python valid_2_fold_cluster_only.py --alpha 10 > logs/cluster/num_heads48_alpha10_n_clusters4.log 2>&1 &
wait
nohup python valid_2_fold_cluster_only.py --alpha 5 > logs/cluster/num_heads48_alpha5_n_clusters4.log 2>&1 &
wait

