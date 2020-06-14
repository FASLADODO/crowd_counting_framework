task="ccnn_v7_t12_sha"

CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=5 HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "MSE mean with sha"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 1e-4 \
--loss_fn "L1" \
--datasetname shanghaitech_20p_random \
--epochs 601 > logs/$task.log  &

echo logs/$task.log  # for convenience