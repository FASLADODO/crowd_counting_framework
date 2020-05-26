task="ccnn_v7_t15_sha"

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=5 HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "MSE mean with sha"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 1e-4 \
--loss_fn "MSE" \
--datasetname shanghaitech_crop_random \
--epochs 1200 > logs/$task.log  &

echo logs/$task.log  # for convenience