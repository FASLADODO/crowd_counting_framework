task="ccnn_v7_t19_sha"

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=5 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "MSE mean with sha"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 4e-5 \
--decay 4e-5 \
--loss_fn "MSE" \
--datasetname shanghaitech_crop_random \
--epochs 1200 > logs/$task.log  &

echo logs/$task.log  # for convenience