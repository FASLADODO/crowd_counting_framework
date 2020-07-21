task="g1_ccnn_v7i_t3_sha"

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "mse l1 sum, with -3 lr and decay"  \
--model "CompactCNNV7i" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_A  \
--lr 1e-3 \
--decay 1e-3 \
--loss_fn "MSEL1Sum" \
--datasetname shanghaitech_flip_only \
--skip_train_eval \
--cache \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience