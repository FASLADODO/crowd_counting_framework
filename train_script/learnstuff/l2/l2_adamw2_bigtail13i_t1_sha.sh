task="l2_adamw2_bigtail13i_t1_sha"

CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "adamW with -3 lr and decay, msel1mean"  \
--model "BigTail13i" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-3 \
--decay 1e-3 \
--loss_fn "MSEL1Mean" \
--datasetname shanghaitech_crop_random \
--optim adamw \
--cache \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience