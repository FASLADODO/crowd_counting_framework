task="adamw2_bigtail13i_t1_sha"

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "adamW with -3 lr and -5 decay (small decay for small batch size) , msel1mean"  \
--model "BigTail13i" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_A  \
--lr 1e-5 \
--decay 1e-5 \
--loss_fn "MSEL1Sum" \
--datasetname shanghaitech_crop_random \
--optim adamw \
--cache \
--epochs 1501 > logs/$task.log  &

echo logs/$task.log  # for convenience