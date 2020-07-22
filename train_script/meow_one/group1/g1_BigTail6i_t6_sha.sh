task="g1_BigTail6i_t6_sha"

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "mse l1 sum, with -4 lr and -1 decay (more help overfit), flip only"  \
--model "BigTail6i" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_A  \
--lr 1e-4 \
--decay 1e-1 \
--loss_fn "MSEL1Sum" \
--datasetname shanghaitech_flip_only \
--skip_train_eval \
--cache \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience