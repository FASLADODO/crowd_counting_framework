task="g1_BigTail14i_t3_sha"

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "mse l1 sum, with -5 lr and -5 decay (help overfit), flip only, sgd"  \
--model "BigTail14i" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_A  \
--lr 1e-4 \
--decay 1e-4 \
--loss_fn "MSEL1Mean" \
--datasetname shanghaitech_flip_only \
--skip_train_eval \
--optim sgd \
--cache \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience