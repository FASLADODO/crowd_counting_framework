task="l2_adamw2_bigtail13i_t11_sha"

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "256, high decay, lr -5 and decay -4"  \
--model "BigTail13i" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-5 \
--decay 1e-4 \
--loss_fn "MSEL1Mean" \
--datasetname shanghaitech_256_v2 \
--optim adamw \
--batch_size 60  \
--skip_train_eval \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience

#shanghaitech_keepfull_r50