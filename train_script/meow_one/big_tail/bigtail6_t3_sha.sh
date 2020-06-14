task="bigtail6_t3_sha"

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "a, default 1e-5, keepfull"  \
--model "BigTail6" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-5 \
--decay 1e-5  \
--loss_fn "MSEMean" \
--skip_train_eval \
--optim "adam" \
--batch_size 1 \
--datasetname shanghaitech_keepfull \
--epochs 601 > logs/$task.log  &

echo logs/$task.log  # for convenience
