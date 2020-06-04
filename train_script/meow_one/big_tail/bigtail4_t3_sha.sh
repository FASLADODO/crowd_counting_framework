task="bigtail4_t3_sha"

CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "bigtail4"  \
--model "BigTail4" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 5e-5 \
--decay 1e-4  \
--loss_fn "MSE" \
--skip_train_eval \
--datasetname shanghaitech_crop_random \
--epochs 800 > logs/$task.log  &

echo logs/$task.log  # for convenience
