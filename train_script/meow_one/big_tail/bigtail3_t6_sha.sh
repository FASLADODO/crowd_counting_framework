task="bigtail3_t6_sha"

CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=5 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "bigtail3 L1"  \
--model "BigTail3" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 5e-5 \
--decay 5e-5 \
--loss_fn "MSE" \
--datasetname shanghaitech_crop_random \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience
