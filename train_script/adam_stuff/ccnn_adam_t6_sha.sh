task="ccnn_adam_t5_sha"

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "lr 1e-4 with L1Mean"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 1e-4  \
--loss_fn "L1Mean" \
--skip_train_eval \
--batch_size 1 \
--optim  "adam" \
--datasetname shanghaitech_crop_random \
--epochs 1200 > logs/$task.log  &

echo logs/$task.log  # for convenience
