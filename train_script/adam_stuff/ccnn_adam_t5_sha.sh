task="ccnn_adam_t6_sha"

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "adam lr 1e-4"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 1e-4  \
--loss_fn "MSEMean" \
--skip_train_eval \
--batch_size 1 \
--optim  "adam" \
--datasetname shanghaitech_crop_random \
--epochs 1200 > logs/$task.log  &

echo logs/$task.log  # for convenience
