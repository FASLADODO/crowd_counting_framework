task="ccnnv9_adam_t3_sha_c1"

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "leaky relu"  \
--model "CompactCNNV9" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--load_model  "saved_model_best/ccnnv9_adam_t3_sha/ccnnv9_adam_t3_sha_checkpoint_valid_mae=-111.91341031392416.pth" \
--lr 1e-5 \
--decay 1e-5  \
--loss_fn "MSEMean" \
--skip_train_eval \
--batch_size 1 \
--optim  "adam" \
--datasetname shanghaitech_crop_random \
--epochs 2000 > logs/$task.log  &

echo logs/$task.log  # for convenience
