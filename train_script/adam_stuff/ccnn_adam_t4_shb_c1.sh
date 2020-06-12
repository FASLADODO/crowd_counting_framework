task="ccnn_adam_t4_shb_c1"

CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "adam lr and decay, 8"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_fixed_sigma/part_B  \
--load_model  "saved_model_best/ccnn_adam_t4_shb/ccnn_adam_t4_shb_checkpoint_valid_mae=-22.762274026870728.pth" \
--lr 1e-6 \
--decay 1e-6  \
--loss_fn "MSEMean" \
--skip_train_eval \
--batch_size 8 \
--optim  "adam" \
--datasetname shanghaitech_more_random \
--epochs 2000 > logs/$task.log  &

echo logs/$task.log  # for convenience
