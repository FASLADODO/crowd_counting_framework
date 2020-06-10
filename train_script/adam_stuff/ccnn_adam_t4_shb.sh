task="ccnn_adam_t4_shb"

CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "adam lr and decay, 8"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_fixed_sigma/part_B  \
--lr 1e-5 \
--decay 1e-5  \
--loss_fn "MSEMean" \
--skip_train_eval \
--batch_size 20 \
--optim  "adam" \
--datasetname shanghaitech_more_random \
--epochs 1200 > logs/$task.log  &

echo logs/$task.log  # for convenience
