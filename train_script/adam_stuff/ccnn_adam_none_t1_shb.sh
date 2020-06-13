task="ccnn_adam_none_t1_shb.sh"

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "try reduction none"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_fixed_sigma/part_B  \
--lr 1e-5 \
--decay 1e-5  \
--loss_fn "MSENone" \
--skip_train_eval \
--batch_size 8 \
--optim  "adam" \
--datasetname shanghaitech_more_random \
--epochs 1001 > logs/$task.log  &

echo logs/$task.log  # for convenience
