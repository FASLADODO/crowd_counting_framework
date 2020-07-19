task="g1_ccnn_v7i_t3_shb"

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "mse l1 sum, with -4 lr and decay with 4 L1 "  \
--model "CompactCNNV7i" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_B  \
--lr 1e-4 \
--decay 1e-4 \
--batch_size 5 \
--loss_fn "MSE4L1Sum" \
--datasetname shanghaitech_non_overlap \
--skip_train_eval \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience