task="g1_ccnn_v7_t1_shb"

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "mse l1 mean, with -4 lr and decay"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_B  \
--lr 1e-4 \
--decay 1e-4 \
--batch_size 4 \
--loss_fn "MSEL1Mean" \
--datasetname shanghaitech_non_overlap \
--skip_train_eval \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience