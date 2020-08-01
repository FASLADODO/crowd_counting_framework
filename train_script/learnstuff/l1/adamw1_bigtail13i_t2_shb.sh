task="adamw1_bigtail13i_t2_shb"

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "adamW with -3 high lr and decay (we reduce decay from -1 to -3 because long training require less decay), msel1mean"  \
--model "BigTail13i" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_B  \
--lr 1e-3 \
--decay 1e-3 \
--loss_fn "MSEL1Mean" \
--batch_size 5 \
--datasetname shanghaitech_non_overlap \
--optim adamw \
--cache \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience