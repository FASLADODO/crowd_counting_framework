task="g1_H3i_t1_sha"

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "mse l1 sum, with -5 lr and decay"  \
--model "H3i" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_A  \
--lr 1e-4 \
--decay 1e-4 \
--loss_fn "MSEL1Sum" \
--datasetname shanghaitech_non_overlap \
--skip_train_eval \
--cache \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience