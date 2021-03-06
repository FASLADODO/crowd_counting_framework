task="g1_BigTail15i_t3_shb"

CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "mse l1 sum, with -5 lr and -2 decay"  \
--model "BigTail15i" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_B  \
--lr 1e-5 \
--decay 1e-2 \
--loss_fn "MSEL1Mean" \
--datasetname shanghaitech_non_overlap \
--skip_train_eval \
--batch_size 5 \
--cache \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience