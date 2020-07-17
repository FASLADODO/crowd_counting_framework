task="g1_BigTail6_t5_shb"

CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "try SGD instead of adam, lr -6, with mean"  \
--model "BigTail6" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_B  \
--lr 1e-6 \
--decay 1e-6 \
--batch_size 4 \
--loss_fn "MSEL1Mean" \
--datasetname shanghaitech_non_overlap \
--skip_train_eval \
--cache \
--optim sgd \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience