task="g1_BigTail12i_t10_shb"

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "mse l1 sum, sgd -3, use batchnorm (default setting) sgd mean loss"  \
--model "BigTail12i" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_B  \
--lr 1e-3 \
--decay 1e-3 \
--loss_fn "MSEL1Mean" \
--batch_size 5 \
--datasetname shanghaitech_non_overlap \
--skip_train_eval \
--optim sgd \
--cache \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience