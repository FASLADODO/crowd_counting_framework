task="bigtail4_t6_shb_fixed"

CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "bigtail4"  \
--model "BigTail4" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_fixed_sigma/part_B    \
--lr 2e-5 \
--decay 1e-4  \
--loss_fn "MSE" \
--skip_train_eval \
--batch_size 10 \
--datasetname shanghaitech_more_random \
--epochs 800 > logs/$task.log  &

echo logs/$task.log  # for convenience
