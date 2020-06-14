task="bigtail4_t14_shb_fixed"

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "bigtail4"  \
--model "BigTail4" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_fixed_sigma/part_B  \
--lr 5e-5 \
--decay 1e-4  \
--loss_fn "MSE" \
--skip_train_eval \
--batch_size 20 \
--datasetname shanghaitech_more_random \
--epochs 600 > logs/$task.log  &

echo logs/$task.log  # for convenience
