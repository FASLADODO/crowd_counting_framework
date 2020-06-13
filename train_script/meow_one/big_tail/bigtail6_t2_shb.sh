task="bigtail6_t2_shb_fixed"

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "b"  \
--model "BigTail6" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_fixed_sigma/part_B  \
--lr 5e-5 \
--decay 1e-4  \
--loss_fn "MSEMean" \
--skip_train_eval \
--optim "adam" \
--batch_size 8 \
--datasetname shanghaitech_more_random \
--epochs 2000 > logs/$task.log  &

echo logs/$task.log  # for convenience
