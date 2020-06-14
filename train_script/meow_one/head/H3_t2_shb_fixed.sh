task="H3_t2_shb_fixed"

CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=5 HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "shb sigma 15"  \
--model "H3" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_fixed_sigma/part_B  \
--lr 1e-5 \
--decay 1e-5 \
--loss_fn "L1Mean" \
--optim "adam"  \
--batch_size 8 \
--skip_train_eval \
--datasetname shanghaitech_more_random \
--epochs 1200 > logs/$task.log  &

echo logs/$task.log  # for convenience
