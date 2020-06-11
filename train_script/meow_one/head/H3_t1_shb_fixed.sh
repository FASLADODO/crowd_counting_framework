task="H3_t1_shb_fixed"

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=5 HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "shb sigma 15"  \
--model "H3" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_fixed_sigma/part_B  \
--lr 1e-5 \
--decay 1e-5 \
--loss_fn "MSEMean" \
--optim "adam"  \
--batch_size 8 \
--datasetname shanghaitech_rnd \
--epochs 1200 > logs/$task.log  &

echo logs/$task.log  # for convenience
