task="H2_t3_shb_fixed"

CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=5 HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "shb sigma 15"  \
--model "H2" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_fixed_sigma/part_B  \
--lr 1e-4 \
--decay 1e-4 \
--loss_fn "L1" \
--batch_size 20 \
--datasetname shanghaitech_rnd \
--epochs 601 > logs/$task.log  &

echo logs/$task.log  # for convenience
