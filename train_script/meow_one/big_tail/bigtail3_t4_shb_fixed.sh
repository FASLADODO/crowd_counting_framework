task="bigtail3_t4_shb_fixed"

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=5 HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_meow_main.py  \
--task_id $task  \
--note "bigtail3 L1"  \
--model "BigTail3" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_fixed_sigma/part_B  \
--lr 1e-4 \
--decay 1e-4 \
--batch_size 8 \
--loss_fn "L1" \
--datasetname shanghaitech_rnd \
--epochs 601 > logs/$task.log  &

echo logs/$task.log  # for convenience
