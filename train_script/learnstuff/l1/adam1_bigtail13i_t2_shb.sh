task="adam1_bigtail13i_t2_shb"

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "adam with extrem high lr and decay, msel1mean"  \
--model "BigTail13i" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_B  \
--lr 1e-3 \
--decay 0.1 \
--loss_fn "MSEL1Mean" \
--batch_size 5 \
--datasetname shanghaitech_non_overlap \
--cache \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience