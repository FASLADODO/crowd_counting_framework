task="adamw1_csrnet_t5_shb"

CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "adamW csrnet on best shb of dccnn"  \
--model "CSRNet" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_B  \
--lr 1e-6 \
--decay 0.0001 \
--loss_fn "MSEL1Mean" \
--batch_size 2 \
--datasetname shanghaitech_non_overlap \
--optim adamw \
--cache \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience