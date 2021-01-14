task="adamw1_csrnet_t8_shb"

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "adamW csrnet batchnorm and relu, batchnorm only tail, freez vggg, now, go higher lr"  \
--model "CSRNet" \
--input /data/rnd/thient/thient_data/shanghaitech_with_people_density_map/ShanghaiTech_3/part_B  \
--lr 1e-6 \
--decay 0.0005 \
--loss_fn "MSEL1Mean" \
--batch_size 2 \
--datasetname shanghaitech_non_overlap \
--optim adamw \
--cache \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience