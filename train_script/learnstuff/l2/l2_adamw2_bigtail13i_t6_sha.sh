task="l2_adamw2_bigtail13i_t6_sha"

CUDA_VISIBLE_DEVICES=4 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "keepfull with lr scheduler, starting -3"  \
--model "BigTail13i" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-3 \
--lr_scheduler \
--step_list 30,50,70 \
--lr_list 1e-3,1e-4,1e-5 \
--decay 1e-5 \
--loss_fn "MSEL1Mean" \
--datasetname shanghaitech_keepfull_r50 \
--optim adamw \
--cache \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience

#shanghaitech_keepfull_r50