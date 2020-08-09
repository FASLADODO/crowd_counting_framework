task="l2_adamw2_bigtail11i_t2_sha"

CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "11i no batchnorm keepfull with lr scheduler, starting -3"  \
--model "BigTail11i" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-3 \
--lr_scheduler \
--step_list 20,50,70,500,1000 \
--lr_list 1e-3,2e-4,1e-4,1e-4,1e-5 \
--decay 1e-5 \
--loss_fn "MSEL1Mean" \
--datasetname shanghaitech_keepfull_r50 \
--optim adamw \
--cache \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience

#shanghaitech_keepfull_r50