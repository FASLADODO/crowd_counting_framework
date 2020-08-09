task="l2_adamw2_bigtail11i_t1_c2_sha"

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "11i no batchnorm keepfull continue train, -4"  \
--model "BigTail11i" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 1e-4 \
--loss_fn "MSEL1Mean" \
--datasetname shanghaitech_keepfull_r50 \
--load_model saved_model/l2_adamw2_bigtail11i_t1_sha/l2_adamw2_bigtail11i_t1_sha_checkpoint_283200.pth \
--optim adamw \
--cache \
--epochs 2101 > logs/$task.log  &

echo logs/$task.log  # for convenience

#shanghaitech_keepfull_r50