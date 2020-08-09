task="l2_adamw2_bigtail11i_t2_c2_sha"

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "keepfull, lr to 5e-5, decay to e-6"  \
--model "BigTail11i" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 5e-5 \
--decay 1e-6 \
--loss_fn "MSEL1Mean" \
--datasetname shanghaitech_keepfull_r50 \
--load_model saved_model_best/l2_adamw2_bigtail11i_t2_sha/l2_adamw2_bigtail11i_t2_sha_checkpoint_valid_mae=-112.82307936350504.pth \
--optim adamw \
--cache \
--epochs 2401 > logs/$task.log  &

echo logs/$task.log  # for convenience

#shanghaitech_keepfull_r50