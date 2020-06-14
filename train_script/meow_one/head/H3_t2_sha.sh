task="H3_t2_sha"

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "h3 with sha, add 3x3"  \
--model "H3" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-5 \
--decay 1e-5 \
--loss_fn "L1Mean" \
--optim "adam"  \
--skip_train_eval \
--datasetname shanghaitech_crop_random \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience
