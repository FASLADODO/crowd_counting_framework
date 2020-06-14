task="H2_t8_sha"

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=2 PYTHONWARNINGS="ignore" HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "a H2 with L1, hope better than baseline"  \
--model "H2" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 1e-4 \
--loss_fn "L1" \
--skip_train_eval \
--datasetname shanghaitech_crop_random \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience
