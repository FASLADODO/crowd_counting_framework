task="H2_t5_sha"

CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=5 HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_main.py  \
--task_id $task  \
--note "a with new 20p random crop 40 percentage"  \
--model "H2" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 1e-4 \
--loss_fn "L1" \
--datasetname shanghaitech_60p_random \
--epochs 601 > logs/$task.log  &

echo logs/$task.log  # for convenience
