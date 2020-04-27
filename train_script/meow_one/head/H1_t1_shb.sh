task="H1_t1_shb"

CUDA_VISIBLE_DEVICES=6 HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_meow_main.py  \
--task_id $task  \
--note "a"  \
--model "H1" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 1e-4 \
--decay 1e-4 \
--batch_size 8 \
--datasetname shanghaitech_keepfull \
--epochs 401 > logs/$task.log  &

echo logs/$task.log  # for convenience
