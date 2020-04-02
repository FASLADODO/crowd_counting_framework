task="m1_t1_shb"

CUDA_VISIBLE_DEVICES=3 HTTPS_PROXY="http://10.30.58.36:81" nohup python experiment_meow_main.py  \
--task_id $task  \
--note "m1 shb shanghaitech_keepfull with decay, change the head, keep the tail "  \
--model "M1" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 1e-4 \
--decay 1e-4 \
--datasetname shanghaitech_keepfull \
--epochs 300 > logs/$task.log  &

echo logs/$task.log  # for convenience