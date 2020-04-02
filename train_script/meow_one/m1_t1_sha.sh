task="m1_t1_sha"

CUDA_VISIBLE_DEVICES=1 HTTPS_PROXY="http://10.30.58.36:81" nohup python experiment_meow_main.py  \
--task_id $task  \
--note "m1 sha shanghaitech_keepfull with decay, change the head, keep the tail "  \
--model "M1" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 1e-4 \
--datasetname shanghaitech_keepfull \
--epochs 502 > logs/$task.log  &