task="m3_t3_sha"

CUDA_VISIBLE_DEVICES=3 HTTPS_PROXY="http://10.30.58.36:81" nohup python experiment_meow_main.py  \
--task_id $task  \
--note "m3 sha shanghaitech_keepfull with decay, change the head, tail dilated. x10 decay and x10 lr "  \
--model "M3" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-3 \
--decay 1e-3 \
--datasetname shanghaitech_keepfull \
--epochs 502 > logs/$task.log  &

echo logs/$task.log  # for convenience