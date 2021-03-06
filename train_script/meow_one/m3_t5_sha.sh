task="m3_t5_sha"

CUDA_VISIBLE_DEVICES=3 HTTPS_PROXY="http://10.30.58.36:81" nohup python experiment_meow_main.py  \
--task_id $task  \
--note "m3 sha shanghaitech_keepfull with decay, change the head, tail dilated, augment dataset"  \
--model "M3" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 1e-4 \
--datasetname shanghaitech_20p_enlarge \
--epochs 602 > logs/$task.log  &

echo logs/$task.log  # for convenience