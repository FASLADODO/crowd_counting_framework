task="bigtail3_t1_sha"

CUDA_VISIBLE_DEVICES=2 HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_meow_main.py  \
--task_id $task  \
--note "bigtail3 shanghaitech_rnd"  \
--model "BigTail3" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 1e-4 \
--datasetname shanghaitech_20p \
--epochs 602 > logs/$task.log  &

echo logs/$task.log  # for convenience
