task="bigtail3_t2_shb"

CUDA_VISIBLE_DEVICES=3 HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_meow_main.py  \
--task_id $task  \
--note "bigtail3 shanghaitech_rnd e-5 lr"  \
--model "BigTail3" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 1e-5 \
--decay 1e-5 \
--batch_size 8 \
--datasetname shanghaitech_rnd \
--epochs 301 > logs/$task.log  &

echo logs/$task.log  # for convenience
