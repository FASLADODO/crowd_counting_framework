task="H1_t5_shb"

CUDA_VISIBLE_DEVICES=3 HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_meow_main.py  \
--task_id $task  \
--note "a continue train from sha"  \
--model "H1" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 2e-5 \
--decay 1e-4 \
--batch_size 20 \
--datasetname shanghaitech_more_rnd \
--epochs 1001 > logs/$task.log  &

echo logs/$task.log  # for convenience
