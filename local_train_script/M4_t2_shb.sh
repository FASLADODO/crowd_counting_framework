task="local_M4_t2_shb_3"

nohup python experiment_meow_main.py  \
--task_id $task  \
--note "M4 shanghaitech_rnd"  \
--model "M4" \
--input /data/ShanghaiTech/part_B  \
--lr 1e-4 \
--decay 1e-4 \
--batch_size 6 \
--datasetname shanghaitech_rnd \
--epochs 301 > logs/$task.log  &

echo logs/$task.log  # for convenience
