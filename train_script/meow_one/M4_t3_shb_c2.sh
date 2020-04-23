task="M4_t3_shb_c2"

CUDA_VISIBLE_DEVICES=4 HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_meow_main.py  \
--task_id $task  \
--note "M4 continue"  \
--model "M4" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 1e-5 \
--decay 1e-5 \
--batch_size 8 \
--load_model \
--datasetname shanghaitech_rnd \
--epochs 701 > logs/$task.log  &

echo logs/$task.log  # for convenience
