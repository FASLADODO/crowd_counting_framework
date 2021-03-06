task="bigtail3_t1_sha_shb"

CUDA_VISIBLE_DEVICES=2 HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_meow_main.py  \
--task_id $task  \
--note "bigtail3 sha 20p then shb rnd crop"  \
--model "BigTail3" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 1e-4 \
--decay 1e-4 \
--batch_size 8 \
--load_model saved_model/bigtail3_t1_sha/bigtail3_t1_sha_checkpoint_720000.pth \
--datasetname shanghaitech_rnd \
--epochs 1002 > logs/$task.log  &

echo logs/$task.log  # for convenience
