task="H1_t3_shb"

CUDA_VISIBLE_DEVICES=1 HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_meow_main.py  \
--task_id $task  \
--note "a continue train from sha"  \
--model "H1" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 1e-4 \
--decay 1e-4 \
--batch_size 8 \
--load_model saved_model/H1_t2_sha/H1_t2_sha_checkpoint_840000.pth \
--datasetname shanghaitech_rnd \
--epochs 2001 > logs/$task.log  &

echo logs/$task.log  # for convenience
