task="H1_t4_shb"

CUDA_VISIBLE_DEVICES=2 HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_meow_main.py  \
--task_id $task  \
--note "a continue train from sha"  \
--model "H1" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 2e-5 \
--decay 1e-4 \
--batch_size 16 \
--load_model saved_model/H1_t2_sha/H1_t2_sha_checkpoint_840000.pth \
--datasetname shanghaitech_rnd \
--epochs 2501 > logs/$task.log  &

echo logs/$task.log  # for convenience
