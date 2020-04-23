task="M4_t2_sha_shb"

CUDA_VISIBLE_DEVICES=3 HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_meow_main.py  \
--task_id $task  \
--note "M4 train sha 20p then shb random crop 1/4 "  \
--model "M4" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 1e-4 \
--load_model saved_model/M4_t2_sha/M4_t2_sha_checkpoint_360000.pth \
--datasetname shanghaitech_rnd \
--epochs 1000 > logs/$task.log  &

echo logs/$task.log  # for convenience
