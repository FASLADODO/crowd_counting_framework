task="M4_t2_sha_coun"

CUDA_VISIBLE_DEVICES=2 HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_meow_main.py  \
--task_id $task  \
--note "M4 20 percentage aug, continue train to 800 epochs"  \
--model "M4" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 1e-4 \
--load_model saved_model/M4_t2_sha/M4_t2_sha_checkpoint_360000.pth \
--datasetname shanghaitech_20p \
--epochs 801 > logs/$task.log  &

echo logs/$task.log  # for convenience
