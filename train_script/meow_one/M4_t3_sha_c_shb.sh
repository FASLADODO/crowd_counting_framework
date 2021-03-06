task="M4_t3_sha_c_shb"

CUDA_VISIBLE_DEVICES=4 HTTPS_PROXY="http://10.60.28.99:86" nohup python experiment_meow_main.py  \
--task_id $task  \
--note "same M4_t2_sha_c_shb but with batch"  \
--model "M4" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 1e-4 \
--decay 1e-4 \
--batch_size 8 \
--load_model saved_model/M4_t2_sha_coun/M4_t2_sha_coun_checkpoint_960000.pth \
--datasetname shanghaitech_rnd \
--epochs 1201 > logs/$task.log  &

echo logs/$task.log  # for convenience
