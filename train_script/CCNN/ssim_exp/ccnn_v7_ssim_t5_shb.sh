task="ccnn_v7_ssim_t5_shb"

CUDA_VISIBLE_DEVICES=2 HTTPS_PROXY="http://10.60.28.99:86" nohup python train_compact_cnn.py  \
--task_id $task  \
--note "1e-5 keepfull"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 1e-5 \
--decay 1e-5 \
--use_ssim  \
--batch_size 8 \
--load_model saved_model/ccnn_v7_t5_shb/ccnn_v7_t5_shb_checkpoint_80000.pth \
--datasetname shanghaitech_keepfull \
--epochs 601 > logs/$task.log  &

echo logs/$task.log  # for convenience