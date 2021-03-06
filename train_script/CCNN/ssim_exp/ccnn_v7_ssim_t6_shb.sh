task="ccnn_v7_ssim_t6_shb"

CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 HTTPS_PROXY="http://10.60.28.99:86" nohup python train_compact_cnn.py  \
--task_id $task  \
--note "1e-5 shanghaitech_rnd"  \
--model "CompactCNNV7" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 1e-5 \
--decay 1e-5 \
--use_ssim  \
--batch_size 20 \
--load_model saved_model/ccnn_v7_t5_shb/ccnn_v7_t5_shb_checkpoint_80000.pth \
--datasetname shanghaitech_rnd \
--epochs 1200 > logs/$task.log  &

echo logs/$task.log  # for convenience