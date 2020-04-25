CUDA_VISIBLE_DEVICES=2 HTTPS_PROXY="http://10.60.28.99:86" nohup python train_compact_cnn.py  \
--task_id ccnn_v2_t6_sha_c2_shb  \
--note "ccnn v2 sha then shb"  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 1e-4 \
--decay 1e-4 \
--load_model saved_model/ccnn_v2_t6_sha_c2/ccnn_v2_t6_sha_c2_checkpoint_1440000.pth \
--datasetname shanghaitech_rnd \
--epochs 1601 > logs/ccnn_v2_t6_sha_c2_shb.log  &