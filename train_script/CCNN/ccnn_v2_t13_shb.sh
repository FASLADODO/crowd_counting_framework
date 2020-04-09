CUDA_VISIBLE_DEVICES=6 HTTPS_PROXY="http://10.30.58.36:81" nohup python train_compact_cnn.py  \
--task_id ccnn_v2_t13_shb  \
--note "ccnnv2 with decay, 1e-5 lr "  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 5e-5 \
--decay 1e-5 \
--batch_size 8 \
--load_model saved_model/ccnn_v2_t4_sha/ccnn_v2_t4_sha_checkpoint_597600.pth  \
--datasetname shanghaitech_rnd \
--epochs 1201 > logs/ccnn_v2_t13_shb.log  &