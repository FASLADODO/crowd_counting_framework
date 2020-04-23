CUDA_VISIBLE_DEVICES=5 HTTPS_PROXY="http://10.30.58.36:81" nohup python train_compact_cnn.py  \
--task_id ccnn_v2_t6_sha_c2  \
--note "ccnnv2 with decay and 20p augment, much lower lr, load model"  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-5 \
--decay 1e-5 \
--load_model saved_model/ccnn_v2_t6_sha/ccnn_v2_t6_sha_checkpoint_601200.pth \
--datasetname shanghaitech_20p \
--epochs 1202 > logs/ccnn_v2_t6_sha_c2.log  &