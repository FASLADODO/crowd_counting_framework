CUDA_VISIBLE_DEVICES=3 HTTPS_PROXY="http://10.60.28.99:86" nohup python train_compact_cnn.py  \
--task_id ccnn_v2_t8_sha  \
--note "ccnnv2 keep full"  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 1e-4 \
--datasetname shanghaitech_keepfull \
--epochs 1500 > logs/ccnn_v2_t8_sha.log  &