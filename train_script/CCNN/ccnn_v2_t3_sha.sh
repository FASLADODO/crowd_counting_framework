CUDA_VISIBLE_DEVICES=3 HTTPS_PROXY="http://10.30.58.36:81" nohup python train_compact_cnn.py  \
--task_id ccnn_v2_t3_sha  \
--note "ccnnv2 no decay"  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 0 \
--datasetname shanghaitech_keepfull \
--epochs 502 > logs/ccnn_v2_t3_sha.log  &