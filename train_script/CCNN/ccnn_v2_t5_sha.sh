CUDA_VISIBLE_DEVICES=6 HTTPS_PROXY="http://10.30.58.36:81" nohup python train_compact_cnn.py  \
--task_id ccnn_v2_t5_sha  \
--note "ccnnv2 with decay and 20p augment, lower lr"  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 5e-5 \
--decay 1e-4 \
--datasetname shanghaitech_20p \
--epochs 502 > logs/ccnn_v2_t5_sha.log  &