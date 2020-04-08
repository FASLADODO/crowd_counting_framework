CUDA_VISIBLE_DEVICES=1 HTTPS_PROXY="http://10.30.58.36:81" nohup python train_compact_cnn.py  \
--task_id ccnn_v2_t7_sha  \
--note "ccnnv2 with decay and 20p augment, much lower lr"  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-5 \
--batch_size 8 \
--decay 1e-4 \
--datasetname shanghaitech_256 \
--epochs 1500 > logs/ccnn_v2_t7_sha.log  &