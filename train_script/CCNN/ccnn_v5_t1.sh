CUDA_VISIBLE_DEVICES=6 HTTPS_PROXY="http://10.30.58.36:81" nohup python train_compact_cnn.py  \
--task_id ccnn_v5_t1  \
--note "train CompactCNNV2 (fixed by adding 1 layer of cnn) with fixed lr 1e-5 no decay on keepfull and crop dataset part_A"  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-5 \
--decay 0 \
--datasetname shanghaitech_keepfull_and_crop \
--epochs 402 > logs/ccnn_v5_t1.log  &