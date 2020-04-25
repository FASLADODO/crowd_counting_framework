CUDA_VISIBLE_DEVICES=6 HTTPS_PROXY="http://10.60.28.99:86" nohup python train_compact_cnn.py  \
--task_id ccnn_v2_t10_sha  \
--note "ccnnv2 keep full, try no_norm to see how it work, with e-5 lr"  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--no_norm \
--lr 1e-5 \
--decay 1e-5 \
--datasetname shanghaitech_keepfull \
--epochs 1500 > logs/ccnn_v2_t10_sha.log  &