CUDA_VISIBLE_DEVICES=6 HTTPS_PROXY="http://10.30.58.36:81" nohup python train_compact_cnn.py  \
--task_id ccnn_v5_t1_shb  \
--note "train CompactCNNV2 with shb on batchsize of 5"  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 1e-5 \
--decay 0 \
--datasetname shanghaitech_keepfull \
--batch_size 5 \
--epochs 402 > logs/ccnn_v5_t1_shb.log  &