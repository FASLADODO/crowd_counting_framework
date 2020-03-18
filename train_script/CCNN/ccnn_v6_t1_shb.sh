CUDA_VISIBLE_DEVICES=1 HTTPS_PROXY="http://10.30.58.36:81" nohup python train_compact_cnn.py  \
--task_id ccnn_v6_t1_shb  \
--note "compact ccnnv6 with bunch of batchnorm, shanghaitech crop quarter "  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 1e-5 \
--decay 0 \
--datasetname shanghaitech \
--batch_size 8 \
--epochs 302 > logs/ccnn_v6_t1_shb.log  &