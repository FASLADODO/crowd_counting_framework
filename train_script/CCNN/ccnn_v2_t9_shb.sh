CUDA_VISIBLE_DEVICES=2 HTTPS_PROXY="http://10.30.58.36:81" nohup python train_compact_cnn.py  \
--task_id ccnn_v2_t9_shb  \
--note "ccnnv2 with decay, 1e-4 lr "  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 1e-4 \
--decay 1e-4 \
--batch_size 8 \
--datasetname shanghaitech_rnd \
--epochs 601 > logs/ccnn_v2_t9_shb.log  &