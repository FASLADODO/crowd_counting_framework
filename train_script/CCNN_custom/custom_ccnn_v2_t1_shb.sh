CUDA_VISIBLE_DEVICES=6 HTTPS_PROXY="http://10.30.58.36:81" nohup python train_custom_compact_cnn.py  \
--task_id custom_ccnn_v2_t1_shb  \
--note "train custom ccnn v1 branching 3 branch"  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_B  \
--lr 1e-4 \
--decay 0 \
--datasetname shanghaitech_keepfull \
--batch_size 5 \
--epochs 502 > logs/custom_ccnn_v2_t1_shb.log  &