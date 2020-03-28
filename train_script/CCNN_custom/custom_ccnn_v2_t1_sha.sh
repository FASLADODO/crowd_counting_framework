CUDA_VISIBLE_DEVICES=2 HTTPS_PROXY="http://10.30.58.36:81" nohup python train_custom_compact_cnn.py  \
--task_id custom_ccnn_v2_t1_sha  \
--note "train custom ccnn v1 branching 3 branchn no batchnorm"  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 0 \
--datasetname shanghaitech_keepfull \
--epochs 502 > logs/custom_ccnn_v2_t1_sha.log  &