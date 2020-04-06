CUDA_VISIBLE_DEVICES=3 HTTPS_PROXY="http://10.30.58.36:81" nohup python train_custom_compact_cnn.py  \
--task_id custom_ccnn_v2_t6_sha  \
--note "reproduce ccnn v2 with 86 mae, try must lower lr, do some aug" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-5 \
--decay 1e-4 \
--datasetname shanghaitech_20p \
--epochs 302 > logs/custom_ccnn_v2_t6_sha.log  &