CUDA_VISIBLE_DEVICES=5 HTTPS_PROXY="http://10.30.58.36:81" nohup python train_custom_compact_cnn.py  \
--task_id custom_ccnn_v2_t3_sha  \
--note "reproduce ccnn v2 with 86 mae" \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 1e-4 \
--datasetname shanghaitech_keepfull \
--epochs 602 > logs/custom_ccnn_v2_t3_sha.log  &