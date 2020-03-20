CUDA_VISIBLE_DEVICES=6 HTTPS_PROXY="http://10.30.58.36:81" nohup python train_custom_compact_cnn.py  \
--task_id custom_ccnn_v5_t4_sha  \
--note "defromable"  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 5e-5 \
--decay 1e-4 \
--datasetname shanghaitech \
--batch_size 1 \
--epochs 400 > logs/custom_ccnn_v5_t4_sha.log  &