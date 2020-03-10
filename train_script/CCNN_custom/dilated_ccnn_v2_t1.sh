CUDA_VISIBLE_DEVICES=5 nohup python train_custom_compact_cnn.py  \
--task_id dilated_ccnn_v2_t1  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-5 \
--decay 5e-5 \
--datasetname shanghaitech \
--epochs 400 > logs/dilated_ccnn_v2_t1.log  &