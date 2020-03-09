CUDA_VISIBLE_DEVICES=4 nohup python train_custom_compact_cnn.py  \
--task_id defdilated_ccnn_v1_t1  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-5 \
--decay 5e-5 \
--datasetname shanghaitech \
--epochs 500 > logs/defdilated_ccnn_v1_t1.log  &