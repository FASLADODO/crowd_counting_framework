CUDA_VISIBLE_DEVICES=5 nohup python train_compact_cnn.py  \
--task_id ccnn_v1_t1  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 5e-4 \
--datasetname shanghaitech \
--epochs 33 > logs/ccnn_v1_t1.log  &