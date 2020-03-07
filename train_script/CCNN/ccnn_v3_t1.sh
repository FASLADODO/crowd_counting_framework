CUDA_VISIBLE_DEVICES=6 nohup python train_compact_cnn.py  \
--task_id ccnn_v3_t1  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-5 \
--decay 5e-5 \
--datasetname shanghaitech_keepfull \
--epochs 401 > logs/ccnn_v1_t1.log  &