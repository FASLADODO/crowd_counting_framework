CUDA_VISIBLE_DEVICES=5 nohup python train_compact_cnn_lrscheduler.py  \
--task_id ccnn_v2_t1  \
--input /data/rnd/thient/thient_data/ShanghaiTech/part_A  \
--lr 1e-4 \
--decay 5e-4 \
--datasetname shanghaitech \
--epochs 400 > logs/ccnn_v2_t1.log  &